"""
PMR-Fast：A+B+C 相位流形配准的低显存重实现（native 1mm 可在 8GB 上运行）
================================================================================
与 phase_proto/server/proto_s1_abcnn.py 数学等价或经消融验证的改动：

【O1 精确重构】傅里叶基在 **粗网格** 上合成，只把 3 通道位移上采样到全分辨率。
    原实现把 24 通道系数场上采样到全分辨率（1mm case8: 24ch x 79Mvox x 4B = 7.1GB，
    加梯度 14.2GB）——这正是 1mm OOM 的根因。
    由于 dvf 对系数是 **空间常数** 的线性组合，且三线性上采样是线性算子：
        interp( sum_k c_k * w_k(theta) ) == sum_k interp(c_k) * w_k(theta)
    故本改动是 **代数恒等**，不是近似。

【O2】仿射项用可广播的 1D 坐标向量，不再预分配 (3,D,H,W) 坐标张量
    （1mm case8 省 0.95GB 常驻）。

【O3】局部 NCC 用 stride 子采样求值（默认 stride=win//2），显存与算力大幅下降。
    属近似改动，需消融验证。

【O4】CNN 编码器输入按 enc_down 预降采样（默认 4），输出网格仍为 D/8。
    1mm case8 编码器首层卷积从 79Mvox 降到 1.2Mvox（约 64x 算力下降）。
    属近似改动，需消融验证。

【O6】周期残差系数改为粗网格（默认 D/4）而非全分辨率。
    原实现 1mm 残差参数 = 24ch x 79Mvox x 4B = 7.1GB，加梯度 + Adam 一/二阶矩
    共约 28GB——这是残差阶段在 1mm 必然 OOM 的原因。属近似改动，需消融验证。

【O8】landmark 处位移采样改为三线性插值（原实现取最近体素，2mm 下可引入
    ~0.5 体素量化误差）。同时输出两种口径以便与历史数字对照。

【O9】10 个相位图像常驻 CPU（pinned），逐相位搬 GPU（1mm case8 省 ~2.9GB 常驻）。

【O10】平滑正则修正：原实现 c[:, 1:]-c[:, :-1] 在 (24,D,H,W) 上是对 **通道维** 求差，
    z 方向实际未被正则（通道耦合项为意外产物）。本实现正确对 z,y,x 三方向求差。
    粗网格梯度幅度约为全分辨率上采样场的 down 倍，故权重按 1/down^2 缩放以保持量级。

用法：
  python pmr_fast.py --dataset dirlab --case 1 --max-down 2 --enc-down 4 --res-down 2
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os

# 必须在 import torch 之前设置，缓解显存碎片（Windows 不支持 expandable_segments）
if os.name != 'nt':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk

torch.manual_seed(0)
np.random.seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True


# ==================== 数据加载（O9：图像常驻 CPU） ====================
def load_imgs(data_root, dataset, cn, down):
    """返回 (imgs, sitk_imgs)：imgs 为 CPU 上 float32 张量列表，逐相位搬 GPU。"""
    imgs, sitk_imgs = [], {}
    for p in range(10):
        if dataset == 'dirlab':
            path = os.path.join(data_root, 'DIRLAB', 'mha', f'case{cn}', f'case{cn}_T{p}0_R.mha')
        else:
            path = os.path.join(data_root, 'CREATIS', str(cn), f'{p*10:02d}_R.mha')
        img = sitk.ReadImage(path)
        if down != 1:
            size = img.GetSize()
            tgt = [max(1, s // down) for s in size]
            rs = sitk.ResampleImageFilter()
            rs.SetSize(tgt)
            rs.SetOutputSpacing([sp * down for sp in img.GetSpacing()])
            rs.SetOutputOrigin(img.GetOrigin())
            rs.SetOutputDirection(img.GetDirection())
            rs.SetInterpolator(sitk.sitkLinear)
            img = rs.Execute(img)
        sitk_imgs[p] = img
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
        imgs.append(torch.from_numpy(arr)[None, None])
    return imgs, sitk_imgs


def load_landmarks(data_root, dataset, cn):
    if dataset == 'dirlab':
        lm0 = np.loadtxt(os.path.join(data_root, 'DIRLAB', 'points', f'case{cn}', f'case{cn}_300_T00_xyz_R.txt'))
        lm5 = np.loadtxt(os.path.join(data_root, 'DIRLAB', 'points', f'case{cn}', f'case{cn}_300_T50_xyz_R.txt'))
        return {'lm0': lm0, 'lm5': lm5}
    pts = {}
    for p in range(10):
        pts[p] = np.loadtxt(os.path.join(data_root, 'CREATIS', str(cn), f'{p*10:02d}.pts'))
    return {'pts': pts}


# ==================== 编码器（C，O4：输入预降采样） ====================
class CoefNetFast(nn.Module):
    """轻量 3D encoder：降采样后的 T00 图像 -> (3*K*2 通道 D/8 粗网格系数, 仿射参数)。

    in_down 控制输入预降采样倍率，输出网格始终为 D/8：
      in_down=1 -> strides (1,2,2,2)
      in_down=2 -> strides (1,1,2,2)
      in_down=4 -> strides (1,1,1,2)   [默认]
      in_down=8 -> strides (1,1,1,1)
    """

    def __init__(self, in_ch=1, out_ch=24, scale=0.05, affine=True, K_a=2, in_down=4):
        super().__init__()
        self.scale = scale
        self.affine = affine
        self.K_a = K_a
        self.in_down = in_down
        n_stride_blocks = {1: 3, 2: 2, 4: 1, 8: 0}.get(in_down, 1)
        widths = (16, 32, 48, 64)
        layers, c = [], in_ch
        for i, w in enumerate(widths):
            stride = 2 if i >= (len(widths) - n_stride_blocks) else 1
            layers += [nn.Conv3d(c, w, 3, stride=stride, padding=1), nn.ReLU()]
            c = w
        self.enc = nn.Sequential(*layers)
        self.out = nn.Conv3d(64, out_ch, 3, padding=1)
        nn.init.normal_(self.out.weight, std=0.01)
        nn.init.zeros_(self.out.bias)
        if affine:
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, K_a * 2 * 12),
            )
            nn.init.zeros_(self.fc[-1].weight)
            nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        f = self.enc(x)
        coef_c = self.out(f) * self.scale
        if self.affine:
            return coef_c, self.fc(f).squeeze(0)
        return coef_c, None


class DirectCoef(nn.Module):
    """AB 消融：不用 CNN（C），在粗网格上直接优化周期系数。

    接口与 CoefNetFast 一致（forward 忽略输入，返回 (coef_c, aff)），
    使 train_stage 无需区分两者。零初始化 -> 初始位移为 0，与原 CNN 小初始化行为一致。
    """

    def __init__(self, out_ch, cshape, K_a=2, affine=True):
        super().__init__()
        self.coef = nn.Parameter(torch.zeros(1, out_ch, *cshape))
        self.affine = affine
        self.aff = nn.Parameter(torch.zeros(K_a * 2 * 12)) if affine else None

    def forward(self, x):
        return self.coef, self.aff


# ==================== 相位流形（O1：粗网格合成 + 3 通道上采样） ====================
def disp_coarse(coef, theta, K):
    """在 **粗网格** 上合成位移：d = sum_k c_k*(cos kθ - 1) + s_k*sin kθ。
    coef: (N, 3*K*2, dc, hc, wc)，通道布局 [k][xyz cos][xyz sin]（与原实现一致）。
    θ=0 与 θ=2π 处恒为 0 —— 闭合为代数恒等。"""
    th = torch.as_tensor(theta, device=coef.device, dtype=coef.dtype)
    d = None
    for k in range(1, K + 1):
        ck = torch.cos(k * th) - 1
        sk = torch.sin(k * th)
        o = (k - 1) * 6
        term = coef[:, o:o + 3] * ck + coef[:, o + 3:o + 6] * sk
        d = term if d is None else d + term
    return d                                          # (N,3,dc,hc,wc)


def disp_full(coef, theta, K, size):
    """粗网格合成后上采样到全分辨率（O1，与原实现代数等价）。"""
    return F.interpolate(disp_coarse(coef, theta, K), size=size,
                         mode='trilinear', align_corners=True)


def affine_disp_b(aff, theta, coords, K_a):
    """仿射全局项（O2：可广播 1D 坐标，不预分配 3D 坐标张量）。
    d = sum_k [A_k*(cos kθ-1) + B_k*sin kθ]·(X-C) + [b_k*(cos kθ-1) + c_k*sin kθ]
    coords: (zz, yy, xx) 三个已减去中心的广播形状张量。返回 (3,D,H,W)。"""
    zz, yy, xx = coords
    th = torch.as_tensor(theta, device=aff.device, dtype=aff.dtype)
    acc = None
    for k in range(1, K_a + 1):
        ck = torch.cos(k * th) - 1
        sk = torch.sin(k * th)
        o = (k - 1) * 24
        A = (aff[o:o + 9].view(3, 3) * ck + aff[o + 9:o + 18].view(3, 3) * sk)
        b = (aff[o + 18:o + 21] * ck + aff[o + 21:o + 24] * sk)
        # out[i] = A[i,0]*xx + A[i,1]*yy + A[i,2]*zz   （x/y/z 与位移通道一致）
        comp = [A[i, 0] * xx + A[i, 1] * yy + A[i, 2] * zz + b[i] for i in range(3)]
        term = torch.stack(comp, dim=0)[None]         # (1,3,D,H,W) 广播
        acc = term if acc is None else acc + term
    return acc


def make_coords(shape, K_a):
    """返回已减去中心的 (zz, yy, xx) 广播形状张量（O2）。"""
    D, H, W = shape[2], shape[3], shape[4]
    zz = (torch.arange(D, device=device, dtype=torch.float32) - D / 2).view(D, 1, 1)
    yy = (torch.arange(H, device=device, dtype=torch.float32) - H / 2).view(1, H, 1)
    xx = (torch.arange(W, device=device, dtype=torch.float32) - W / 2).view(1, 1, W)
    return (zz, yy, xx)


def grid_(shape):
    _, _, D, H, W = shape
    gz = torch.linspace(-1, 1, D, device=device)
    gy = torch.linspace(-1, 1, H, device=device)
    gx = torch.linspace(-1, 1, W, device=device)
    gz, gy, gx = torch.meshgrid(gz, gy, gx, indexing='ij')
    return torch.stack([gx, gy, gz], dim=-1)[None]


# ==================== 相似度 ====================
def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return -(a * b).sum() / (a.norm() * b.norm() + 1e-8)


def pool_stats(x, win, stride):
    """窗口均值与平方均值（局部 NCC 用）。固定图像统计量每阶段只需算一次。"""
    pad = win // 2
    p = lambda t: F.avg_pool3d(t, win, stride=stride, padding=pad, count_include_pad=False)
    return p(x), p(x * x)


def local_ncc(w, f, f_stats, win, stride):
    """O3：局部 NCC 在 stride 子采样网格上求值。

    代数上等价于原实现（stride=1 时）：
        mean_(ac*bc) = mean_(w*f) - mean_(w)*mean_(f)
        mean_(ac^2)  = mean_(w^2) - mean_(w)^2
    但不再物化全分辨率 (w-am)/(f-bm) 张量（1mm case8 每项 0.32GB），
    且修正了原实现在 stride>1 时无法广播的问题。"""
    pad = win // 2
    pool = lambda t: F.avg_pool3d(t, win, stride=stride, padding=pad, count_include_pad=False)
    fp, f2 = f_stats                      # 固定图像：窗口均值 / 平方均值（每阶段只算一次）
    wp = pool(w)
    wf = pool(w * f)
    var_w = pool(w * w) - wp * wp
    var_f = f2 - fp * fp
    cov = wf - wp * fp
    return -(cov / torch.sqrt(var_w * var_f + 1e-5)).mean()


def reg_coarse(coef_c, w_smooth, w_l2, down, l2_phys=1.0):
    """O10：在粗网格上对 z,y,x 三方向求差（修正原实现的通道维求差）。

    分辨率不变性（1mm 收敛缺陷修复的关键）：
      * 平滑项：物理位移梯度 ≈ Δcoef·spacing / (8·spacing) = Δcoef/8，
        与控制网格物理尺寸无关，故 w_smooth/down^2 已天然分辨率不变；
      * L2 项：物理位移 = coef·spacing，原式直接罚 coef^2 等价于罚
        (phys/spacing)^2 —— 1mm 下同样的权重在**物理量纲上强 4 倍**，
        会把形变压死。l2_phys = (spacing/ref_mm)^2 将其换算回物理量纲，
        2mm 时 l2_phys=1（与原行为完全一致），1mm 时为 0.25。"""
    s = (coef_c[:, :, 1:] - coef_c[:, :, :-1]).pow(2).mean()
    s = s + (coef_c[:, :, :, 1:] - coef_c[:, :, :, :-1]).pow(2).mean()
    s = s + (coef_c[:, :, :, :, 1:] - coef_c[:, :, :, :, :-1]).pow(2).mean()
    return w_smooth * s / (down ** 2) + w_l2 * l2_phys * coef_c.pow(2).mean()


def warp(img, g, d, S):
    return F.grid_sample(img, (g + (d * (2.0 / S).view(3, 1, 1, 1)).permute(1, 2, 3, 0)
                               ).clamp(-1, 1),
                         mode='bilinear', padding_mode='border', align_corners=True)


# ==================== 训练阶段 ====================
def train_stage(net, imgs_cpu, img0, coef_in0, K, thetas, S, g, shape, iters, metric,
                w_smooth, w_l2, lr, label, win, coords, K_a, amp, ncc_stride, down,
                l2_phys=1.0, diag=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    t0 = time.time()
    f_stats = pool_stats(img0, win, ncc_stride) if metric == 'local' else None
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for i in range(10):
            img_i = imgs_cpu[i].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp):
                coef_c, aff = net(coef_in0)
                d = disp_full(coef_c, thetas[i], K, shape[2:]).squeeze(0)
                if aff is not None:
                    d = d + affine_disp_b(aff, thetas[i], coords, K_a).squeeze(0)
                w = warp(img_i, g, d, S)
                li = (local_ncc(w, img0, f_stats, win, ncc_stride) if metric == 'local'
                      else ncc(w, img0)) / 10
                if i == 9:
                    li = li + reg_coarse(coef_c, w_smooth, w_l2, down, l2_phys)
            scaler.scale(li).backward()
            loss_sum += li.detach().float().item()
            del d, w, li
        scaler.step(opt)
        scaler.update()
        if (it + 1) % 50 == 0 or it == 0:
            mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == 'cuda' else 0
            extra = ''
            if diag is not None:
                # 诊断：T50 处平均位移幅值（mm）与正则分量，判断形变是否被压死
                with torch.no_grad():
                    c_, a_ = net(coef_in0)
                    d_ = disp_full(c_, np.pi, K, shape[2:]).squeeze(0)
                    if a_ is not None:
                        d_ = d_ + affine_disp_b(a_, np.pi, coords, K_a).squeeze(0)
                    dmm = (d_ * torch.as_tensor(diag['spacing'], device=device,
                                                dtype=torch.float32).view(3, 1, 1, 1)).norm(dim=0)
                    sm = reg_coarse(c_, w_smooth, 0.0, down, l2_phys).item()
                    l2 = reg_coarse(c_, 0.0, w_l2, down, l2_phys).item()
                    diag['mean_disp_mm'] = float(dmm.mean())
                    diag['p95_disp_mm'] = float(torch.quantile(dmm.flatten().float(), 0.95))
                    diag['reg_smooth'] = sm
                    diag['reg_l2'] = l2
                extra = (f' | |d(T50)| {diag["mean_disp_mm"]:6.2f}mm '
                         f'p95 {diag["p95_disp_mm"]:6.2f} | reg_s {diag["reg_smooth"]:.3f} '
                         f'reg_l2 {diag["reg_l2"]:.3f}')
            print(f'    {label}iter {it+1:4d} loss {loss_sum:.4f} '
                  f'({time.time()-t0:.0f}s, peak {mem:.2f}GB){extra}', flush=True)
    return loss_sum


# ==================== 评估 ====================
def sample_dvf(d_mm, lms, spacing, mode='linear'):
    """在 landmark 位置采样位移场。mode='linear' 用三线性（O8），'nearest' 复现旧口径。"""
    if mode == 'nearest':
        out = np.zeros((len(lms), 3))
        _, D, H, W = d_mm.shape
        for j, lm in enumerate(lms):
            x = int(round(lm[0] / spacing[0])); y = int(round(lm[1] / spacing[1])); z = int(round(lm[2] / spacing[2]))
            z = min(max(z, 0), D - 1); y = min(max(y, 0), H - 1); x = min(max(x, 0), W - 1)
            out[j] = d_mm[:, z, y, x]
        return out
    # 三线性：先转成体素单位的位移场，再按体素坐标 grid_sample
    d_vox = torch.from_numpy(d_mm / spacing[:, None, None, None]).float()[None].to(device)
    _, D, H, W = d_mm.shape
    lms = np.asarray(lms, dtype=np.float64)
    idx = lms / spacing[None, :]                      # (N,3) 体素坐标 x,y,z
    # grid 形状 (N, D_out, H_out, W_out, 3)，batch 必须与 input 一致(1)，
    # 故把 N 个采样点放在 D_out 维上
    norm = torch.zeros(1, len(lms), 1, 1, 3, device=device)
    for c, s in enumerate([W, H, D]):
        val = (2.0 * torch.from_numpy(np.ascontiguousarray(idx[:, c])).float().to(device)
               / max(s - 1, 1)) - 1.0
        norm[0, :, 0, 0, c] = val
    out = F.grid_sample(d_vox, norm, mode='bilinear', padding_mode='border',
                        align_corners=True)
    return (out[0, :, :, 0, 0].T.cpu().numpy()) * spacing[None, :]


def main():
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default=os.path.join(HERE, '..', 'reference_4', 'data'))
    ap.add_argument('--dataset', choices=['dirlab', 'creatis'], default='dirlab')
    ap.add_argument('--case', type=int, default=1)
    ap.add_argument('--max-down', type=int, default=1, choices=[4, 2, 1])
    ap.add_argument('--iters1', type=int, default=0)
    ap.add_argument('--iters2', type=int, default=0)
    ap.add_argument('--res-iters', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.0)
    ap.add_argument('--affine', type=int, default=1)
    ap.add_argument('--enc-down', type=int, default=4, choices=[1, 2, 4, 8],
                    help='O4：编码器输入预降采样倍率（1=原实现）')
    ap.add_argument('--res-down', type=int, default=0,
                    help='O6：残差系数粗网格倍率；0=自动（2mm->2, 1mm->4）')
    ap.add_argument('--ncc-stride', type=int, default=0,
                    help='O3：局部 NCC 求值 stride；0=自动（win//2，1=原实现）')
    ap.add_argument('--amp', type=int, default=-1, help='-1=自动（1mm 开，2mm 关）')
    ap.add_argument('--ref-mm', type=float, default=2.0,
                    help='L2 正则物理量纲化的参考分辨率(mm)；2mm 时与旧行为完全一致')
    ap.add_argument('--coef-scale', type=float, default=0.05,
                    help='CNN 输出系数缩放（初始化幅值）')
    ap.add_argument('--K', type=int, default=4, help='流形谐波阶数（消融用）')
    ap.add_argument('--no-cnn', type=int, default=0,
                    help='AB 消融：1=不用 CNN，直接在粗网格优化周期系数')
    ap.add_argument('--init-from', type=str, default='',
                    help='从已训练 CNN 权重热启动（粗到细：2mm 训练 -> 1mm 精调）')
    ap.add_argument('--save-net', type=str, default='',
                    help='训练结束后保存 CNN 权重路径（供更细分辨率热启动）')
    ap.add_argument('--smooth-mode', choices=['coarse'], default='coarse')
    ap.add_argument('--tag', type=str, default='s1f')
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'results'))
    ap.add_argument('--save-dvf', type=int, default=1)
    ap.add_argument('--save-dvf-max-gb', type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cn, K, KR, K_a = args.case, args.K, 4, 2
    down = args.max_down
    scale = {1: 1.0, 2: 0.5, 4: 0.25}[down]
    iters1 = args.iters1 if args.iters1 > 0 else int(round(800 * scale))
    iters2 = args.iters2 if args.iters2 > 0 else int(round(800 * scale))
    lr1 = args.lr if args.lr > 0 else 2e-3 * scale
    res_down = args.res_down if args.res_down > 0 else (4 if down == 1 else 2)
    amp = (down == 1) if args.amp < 0 else bool(args.amp)

    print(f'[{args.dataset} case{cn}] PMR-Fast | {down}mm | iter1={iters1} iter2={iters2} '
          f'lr1={lr1:.1e} affine={bool(args.affine)} | enc_down={args.enc_down} '
          f'res_down={res_down} amp={amp}', flush=True)

    t_start = time.time()
    imgs, sitk_imgs = load_imgs(args.data, args.dataset, cn, down)
    spacing = np.array(sitk_imgs[0].GetSpacing())
    if device == 'cuda':
        for i in range(10):                                  # O9
            imgs[i] = imgs[i].pin_memory()
    img0 = imgs[0].to(device)
    shape = (1, 1) + tuple(img0.shape[2:])
    D, H, W = shape[2], shape[3], shape[4]
    S = torch.tensor([W, H, D], device=device)
    g = grid_(shape)
    thetas = [2 * np.pi * i / 10 for i in range(10)]
    coords = make_coords(shape, K_a)

    n_vox = D * H * W
    print(f'  网格 {D}x{H}x{W} = {n_vox/1e6:.1f} Mvox, spacing {spacing[0]:.2f}mm', flush=True)
    print(f'  旧实现 24ch 全分辨率系数场 = {n_vox*24*4/1024**3:.2f} GB (x2 含梯度)', flush=True)

    # 编码器输入（O4）
    coef_in0 = F.avg_pool3d(img0, args.enc_down) if args.enc_down > 1 else img0
    net = CoefNetFast(out_ch=3 * K * 2, affine=bool(args.affine), K_a=K_a,
                      in_down=args.enc_down, scale=args.coef_scale).to(device)
    if args.no_cnn:
        # AB 消融：先跑一次 dummy forward 拿到 CNN 的粗网格尺寸，再换成直接优化
        with torch.no_grad():
            cshape = tuple(net(coef_in0)[0].shape[2:])
        net = DirectCoef(3 * K * 2, cshape, K_a=K_a, affine=bool(args.affine)).to(device)
        print(f'  [AB 消融] 无 CNN，粗网格 {cshape} 直接优化 '
              f'({sum(p.numel() for p in net.parameters())/1e3:.0f}K 参数)', flush=True)
    print(f'  CNN 参数量 {sum(p.numel() for p in net.parameters())/1e3:.0f}K, '
          f'K={K}, coef_scale={args.coef_scale}', flush=True)

    # 粗到细热启动：1mm 的相似度目标更嘈杂，单分辨率优化易陷局部极小；
    # 用较粗分辨率训练好的权重初始化，可显著改善细分辨率收敛。
    if args.init_from:
        sd = torch.load(args.init_from, map_location=device)
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f'  [warm-start] 载入 {args.init_from} '
              f'(missing={len(missing)}, unexpected={len(unexpected)})', flush=True)

    # 分辨率不变性：L2 换算到物理量纲（1mm 下 0.25，2mm 下 1.0=旧行为）
    l2_phys = (float(spacing[0]) / args.ref_mm) ** 2
    print(f'  [res-scale] spacing={spacing[0]:.2f}mm -> L2 物理因子 {l2_phys:.3f} '
          f'(ref {args.ref_mm}mm)', flush=True)
    diag = {'spacing': np.asarray(spacing, dtype=np.float32)}

    win_phys = 30.0
    win = max(5, int(round(win_phys / spacing[0]))) if spacing[0] > 0 else 15
    ncc_stride = args.ncc_stride if args.ncc_stride > 0 else max(1, win // 2)
    print(f'  [win] 物理窗口 {win_phys:.0f}mm -> {win} 体素, NCC stride {ncc_stride}', flush=True)

    print(f'  阶段1（全局NCC粗对齐）: {iters1} iter, lr={lr1:.1e}', flush=True)
    train_stage(net, imgs, img0, coef_in0, K, thetas, S, g, shape, iters1, 'global',
                1.0, 0.1, lr1, 'C1 ', win, coords, K_a, amp, ncc_stride, 8,
                l2_phys=l2_phys, diag=diag)
    print(f'  阶段2（局部NCC精调）: {iters2} iter', flush=True)
    train_stage(net, imgs, img0, coef_in0, K, thetas, S, g, shape, iters2, 'local',
                0.3, 0.05, lr1 * 0.3, 'C2 ', win, coords, K_a, amp, ncc_stride, 8,
                l2_phys=l2_phys, diag=diag)

    # ---- 周期残差（O6：粗网格系数）----
    if args.save_net:
        torch.save(net.state_dict(), args.save_net)
        print(f'  [save-net] CNN 权重已保存: {args.save_net}', flush=True)
    with torch.no_grad():
        coef_c0, aff0 = net(coef_in0)
        coef_c0 = coef_c0.detach()
        aff0 = aff0.detach() if aff0 is not None else None
    dc, hc, wc = coef_c0.shape[2], coef_c0.shape[3], coef_c0.shape[4]
    rshape = (max(1, D // res_down), max(1, H // res_down), max(1, W // res_down))
    coef_r = torch.zeros(1, 3 * KR * 2, *rshape, device=device, requires_grad=True)
    print(f'  残差阶段（KR={KR}, 粗网格 {rshape}, {args.res_iters} iter）: '
          f'{coef_r.numel()*4/1024**2:.1f} MB 参数（全分辨率需 {n_vox*24*4/1024**3:.2f} GB）',
          flush=True)
    opt = torch.optim.Adam([coef_r], lr=1e-2)
    t0 = time.time()
    for it in range(args.res_iters):
        opt.zero_grad(set_to_none=True)
        for i in range(10):
            img_i = imgs[i].to(device, non_blocking=True)
            with torch.no_grad():
                d_m = disp_full(coef_c0, thetas[i], K, (D, H, W)).squeeze(0)
                if aff0 is not None:
                    d_m = d_m + affine_disp_b(aff0, thetas[i], coords, K_a).squeeze(0)
            d = d_m + disp_full(coef_r, thetas[i], KR, (D, H, W)).squeeze(0)
            w = warp(img_i, g, d, S)
            li = ncc(w, img0) / 10
            li.backward()
            del d, w, li
        r = 0.3 * (coef_r[:, :, 1:] - coef_r[:, :, :-1]).pow(2).mean()
        r = r + 0.3 * (coef_r[:, :, :, 1:] - coef_r[:, :, :, :-1]).pow(2).mean()
        r = r + 0.3 * (coef_r[:, :, :, :, 1:] - coef_r[:, :, :, :, :-1]).pow(2).mean()
        r = r + 0.1 * coef_r.pow(2).mean()
        r.backward()
        opt.step()
        if (it + 1) % 50 == 0 or it == 0:
            print(f'    残差 iter {it+1:4d} ({time.time()-t0:.0f}s)', flush=True)

    # ---- 评估 ----
    lms = load_landmarks(args.data, args.dataset, cn)
    with torch.no_grad():
        coef_c1, aff1 = net(coef_in0)
        d50_m = disp_full(coef_c1, np.pi, K, (D, H, W)).squeeze(0)
        if aff1 is not None:
            d50_m = d50_m + affine_disp_b(aff1, np.pi, coords, K_a).squeeze(0)
        d50_t = d50_m + disp_full(coef_r, np.pi, KR, (D, H, W)).squeeze(0)
        d50_m_mm = d50_m.cpu().numpy() * spacing[:, None, None, None]
        d50_t_mm = d50_t.cpu().numpy() * spacing[:, None, None, None]
        # 闭合验证：完整位移场（流形 + 仿射 + 残差）在 θ=0 与 θ=2π 处，单位 mm
        def full_disp(th):
            dd = disp_full(coef_c1, th, K, (D, H, W)).squeeze(0)
            if aff1 is not None:
                dd = dd + affine_disp_b(aff1, th, coords, K_a).squeeze(0)
            return dd + disp_full(coef_r, th, KR, (D, H, W)).squeeze(0)
        sp_t = torch.as_tensor(spacing, device=device, dtype=torch.float32).view(3, 1, 1, 1)
        d0 = (full_disp(0.0) * sp_t).abs().max().item()
        d2p = ((full_disp(2 * np.pi) - full_disp(0.0)) * sp_t).abs().max().item()

    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == 'cuda' else 0.0
    res = {'dataset': args.dataset, 'case': cn, 'down_mm': down, 'affine': bool(args.affine),
           'impl': 'pmr_fast', 'enc_down': args.enc_down, 'res_down': res_down,
           'ncc_stride': ncc_stride, 'amp': bool(amp), 'win': win,
           'iters1': iters1, 'iters2': iters2, 'res_iters': args.res_iters, 'lr1': lr1,
           'n_vox': int(n_vox), 'peak_mem_gb': round(peak_gb, 2),
           'coef_scale': args.coef_scale, 'l2_phys': round(l2_phys, 4),
           'mean_disp_mm_t50': round(diag.get('mean_disp_mm', float('nan')), 4),
           'p95_disp_mm_t50': round(diag.get('p95_disp_mm', float('nan')), 4),
           'closure_d0': float(d0), 'closure_2pi': float(d2p)}

    if args.dataset == 'dirlab':
        lm0, lm5 = lms['lm0'], lms['lm5']
        init = np.linalg.norm(lm0 - lm5, axis=1).mean()
        out = {}
        for mode in ('nearest', 'linear'):
            tm = np.linalg.norm(lm5 - sample_dvf(d50_m_mm, lm5, spacing, mode) - lm0, axis=1).mean()
            tt = np.linalg.norm(lm5 - sample_dvf(d50_t_mm, lm5, spacing, mode) - lm0, axis=1).mean()
            out[mode] = (float(tm), float(tt))
        res.update({'init_tre': float(init),
                    'tre_manifold': out['nearest'][0], 'tre_total': out['nearest'][1],
                    'tre_manifold_linear': out['linear'][0], 'tre_total_linear': out['linear'][1]})
        print(f'\n[DIRLAB case{cn} {down}mm PMR-Fast] 初始 {init:.2f} | '
              f'流形 {out["nearest"][0]:.2f} | +残差 {out["nearest"][1]:.2f} mm (nearest 口径)')
        print(f'                        三线性口径: 流形 {out["linear"][0]:.2f} | '
              f'+残差 {out["linear"][1]:.2f} mm')
    else:
        pts = lms['pts']
        init = np.linalg.norm(pts[5] - pts[0], axis=1).mean()
        tre_m_ph, tre_ph = [], []
        with torch.no_grad():
            for i in range(10):
                dm = disp_full(coef_c1, thetas[i], K, (D, H, W)).squeeze(0)
                if aff1 is not None:
                    dm = dm + affine_disp_b(aff1, thetas[i], coords, K_a).squeeze(0)
                dt = dm + disp_full(coef_r, thetas[i], KR, (D, H, W)).squeeze(0)
                dm = dm.cpu().numpy() * spacing[:, None, None, None]
                dt = dt.cpu().numpy() * spacing[:, None, None, None]
                tre_m_ph.append(np.linalg.norm(pts[i] - sample_dvf(dm, pts[i], spacing) - pts[0], axis=1).mean())
                tre_ph.append(np.linalg.norm(pts[i] - sample_dvf(dt, pts[i], spacing) - pts[0], axis=1).mean())
        res.update({'init_tre': float(init), 'tre_manifold': float(np.mean(tre_m_ph)),
                    'tre_total': float(np.mean(tre_ph)),
                    'tre_phases': [float(t) for t in tre_ph]})
        print(f'\n[CREATIS case{cn} {down}mm PMR-Fast] 流形 {np.mean(tre_m_ph):.2f} | '
              f'+残差 {np.mean(tre_ph):.2f} mm')

    res['time_s'] = round(time.time() - t_start, 1)
    print(f'闭合验证: |d(0)|max={d0:.2e} |d(2π)-d(0)|max={d2p:.2e} mm')
    print(f'峰值显存 {peak_gb:.2f} GB | 总耗时 {res["time_s"]:.0f}s')

    fn = os.path.join(args.out, f'{args.tag}_{args.dataset}_case{cn}_{down}mm.json')
    with open(fn, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'已保存: {fn}')

    if args.save_dvf:
        est_gb = 10 * 3 * n_vox * 4 / 1024 ** 3
        if est_gb > args.save_dvf_max_gb:
            print(f'[跳过 DVF 保存] 预计 {est_gb:.1f} GB > 阈值 {args.save_dvf_max_gb} GB '
                  f'（用 --save-dvf-max-gb 调整）')
        else:
            with torch.no_grad():
                d_phases = []
                for th in thetas:
                    dm = disp_full(coef_c1, th, K, (D, H, W)).squeeze(0)
                    if aff1 is not None:
                        dm = dm + affine_disp_b(aff1, th, coords, K_a).squeeze(0)
                    dm = dm + disp_full(coef_r, th, KR, (D, H, W)).squeeze(0)
                    d_phases.append((dm.cpu().numpy() * spacing[:, None, None, None]).astype(np.float32))
            dvf_path = os.path.join(args.out, f'{args.tag}_{args.dataset}_case{cn}_{down}mm_dvf.npy')
            np.save(dvf_path, np.stack(d_phases))
            print(f'全相位 DVF 已保存: {dvf_path} ({est_gb:.1f} GB)')


if __name__ == '__main__':
    main()
