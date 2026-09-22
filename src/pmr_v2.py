"""
pmr_v2.py —— PMR 第二版训练核心（大胆改造版）
================================================================================
与 v1（pmr_fast.py，保留为参考实现）的差别：

 1. **肺掩膜相似度**（已由 test_mask_hypothesis.py 证实的根因修复）
    全图 NCC 中约 82% 体素是非肺组织且几乎不动，它们惩罚位移；肺只占 ~18%
    却承载全部真实运动 => 全局最优 = 小位移解（case1: 真值 3.89mm，模型 1.69mm）。
    只在掩膜内求相似度可解除该"锚"（case1: TRE 1.838 -> 1.440，|d| 1.69 -> 2.89）。
    mask 模式：none / t00 / union(T00∪T50，膨胀 2)。

 2. **相位随机采样**：v1 每步对 10 个相位各做一次反向传播再走一步 Adam；
    v2 每步只采样 `--phases-per-step`（默认 2）个相位 => 同算力下优化步数 ×5，
    Elastix 的 NewSamplesEveryIteration 同理。

 3. **度量可选**：全局 NCC / 局部 NCC / MIND（Heinrich 2012）/ 组合。

 4. **正则强度可缩放**（掩膜后数据项量级变化，必须重配）。

 5. **仿射预对齐阶段**（Elastix 风格：先纯仿射，再形变）。

 6. **鲁棒强度归一化**（分位数裁剪，避免 max=13447 的离群值压掉肺对比度）。

 7. **跨分辨率系数热启动**（`--init-coef`：粗分辨率收敛的周期系数三线性上采样后
    作为初值，CNN 输出作为附加修正项 —— 真正的多尺度）。

不动的东西（论文卖点必须保住）：
  * 周期流形 d = Σ_k [a_k(cos kθ−1) + b_k sin kθ] + d_aff，θ=0 与 2π 处恒为 0
    => **闭合仍是代数恒等，mask/度量/优化的改动不影响它**。
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path shim ---
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmr_fast import (CoefNetFast, DirectCoef, disp_full, affine_disp_b, make_coords,
                      grid_, ncc, pool_stats, reg_coarse, warp, sample_dvf, device)
from recompute_tre_standard import load_lm, sample_trilinear
from lung_mask import mask_at_down

# 数据根目录。可用环境变量覆盖 —— 公开仓库里必须能被别人改，
# 否则 clone 下来无法复现（`export PMR_DATA_ROOT=/path/to/data`）。
DATA = os.environ.get('PMR_DATA_ROOT') or os.path.join(HERE, '..', 'reference_4', 'data')


def creatis_valid_phases(cn, phases=None):
    """探测 CREATIS 某病例哪些相位的 `.pts` 是**真正的 landmark 数据**。

    🔴 为什么必须探测：**官方数据并非每个病例都有全相位标注**。
    依据 Vandemeulebroucke et al., Med. Phys. 38(1):166–178 (2011) 原文：
      * "For all six patients, 100 point correspondences were provided between the
         end-exhale and the end-inhale frame"
      * "**For Patients 1-3**, a single observer provided 100 correspondences for
         **each of the frames** of the 4D CT … 900 manually identified landmarks"

    ⇒ **只有病例 0/1/2（= 原文 Patients 1–3）有全 10 相位的专家标注**；
      病例 3/4/5 仅 end-exhale(00) 与 end-inhale(50) 有效。
      其余相位的 `.pts` 在上游**不存在**，下载时会存成 **HTML 404 页面**。
    """
    d = os.path.join(DATA, 'CREATIS', str(cn))
    ok = []
    for p in (range(10) if phases is None else phases):
        f = os.path.join(d, f'{p * 10:02d}.pts')
        if not os.path.exists(f) or os.path.getsize(f) < 64:
            continue
        try:
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                head = fh.readline().strip()
            if head.startswith('<') or 'html' in head.lower():
                continue                      # HTML 404 页，不是数据
            a = np.loadtxt(f)
            if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] >= 10:
                ok.append(p)
        except Exception:
            continue
    return ok


def load_lm_dataset(cn, dataset='dirlab', p_ref=0, p_tgt=5):
    """读取 (参考相位, 目标相位) 的对应 landmark，返回**原点相对**物理坐标 (N,3) mm。

    🔴 坐标参考系逐数据集不同（已实测，`tools/check_lm_frame.py`）：
      * DIRLAB: origin=(0,0,0)、direction=单位阵 ⇒ 文件坐标即原点相对坐标。
      * CREATIS: origin=(-250,-250,-164.5)、direction=单位阵，且 `.pts` 里**有负值**
        ⇒ 文件坐标是**绝对物理坐标**，必须减去 origin 才能喂给 `sample_trilinear`
        （后者做 `idx = pts/spacing`，隐含假设 pts 相对原点）。
    两点对应关系按**行号**（各相位的点数与顺序一致，已核验）。

    🔴 CREATIS 的相位覆盖**有条件**，见 `creatis_valid_phases`：
    病例 3/4/5 只有 00 与 50 有效，用其他相位会抛错而不是静默出错。
    """
    if dataset == 'creatis':
        d = os.path.join(DATA, 'CREATIS', str(cn))
        need = sorted({p_ref, p_tgt})
        ok = creatis_valid_phases(cn, need)
        missing = [p for p in need if p not in ok]
        if missing:
            raise ValueError(
                f'CREATIS case{cn} 的相位 {missing} 没有有效 landmark'
                f'（该病例有效相位为 {creatis_valid_phases(cn)}）。'
                f'依原始数据集，病例 3/4/5 只有 00/50，其余相位上游不存在。')
        a = np.loadtxt(os.path.join(d, f'{p_ref * 10:02d}.pts')).reshape(-1, 3)
        b = np.loadtxt(os.path.join(d, f'{p_tgt * 10:02d}.pts')).reshape(-1, 3)
        if len(a) != len(b):
            raise ValueError(f'CREATIS case{cn}: 相位 {p_ref} 有 {len(a)} 点、'
                             f'相位 {p_tgt} 有 {len(b)} 点，无法按行号对应')
        og = np.array(sitk.ReadImage(os.path.join(d, f'{p_ref * 10:02d}_R.mha')).GetOrigin(),
                      dtype=np.float64)
        return a - og, b - og
    return load_lm(cn)


# ==================== 数据加载 ====================
def resample_down(img, down, interp=None):
    """按整数倍降采样，**显式保留 origin/direction**。

    🔴 血泪教训（证据：`tools/prove_origin_bug.py`，回归测试：`verify_resample_geometry.py`）：
    SimpleITK 的 `ResampleImageFilter` 在只设了 `SetSize`/`SetOutputSpacing` 时，
    输出 origin 会**回落到 (0,0,0)**，并**不继承**输入。

      * DIRLAB 的 `_R.mha` origin 恰为 (0,0,0) ⇒ 这个 bug 一直是隐形的；
      * CREATIS 的 `_R.mha` origin=(-250,-250,-164.5) ⇒ 采样区域整体平移 250 mm，
        越界处被默认 `paddingValue=0.0` 填充 —— 而 **HU 0 恰好是软组织**，
        于是填充值伪装成真实组织：不报错、不越界、图看着"正常"。
        实测掩膜内平均 HU 从 −742 被污染成 −72，平均绝对误差 695 HU。

    修好后输出与「原生按索引 every-down 抽样」**逐体素完全一致**（diff = 0.000）。
    """
    rs = sitk.ResampleImageFilter()
    rs.SetSize([max(1, s // down) for s in img.GetSize()])
    rs.SetOutputSpacing([sp * down for sp in img.GetSpacing()])
    rs.SetOutputOrigin(img.GetOrigin())            # ← 关键，勿删
    rs.SetOutputDirection(img.GetDirection())      # ← 关键，勿删
    rs.SetInterpolator(sitk.sitkLinear if interp is None else interp)
    return rs.Execute(img)


def load_imgs_v2(cn, down, norm='robust', dataset='dirlab'):
    """norm='robust' 用 0.5–99.5 分位裁剪，避免离群高值压掉肺对比度。"""
    imgs, sitk_imgs = [], {}
    for p in range(10):
        path = (os.path.join(DATA, 'DIRLAB', 'mha', f'case{cn}', f'case{cn}_T{p}0_R.mha')
                if dataset == 'dirlab' else os.path.join(DATA, 'CREATIS', str(cn), f'{p*10:02d}_R.mha'))
        img = sitk.ReadImage(path)
        if down != 1:
            img = resample_down(img, down)
        sitk_imgs[p] = img
        a = sitk.GetArrayFromImage(img).astype(np.float32)
        if norm == 'robust':
            lo, hi = np.percentile(a, 0.5), np.percentile(a, 99.5)
            a = np.clip(a, lo, hi)
            a = (a - lo) / (hi - lo + 1e-9)
        else:
            a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        imgs.append(torch.from_numpy(a)[None, None])
    return imgs, sitk_imgs


def build_mask(cn, down, shape, mode='union', device='cuda', dataset='dirlab'):
    """mask 模式：none / t00 / union（T00∪T50 再膨胀 2 体素）。"""
    if mode == 'none':
        return None
    m = torch.from_numpy(mask_at_down(cn, down, 0, dataset=dataset))[None, None].float().to(device)
    if mode == 'union':
        m5 = torch.from_numpy(mask_at_down(cn, down, 5, dataset=dataset))[None, None].float().to(device)
        m = torch.clamp(m + m5, 0, 1)
        m = F.max_pool3d(m, 5, stride=1, padding=2)          # 膨胀 2 体素
    if tuple(m.shape[2:]) != tuple(shape[2:]):
        m = F.interpolate(m, size=tuple(shape[2:]), mode='nearest')
    return m


# ==================== 相似度（全部支持掩膜） ====================
def masked_ncc(w, f, m=None):
    """掩膜内全局 NCC。m=None 时退化为全图 NCC（与 v1 数值一致）。"""
    if m is None:
        a = w - w.mean(); b = f - f.mean()
        return -(a * b).sum() / (a.norm() * b.norm() + 1e-8)
    n = m.sum().clamp(min=1.0)
    wm = (w * m).sum() / n
    fm = (f * m).sum() / n
    wc = (w - wm) * m
    fc = (f - fm) * m
    return -(wc * fc).sum() / (wc.norm() * fc.norm() + 1e-8)


def masked_local_ncc(w, f, f_stats, m, win, stride):
    """掩膜内局部 NCC。窗口统计量在全图算，相关系数的聚合方式见 LOCAL_MASK_MODE：

      'select'（默认）：只平均"窗口 ≥50% 落在掩膜内"的那些窗口。
                        掩膜小时会丢弃大量边界窗口。
      'weight'        ：按窗口的掩膜占比加权平均 —— 用上更多数据，梯度更稳。
                        对小掩膜（如 case8 的 9.8%）尤其重要。
    """
    pad = win // 2
    pool = lambda t: F.avg_pool3d(t, win, stride=stride, padding=pad, count_include_pad=False)
    fp, f2 = f_stats
    wp = pool(w); wf = pool(w * f)
    var_w = pool(w * w) - wp * wp
    var_f = f2 - fp * fp
    cov = wf - wp * fp
    cc = cov / torch.sqrt(var_w * var_f + 1e-5)
    if m is None:
        return -cc.mean()
    mm = pool(m)
    if LOCAL_MASK_MODE == 'weight':
        s = mm.sum()
        if float(s) < 1e-6:
            return cc.sum() * 0.0
        return -(cc * mm).sum() / s
    sel = mm > 0.5
    if int(sel.sum()) < 16:
        return cc.sum() * 0.0
    return -cc[sel].mean()


# ==================== C2 跨领域移植：局部相似性属性（时移地震 → 4D-CT） ====================
# 出处：Fomel & Jin (2009) Geophysics 74(2) A7-A11, doi:10.1190/1.3054136；
#       Fomel (2007) Geophysics 72(3) A29-A33, doi:10.1190/1.2437573（shaping regularization）。
#
# 与 `local`（盒窗局部 NCC）的两点结构性差别：
#   1) 窗口核：盒窗（硬截断） → 高斯核（shaping 正则化）。高斯核与等宽盒窗**二阶矩相同**
#      （sigma = win/sqrt(12)），因此二者只差在核形状，是可比的单变量改动。
#      动机：盒窗在肺/胸壁交界会把窗内、窗外内容等权混合——这正是 P1「静止背景污染估计量」
#      的机制；高斯核给出连续的软支持。
#   2) 聚合权重：均匀（或仅按掩膜占比） → 再乘**局部结构显著性** e/(e+ē)，
#      其中 e = G_sigma(f^2) - (G_sigma f)^2 为参考图局部方差。
#      e→0（HU 均匀、无结构）处权重→0：这些窗口的相关系数是噪声主导的，
#      旧实现让它们与有信息的窗口等权参与平均，从而稀释且偏置梯度。
#
# 与医学侧已有逐体素加权方法的边界（`lit/A6` §C2③）：DRAMMS 用局部属性**可区分性**、
# Rivaz 用**单幅图自相似度**；这里的权重来自**两期之间的局部回归是否可解**（参考图有结构、
# 被配准图才能在该处被约束），即跨期一致性本身，而非单图显著性。
LS_SIGMA_FRAC = 1.0 / (12.0 ** 0.5)   # sigma = win * frac，使高斯核与宽 win 的盒窗方差相同


def gauss_kernel1d(sigma, truncate=3.0, device='cuda', dtype=torch.float32):
    r = max(1, int(round(truncate * float(sigma))))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * float(sigma) ** 2))
    return k / k.sum()


_GAUSS_KERNEL_CACHE = {}


def gauss_smooth3d(t, sigma, truncate=3.0):
    """可分离 3-D 高斯平滑（边缘 replicate）。等价于 shaping 正则化里的高斯 shaping 算子。"""
    key = (round(float(sigma), 4), str(t.device), str(t.dtype))
    k = _GAUSS_KERNEL_CACHE.get(key)
    if k is None:
        k = gauss_kernel1d(sigma, truncate, t.device, t.dtype)
        _GAUSS_KERNEL_CACHE[key] = k
    r = (k.numel() - 1) // 2
    x = F.pad(t, (0, 0, 0, 0, r, r), mode='replicate')
    x = F.conv3d(x, k.view(1, 1, -1, 1, 1))
    x = F.pad(x, (0, 0, r, r, 0, 0), mode='replicate')
    x = F.conv3d(x, k.view(1, 1, 1, -1, 1))
    x = F.pad(x, (r, r, 0, 0, 0, 0), mode='replicate')
    x = F.conv3d(x, k.view(1, 1, 1, 1, -1))
    return x


def ls_stats(f, win, m=None):
    """预计算参考图的局部相似性统计量。

    返回 (G f, G f^2, var_f, e_norm)：var_f 为局部方差（结构显著性），
    e_norm 为掩膜内 var_f 的均值，用作 e/(e+e_norm) 的无量纲尺度。
    """
    sigma = max(0.6, win * LS_SIGMA_FRAC)
    fp = gauss_smooth3d(f, sigma)
    f2 = gauss_smooth3d(f * f, sigma)
    var_f = (f2 - fp * fp).clamp_(min=0.0)
    if m is None:
        e_norm = float(var_f.mean())
    else:
        e_norm = float((var_f * m).sum() / m.sum().clamp(min=1.0))
    return (fp, f2, var_f, max(e_norm, 1e-8))


def shaping_local_ncc(w, f, f_ls, m=None, win=15, weight='energy'):
    """Fomel & Jin (2009) 局部相似性属性 → 4D-CT 相似度（越小越好）。

      c(x) = G_sigma(w f) - (G_sigma w)(G_sigma f)  /  sqrt(var_w var_f)
           = 局部回归相关系数（局部相似性属性，带符号）
      L    = - sum[ s(x) * c(x) ] / sum[ s(x) ]
      s(x) = G_sigma(m) * [ e(x)/(e(x)+e_norm) ]^(weight=='energy')   # 软掩膜支持 × 结构显著性

    weight='none' 时关闭结构显著性（只保留高斯核 + 软掩膜），用于单变量拆解。
    """
    fp, f2, var_f, e_norm = f_ls
    # ⚠️ 1 mm 用例开启 autocast(fp16)。局部方差 = G(f²) − (Gf)² 是**相减**，
    #    半精度下会灾难性抵消，故此处强制 float32（autocast 不会把显式 float32 再降精度）。
    w = w.float()
    sigma = max(0.6, win * LS_SIGMA_FRAC)
    wp = gauss_smooth3d(w, sigma)
    wf = gauss_smooth3d(w * f, sigma)
    var_w = (gauss_smooth3d(w * w, sigma) - wp * wp).clamp_(min=0.0)
    cc = (wf - wp * fp) / torch.sqrt(var_w * var_f + 1e-5)
    if m is None:
        ms = torch.ones_like(cc)
    else:
        ms = gauss_smooth3d(m, sigma)
    if weight == 'energy':
        s = ms * (var_f / (var_f + e_norm))
    else:
        s = ms
    tot = s.sum()
    if float(tot) < 1e-6:
        return cc.sum() * 0.0
    return -(cc * s).sum() / tot


# ==================== C3 整合：相位留出 + NRMS 检测门槛 ====================
# 出处：时移地震的可重复性度量 NRMS（归一化均方根差），见 docs/30 §4-§5。
#
# 🔴 关键区别（不得混淆）：地震侧 NRMS 测的是**采集的不可重复性**，需要真正的重复采集，
#    我们没有（红线 X-17）。这里的 NRMS 测的是**留出相位上的模型预测残差** ——
#    即「流形沿 θ 外推到没见过的那一相时，图像对得准不准」。
#    它不是采集噪声底，而是**流形假设的检验**与**逐体素可靠性**的度量。
def nrms(w, f, m=None):
    """归一化均方根差 NRMS = 2·RMS(w−f) / (RMS(w)+RMS(f))。完全按地震侧定义。"""
    if m is None:
        rd = (w - f).pow(2).mean().clamp(min=1e-12).sqrt()
        rw = w.pow(2).mean().clamp(min=1e-12).sqrt()
        rf = f.pow(2).mean().clamp(min=1e-12).sqrt()
    else:
        n = m.sum().clamp(min=1.0)
        rd = ((w - f).pow(2) * m).sum().div(n).clamp(min=1e-12).sqrt()
        rw = (w.pow(2) * m).sum().div(n).clamp(min=1e-12).sqrt()
        rf = (f.pow(2) * m).sum().div(n).clamp(min=1e-12).sqrt()
    return float(2.0 * rd / (rw + rf + 1e-8))


def nrms_local(w, f, sigma=6.0):
    """逐体素局部 NRMS 场 (D,H,W)，用于稠密可靠性图。不做掩膜归一化，由调用方加掩膜。"""
    w = w.float(); f = f.float()
    d2 = gauss_smooth3d((w - f) ** 2, sigma)
    w2 = gauss_smooth3d(w * w, sigma)
    f2 = gauss_smooth3d(f * f, sigma)
    return 2.0 * d2.clamp(min=1e-12).sqrt() / (
        w2.clamp(min=1e-12).sqrt() + f2.clamp(min=1e-12).sqrt() + 1e-8)


_MIND_OFFS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
# 由 --local-mask-mode 设置；模块级以避免层层传参
LOCAL_MASK_MODE = 'select'


def _shift(x, off):
    dz, dy, dx = off
    xp = F.pad(x, (abs(dx), abs(dx), abs(dy), abs(dy), abs(dz), abs(dz)), mode='replicate')
    D, H, W = x.shape[2:]
    z0, y0, x0 = abs(dz) + dz, abs(dy) + dy, abs(dx) + dx
    return xp[:, :, z0:z0 + D, y0:y0 + H, x0:x0 + W]


def mind_desc(x, patch=1):
    """MIND 自相似描述子（Heinrich 2012），返回 (N,6,D,H,W)。"""
    ssds = []
    for o in _MIND_OFFS:
        d = (x - _shift(x, o)) ** 2
        if patch > 0:
            k = 2 * patch + 1
            d = F.avg_pool3d(d, k, stride=1, padding=patch, count_include_pad=False)
        ssds.append(d)
    S = torch.stack(ssds, dim=1)
    var = S.mean(dim=1, keepdim=True) + 1e-8
    return torch.exp(-S / var)


def mind_loss(w, f_desc, m=None):
    """w: 形变后的浮动图；f_desc: 预计算的固定图 MIND 描述子。"""
    d = (mind_desc(w) - f_desc) ** 2
    if m is not None:
        d = d * m
        return d.sum() / (m.sum() * 6 + 1e-8)
    return d.mean()


# ==================== 训练阶段 ====================
def train_stage(net, imgs, img0, coef_in0, K, thetas, S, g, shape, iters, metric,
                w_smooth, w_l2, lr, label, win, coords, K_a, amp, ncc_stride, down,
                mask=None, f_stats=None, f_mind=None, phases_per_step=2, mind_w=1.0,
                l2_phys=1.0, log_every=50, gen=None, f_ls=None, ls_weight='energy',
                train_phases=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    n_ph = len(thetas)
    # C3 整合：train_phases 指定**参与训练**的相位索引；被排除的相位在训练中完全不可见
    # （用于「留出相位 NRMS 检测门槛」，见 docs/30 §5）
    pool = list(range(n_ph)) if train_phases is None else list(train_phases)
    k_ph = min(len(pool), max(1, phases_per_step))
    t0 = time.time()
    loss_sum = 0.0
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        idxs = ([pool[j] for j in torch.randperm(len(pool), device='cpu', generator=gen)[:k_ph].tolist()]
                if k_ph < len(pool) else list(pool))
        loss_sum = 0.0
        for j, i in enumerate(idxs):
            img_i = imgs[i].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp):
                coef_c, aff = net(coef_in0)
                d = disp_full(coef_c, thetas[i], K, shape[2:]).squeeze(0)
                if aff is not None:
                    d = d + affine_disp_b(aff, thetas[i], coords, K_a).squeeze(0)
                w = warp(img_i, g, d, S)
                if metric == 'mind':
                    li = mind_loss(w, f_mind, mask) * mind_w / len(idxs)
                elif metric in ('ls', 'lsg'):
                    li = shaping_local_ncc(w, img0, f_ls, mask, win,
                                           'energy' if metric == 'ls' else 'none') / len(idxs)
                elif metric == 'local':
                    li = masked_local_ncc(w, img0, f_stats, mask, win, ncc_stride) / len(idxs)
                else:
                    li = masked_ncc(w, img0, mask) / len(idxs)
                if j == len(idxs) - 1:
                    li = li + reg_coarse(coef_c, w_smooth, w_l2, down, l2_phys)
            scaler.scale(li).backward()
            loss_sum += li.detach().float().item()
            del d, w, li
        scaler.step(opt); scaler.update()
        if (it + 1) % log_every == 0 or it == 0:
            mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == 'cuda' else 0
            print(f'    {label}iter {it+1:5d} loss {loss_sum:+.5f} '
                  f'({time.time()-t0:.0f}s, peak {mem:.2f}GB)', flush=True)
    return loss_sum


def train_residual(coef_r, imgs, img0, d_base_list, thetas, S, g, KR, shape, iters,
                   metric, mask, f_stats, f_mind, win, ncc_stride, lr, mind_w,
                   phases_per_step=2, label='R  ', log_every=100, gen=None,
                   reg_scale=1.0, jac_weight=0.0, f_ls=None, train_phases=None):
    """残差阶段。

    🔴 注意（`docs/26` §4.1）：本阶段的正则**原先是硬编码的**，不受 `--reg-scale` 影响，
    因此折叠无法通过调 `--reg-scale` 降低（已实测：reg=1/3/10 折叠率几乎相同）。
    现引入 `reg_scale`（默认 1.0 = 原行为）与 `jac_weight`（det(J) hinge 惩罚权重，
    默认 0 = 关闭）以便做精度–折叠权衡。
    """
    opt = torch.optim.Adam([coef_r], lr=lr)
    n_ph = len(thetas)
    pool = list(range(n_ph)) if train_phases is None else list(train_phases)
    k_ph = min(len(pool), max(1, phases_per_step))
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        idxs = ([pool[j] for j in torch.randperm(len(pool), device='cpu', generator=gen)[:k_ph].tolist()]
                if k_ph < len(pool) else list(pool))
        for j, i in enumerate(idxs):
            img_i = imgs[i].to(device, non_blocking=True)
            d = d_base_list[i] + disp_full(coef_r, thetas[i], KR, shape).squeeze(0)
            w = warp(img_i, g, d, S)
            if metric == 'mind':
                li = mind_loss(w, f_mind, mask) * mind_w / len(idxs)
            elif metric in ('ls', 'lsg'):
                li = shaping_local_ncc(w, img0, f_ls, mask, win,
                                       'energy' if metric == 'ls' else 'none') / len(idxs)
            elif metric == 'local':
                li = masked_local_ncc(w, img0, f_stats, mask, win, ncc_stride) / len(idxs)
            else:
                li = masked_ncc(w, img0, mask) / len(idxs)
            li.backward()
            del d, w, li
        r = 0.3 * (coef_r[:, :, 1:] - coef_r[:, :, :-1]).pow(2).mean()
        r = r + 0.3 * (coef_r[:, :, :, 1:] - coef_r[:, :, :, :-1]).pow(2).mean()
        r = r + 0.3 * (coef_r[:, :, :, :, 1:] - coef_r[:, :, :, :, :-1]).pow(2).mean()
        r = r + 0.1 * coef_r.pow(2).mean()
        r = r * reg_scale
        if jac_weight > 0:
            r = r + jac_weight * folding_penalty(
                d_base_list[idxs[0]].detach(), coef_r, thetas[idxs[0]], KR, shape)
        r.backward(); opt.step()
        if (it + 1) % log_every == 0:
            print(f'    {label}iter {it+1:5d} ({time.time()-t0:.0f}s)', flush=True)


def folding_penalty(d_base, coef_r, theta, KR, shape, step=None):
    """对 det(J) 的 hinge 惩罚：`mean( min(0, det J)^2 )`，用于抑制折叠。

    参数
      d_base : 该 θ 处的基准位移场 (3,D,H,W)，体素单位（常数，不参与梯度）
      coef_r : 残差系数（参与梯度）
      step   : 计算 det(J) 的下采样倍率；默认自动（把最长边降到 ~64）

    实现：在**下采样网格**上做中心差分求 ∂d/∂x（体素坐标），
    则 J = I + ∂d/∂x，det 用 3×3 行列式展开。下采样使成本可控。
    """
    D, H, W = shape
    if step is None:
        step = max(1, max(D, H, W) // 64)
    d = d_base + disp_full(coef_r, theta, KR, shape).squeeze(0)
    if step > 1:
        d = F.avg_pool3d(d[None], step, stride=step)[0]
    if min(d.shape[1:]) < 5:
        return d.sum() * 0.0
    # 中心差分（体素单位；通道顺序为 x,y,z）
    gx = (d[:, 1:-1, 1:-1, 2:] - d[:, 1:-1, 1:-1, :-2]) * 0.5
    gy = (d[:, 1:-1, 2:, 1:-1] - d[:, 1:-1, :-2, 1:-1]) * 0.5
    gz = (d[:, 2:, 1:-1, 1:-1] - d[:, :-2, 1:-1, 1:-1]) * 0.5
    # J[i,j] = δ_ij + ∂d_i/∂x_j，其中 (gx,gy,gz)[i] = ∂d_i/∂x
    a, b, c = 1 + gx[0], 1 + gy[1], 1 + gz[2]
    det = (a * b * c
           + gx[1] * gy[2] * gz[0] + gx[2] * gy[0] * gz[1]
           - gx[2] * b * gz[0] - gx[1] * gy[0] * c - a * gy[2] * gz[1])
    return torch.clamp(-det, min=0).pow(2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['dirlab', 'creatis'], default='dirlab',
                    help='DIRLAB 用例编号 1–10；CREATIS 用例编号 0–5（每例 10 相位、'
                         '每相位 100+ landmark）')
    ap.add_argument('--case', type=int, default=1)
    ap.add_argument('--down', type=int, default=2, choices=[4, 2, 1])
    ap.add_argument('--mask', choices=['none', 't00', 'union'], default='union')
    ap.add_argument('--metric', choices=['global', 'local', 'mind', 'ls', 'lsg'], default='local')
    ap.add_argument('--res-metric', choices=['global', 'local', 'mind', 'ls', 'lsg'],
                    default='local')
    ap.add_argument('--local-mask-mode', choices=['select', 'weight'], default='select',
                    help="局部 NCC 的掩膜聚合：select=只留 ≥50%% 在掩膜内的窗口；"
                         "weight=按掩膜占比加权（小掩膜更稳）")
    ap.add_argument('--norm', choices=['minmax', 'robust'], default='robust')
    ap.add_argument('--reg-scale', type=float, default=1.0)
    ap.add_argument('--iters1', type=int, default=0)
    ap.add_argument('--iters2', type=int, default=0)
    ap.add_argument('--res-iters', type=int, default=400)
    ap.add_argument('--lr', type=float, default=0.0)
    ap.add_argument('--phases-per-step', type=int, default=2)
    ap.add_argument('--affine-first-iters', type=int, default=0)
    ap.add_argument('--affine', type=int, default=1)
    ap.add_argument('--enc-down', type=int, default=4, choices=[1, 2, 4, 8])
    ap.add_argument('--res-down', type=int, default=0)
    ap.add_argument('--ncc-stride', type=int, default=0)
    ap.add_argument('--win-mm', type=float, default=30.0)
    ap.add_argument('--amp', type=int, default=-1)
    ap.add_argument('--ref-mm', type=float, default=2.0)
    ap.add_argument('--coef-scale', type=float, default=0.05)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--no-cnn', type=int, default=0)
    ap.add_argument('--mind-weight', type=float, default=1.0)
    ap.add_argument('--res-reg-scale', type=float, default=1.0,
                    help='残差阶段正则强度倍率（1.0=原行为）。⚠️ `--reg-scale` 不影响残差阶段')
    ap.add_argument('--jac-weight', type=float, default=0.0,
                    help='det(J) hinge 惩罚权重（0=关闭）。用于抑制折叠')
    ap.add_argument('--holdout-phases', type=str, default='',
                    help='C3 整合：留出（训练中完全不可见）的相位索引，逗号分隔，如 "2,7"。'
                         '留出相位用于计算 NRMS 检测门槛；θ=0 恒为重合，不可留出')
    ap.add_argument('--save-nrms', type=int, default=0,
                    help='1=保存逐体素局部 NRMS 可靠性图 (.npy)')
    ap.add_argument('--init-coef', type=str, default='')
    ap.add_argument('--init-coef-mm', type=float, default=0.0,
                    help='--init-coef 文件对应的分辨率(mm)；>0 时做体素单位换算')
    ap.add_argument('--save-coef', type=str, default='')
    ap.add_argument('--tag', type=str, default='v2')
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'results', 'pmr_v2'))
    ap.add_argument('--save-dvf', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cudnn-benchmark', type=int, default=-1,
                    help='-1=沿用 pmr_fast 的默认(True)；0=关闭。'
                         '关闭后配合固定 seed 可得到可复现结果（benchmark=True 会引入'
                         '运行间波动，实测 SD≈6%%，见 docs/17 C56）')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global LOCAL_MASK_MODE
    LOCAL_MASK_MODE = args.local_mask_mode
    if args.cudnn_benchmark >= 0:
        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        print(f'  [cudnn] benchmark = {bool(args.cudnn_benchmark)}', flush=True)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    cn, K, KR, K_a = args.case, args.K, 4, 2
    # 输出文件名里的病例标识：CREATIS 用 cr{n}，避免与 DIRLAB 的 case{n} 冲突
    caseid = f'case{cn}' if args.dataset == 'dirlab' else f'cr{cn}'
    down, scale = args.down, {1: 1.0, 2: 0.5, 4: 0.25}[args.down]
    iters1 = args.iters1 if args.iters1 > 0 else int(round(1200 * scale))
    iters2 = args.iters2 if args.iters2 > 0 else int(round(1200 * scale))
    lr1 = args.lr if args.lr > 0 else 2e-3 * scale
    res_down = args.res_down if args.res_down > 0 else (4 if down == 1 else 2)
    amp = (down == 1) if args.amp < 0 else bool(args.amp)

    print(f'[PMR-v2 {args.dataset} {caseid}] {down}mm | mask={args.mask} metric={args.metric} '
          f'norm={args.norm} reg×{args.reg_scale} | iters {iters1}/{iters2}/{args.res_iters} '
          f'| lr {lr1:.1e} | phases/step {args.phases_per_step} | enc_down {args.enc_down} '
          f'res_down {res_down}', flush=True)

    t_start = time.time()
    imgs, sitk_imgs = load_imgs_v2(cn, down, args.norm, args.dataset)
    spacing = np.array(sitk_imgs[0].GetSpacing())
    img0 = imgs[0].to(device)
    shape = (1, 1) + tuple(img0.shape[2:])
    D, H, W = shape[2], shape[3], shape[4]
    S = torch.tensor([W, H, D], device=device)
    g = grid_(shape)
    thetas = [2 * np.pi * i / 10 for i in range(10)]
    coords = make_coords(shape, K_a)
    n_vox = D * H * W

    # ---- C3 整合：留出相位（训练中完全不可见）----
    holdout = sorted({int(x) for x in args.holdout_phases.split(',') if x.strip()})
    if 0 in holdout:
        sys.exit('🔴 θ=0 是参考相位（d(x,0)≡0），不能留出')
    train_phases = [i for i in range(10) if i not in holdout]
    if holdout:
        print(f'  [C3] 留出相位 {holdout}；训练相位 {train_phases} '
              f'（{len(train_phases)} 个 ≥ 基函数数 8，模型仍可辨识）', flush=True)

    mask = build_mask(cn, down, shape, args.mask, device, args.dataset)
    if mask is not None:
        print(f'  肺掩膜({args.mask}) 占比 {float(mask.mean())*100:.1f}%', flush=True)
    win = max(5, int(round(args.win_mm / spacing[0])))
    ncc_stride = args.ncc_stride if args.ncc_stride > 0 else max(1, win // 2)
    l2_phys = (float(spacing[0]) / args.ref_mm) ** 2
    ws, wl = 1.0 * args.reg_scale, 0.1 * args.reg_scale
    ws2, wl2_ = 0.3 * args.reg_scale, 0.05 * args.reg_scale
    f_stats = pool_stats(img0, win, ncc_stride) if 'local' in (args.metric, args.res_metric) else None
    f_mind = mind_desc(img0) if 'mind' in (args.metric, args.res_metric) else None
    f_ls = (ls_stats(img0, win, mask)
            if ('ls' in (args.metric, args.res_metric)
                or 'lsg' in (args.metric, args.res_metric)) else None)
    if f_ls is not None:
        print(f'  局部相似性属性(C2/Fomel&Jin2009)：sigma={max(0.6, win*LS_SIGMA_FRAC):.2f}vox '
              f'(win={win}), e_norm={f_ls[3]:.3e}', flush=True)

    coef_in0 = F.avg_pool3d(img0, args.enc_down) if args.enc_down > 1 else img0
    net = CoefNetFast(out_ch=3 * K * 2, affine=bool(args.affine), K_a=K_a,
                      in_down=args.enc_down, scale=args.coef_scale).to(device)
    if args.no_cnn:
        with torch.no_grad():
            cshape = tuple(net(coef_in0)[0].shape[2:])
        net = DirectCoef(3 * K * 2, cshape, K_a=K_a, affine=bool(args.affine)).to(device)

    coef_warm = None
    if args.init_coef and os.path.exists(args.init_coef):
        cw = torch.from_numpy(np.load(args.init_coef)).float().to(device)
        with torch.no_grad():
            cs = tuple(net(coef_in0)[0].shape[2:])
        # 跨分辨率缩放：disp_full 返回的是**体素单位**位移。物理位移 = coef·spacing，
        # 故从 spacing_old 的网格搬到 spacing_new 时须乘 (spacing_old/spacing_new)。
        s_old, s_new = args.init_coef_mm, float(spacing[0])
        if s_old > 0 and abs(s_old - s_new) > 1e-6:
            cw = cw * (s_old / s_new)
            print(f'  [warm] 分辨率换算 ×{s_old/s_new:.3f} '
                  f'({s_old}mm -> {s_new}mm, 体素单位)', flush=True)
        if tuple(cw.shape[2:]) != cs:
            cw = F.interpolate(cw, size=cs, mode='trilinear', align_corners=True)
        coef_warm = cw
        print(f'  [warm] 载入周期系数 {tuple(cw.shape)} -> {cs}', flush=True)

    # ---- 仿射预对齐 ----
    if args.affine_first_iters > 0 and args.affine:
        print(f'  仿射预对齐 {args.affine_first_iters} iter', flush=True)
        aff_p = torch.zeros(K_a * 2 * 12, device=device, requires_grad=True)
        opt_a = torch.optim.Adam([aff_p], lr=1e-2)
        for it in range(args.affine_first_iters):
            opt_a.zero_grad(set_to_none=True)
            d = affine_disp_b(aff_p, thetas[5], coords, K_a).squeeze(0)
            loss = masked_ncc(warp(imgs[5].to(device), g, d, S), img0, mask)
            loss.backward(); opt_a.step()
            if (it + 1) % 200 == 0:
                print(f'    AFF iter {it+1:4d} loss {float(loss):+.5f}', flush=True)
        with torch.no_grad():
            net.fc[-1].bias.copy_(aff_p.data)

    print(f'  阶段1（{args.metric} 粗对齐）: {iters1} iter, lr={lr1:.1e}', flush=True)
    train_stage(net, imgs, img0, coef_in0, K, thetas, S, g, shape, iters1, args.metric,
                ws, wl, lr1, 'S1 ', win, coords, K_a, amp, ncc_stride, 8, mask=mask,
                f_stats=f_stats, f_mind=f_mind, phases_per_step=args.phases_per_step,
                mind_w=args.mind_weight, l2_phys=l2_phys, gen=gen, f_ls=f_ls,
                train_phases=train_phases)
    print(f'  阶段2（{args.metric} 精调）: {iters2} iter', flush=True)
    train_stage(net, imgs, img0, coef_in0, K, thetas, S, g, shape, iters2, args.metric,
                ws2, wl2_, lr1 * 0.3, 'S2 ', win, coords, K_a, amp, ncc_stride, 8, mask=mask,
                f_stats=f_stats, f_mind=f_mind, phases_per_step=args.phases_per_step,
                mind_w=args.mind_weight, l2_phys=l2_phys, gen=gen, f_ls=f_ls,
                train_phases=train_phases)

    with torch.no_grad():
        coef_c, aff = net(coef_in0)
        coef_c = coef_c.detach() + (coef_warm if coef_warm is not None else 0)
        aff = aff.detach() if aff is not None else None
    if args.save_coef:
        np.save(args.save_coef, coef_c.cpu().numpy())
        print(f'  [save-coef] {args.save_coef} {tuple(coef_c.shape)}', flush=True)

    d_base_list = []
    with torch.no_grad():
        for th in thetas:
            d = disp_full(coef_c, th, K, (D, H, W)).squeeze(0)
            if aff is not None:
                d = d + affine_disp_b(aff, th, coords, K_a).squeeze(0)
            d_base_list.append(d.detach())

    rshape = (max(1, D // res_down), max(1, H // res_down), max(1, W // res_down))
    coef_r = torch.zeros(1, 3 * KR * 2, *rshape, device=device, requires_grad=True)
    print(f'  残差阶段（{args.res_metric}, KR={KR}, 粗网格 {rshape}, {args.res_iters} iter）', flush=True)
    train_residual(coef_r, imgs, img0, d_base_list, thetas, S, g, KR, (D, H, W),
                   args.res_iters, args.res_metric, mask, f_stats, f_mind, win,
                   ncc_stride, 1e-2, args.mind_weight,
                   phases_per_step=args.phases_per_step, gen=gen,
                   reg_scale=args.res_reg_scale, jac_weight=args.jac_weight, f_ls=f_ls,
                   train_phases=train_phases)

    # ---- C3 整合：逐相位 NRMS（留出相位 vs 训练相位）----
    # 这正是地震「检测门槛」在我们这里的可执行形式：不需要重复采集，
    # 因为相位流形能对**没见过的相位**做预测，θ 维本身就是重复轴。
    # 🔴 留出相位 NRMS 测的是**模型沿 θ 外推的预测残差**，不是采集不可重复性（红线 X-17）。
    nrm = {}
    nrm_map = None
    with torch.no_grad():
        for i in range(1, 10):
            d_i = d_base_list[i] + disp_full(coef_r, thetas[i], KR, (D, H, W)).squeeze(0)
            w_i = warp(imgs[i].to(device), g, d_i, S)
            nrm[i] = nrms(w_i, img0, mask)
            if args.save_nrms and i == (holdout[0] if holdout else 5):
                nl = nrms_local(w_i, img0, sigma=max(1.0, 2.0 / float(spacing[0])))
                nrm_map = (nl * (mask if mask is not None else 1.0)).cpu().numpy().astype(np.float32)
            del d_i, w_i
    tr_v = [nrm[i] for i in range(1, 10) if i not in holdout]
    ho_v = [nrm[i] for i in holdout]
    nrm_extra = {
        'holdout_phases': holdout,
        'nrm_per_phase': {str(k): round(v, 5) for k, v in nrm.items()},
        'nrm_train_mean': round(float(np.mean(tr_v)), 5) if tr_v else None,
        'nrm_holdout_mean': round(float(np.mean(ho_v)), 5) if ho_v else None,
        'nrm_ratio': (round(float(np.mean(ho_v) / np.mean(tr_v)), 4)
                      if (tr_v and ho_v and np.mean(tr_v) > 1e-9) else None),
    }
    if holdout:
        print(f"  [C3] NRMS 训练相位均值 {nrm_extra['nrm_train_mean']} | "
              f"留出相位均值 {nrm_extra['nrm_holdout_mean']} | "
              f"比值 {nrm_extra['nrm_ratio']}", flush=True)
    if nrm_map is not None:
        np.save(os.path.join(args.out, f'{args.tag}_{caseid}_{down}mm_nrmsmap.npy'), nrm_map)

    # ---- 评估 ----
    lm0, lm5 = load_lm_dataset(cn, args.dataset)
    init = float(np.linalg.norm(lm0 - lm5, axis=1).mean())
    with torch.no_grad():
        d50_m = d_base_list[5]
        d50_t = d50_m + disp_full(coef_r, thetas[5], KR, (D, H, W)).squeeze(0)
        d_m = d50_m.cpu().numpy() * spacing[:, None, None, None]
        d_t = d50_t.cpu().numpy() * spacing[:, None, None, None]
        dd = (d50_t * torch.as_tensor(spacing, device=device, dtype=torch.float32).view(3, 1, 1, 1))
        # 🔴 torch.quantile 对输入张量有大小上限（1 mm 的大病例会触发
        #    "quantile() input tensor is too large"）。改为在子样本上求分位数。
        mag = dd.norm(dim=0)
        mean_d = float(mag.mean())
        flat = mag.flatten()
        if flat.numel() > 8_000_000:
            sel = torch.randperm(flat.numel(), device=flat.device)[:8_000_000]
            flat = flat[sel]
        p95_d = float(torch.quantile(flat, 0.95))
        full0 = d_base_list[0] + disp_full(coef_r, thetas[0], KR, (D, H, W)).squeeze(0)
        sp_t = torch.as_tensor(spacing, device=device, dtype=torch.float32).view(3, 1, 1, 1)
        clo0 = float((full0 * sp_t).abs().max())
        # 2π 闭合（用周期基直接验证）
        d2pi = (disp_full(coef_c, 2 * np.pi, K, (D, H, W)).squeeze(0)
                + (affine_disp_b(aff, 2 * np.pi, coords, K_a).squeeze(0) if aff is not None else 0)
                + disp_full(coef_r, 2 * np.pi, KR, (D, H, W)).squeeze(0))
        clo2 = float(((d2pi - full0) * sp_t).abs().max())
    tre_m = float(np.linalg.norm(lm0 + sample_trilinear(d_m, lm0, spacing) - lm5, axis=1).mean())
    tre_t = float(np.linalg.norm(lm0 + sample_trilinear(d_t, lm0, spacing) - lm5, axis=1).mean())

    res = {'impl': 'pmr_v2', 'dataset': args.dataset, 'case': cn, 'caseid': caseid,
           'down_mm': down, 'mask': args.mask, 'metric': args.metric,
           'res_metric': args.res_metric, 'norm': args.norm, 'reg_scale': args.reg_scale,
           'iters1': iters1, 'iters2': iters2, 'res_iters': args.res_iters, 'lr1': lr1,
           'phases_per_step': args.phases_per_step, 'enc_down': args.enc_down,
           'res_down': res_down, 'win': win, 'ncc_stride': ncc_stride, 'K': K,
           'local_mask_mode': LOCAL_MASK_MODE, 'affine_first_iters': args.affine_first_iters,
           'res_reg_scale': args.res_reg_scale, 'jac_weight': args.jac_weight,
           'mask_frac': float(mask.mean()) if mask is not None else None,
           'n_vox': int(n_vox), 'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1024**3, 2)
           if device == 'cuda' else 0.0,
           'init_tre': init, 'tre_manifold_STD': tre_m, 'tre_total_STANDARD': tre_t,
           'mean_disp_mm_t50': round(mean_d, 4), 'p95_disp_mm_t50': round(p95_d, 4),
           'closure_0': clo0, 'closure_2pi': clo2, 'time_s': round(time.time() - t_start, 1)}
    res.update(nrm_extra)
    print(f'\n[{caseid} {down}mm PMR-v2] 初始 {init:.2f} | 流形 {tre_m:.3f} | '
          f'+残差 {tre_t:.3f} mm (标准口径) | 平均|d| {mean_d:.2f}mm')
    print(f'闭合: |d(0)|max={clo0:.2e}  |d(2π)-d(0)|max={clo2:.2e} mm | '
          f'峰值显存 {res["peak_mem_gb"]}GB | {res["time_s"]:.0f}s')

    fn = os.path.join(args.out, f'{args.tag}_{caseid}_{down}mm.json')
    json.dump(res, open(fn, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'已保存 {fn}', flush=True)

    if args.save_dvf:
        with torch.no_grad():
            ds = [((d_base_list[i] + disp_full(coef_r, thetas[i], KR, (D, H, W)).squeeze(0))
                   .cpu().numpy() * spacing[:, None, None, None]).astype(np.float32)
                  for i in range(10)]
        np.save(os.path.join(args.out, f'{args.tag}_{caseid}_{down}mm_dvf.npy'), np.stack(ds))
    return res


if __name__ == '__main__':
    main()
