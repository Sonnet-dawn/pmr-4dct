"""
dl_baseline.py —— Paper A 的**同协议**深度学习配准基线（VoxelMorph 级 / LapIRN 级）
================================================================================
为什么要自己训而不是引用文献数字
----------------------------------
跨协议比较是本项目明令禁止的红线（`docs/17` X-6）：文献里 DIRLAB 的数字用的是
各自的预处理、各自的划分、各自的 TRE 口径。**唯一诚实的做法是在同一数据、同一
分辨率、同一 landmark 协议下自己训练并评测。**

协议（与 PMR 完全对齐）
-----------------------
  * 数据：**DIRLAB 或 CREATIS**，`_R.mha`（1 mm 各向同性），本脚本按 `--down` 降采样
  * **留一法**：`--fold k` 表示把 case k 留作测试，其余例训练
    —— 这是这两个数据集上深度学习配准的标准做法，避免"训练集即测试集"
    （DIRLAB 10 例 ⇒ 训 9；CREATIS 6 例 ⇒ 训 5）
  * 监督：**无监督**（图像相似度 + 平滑正则），与 PMR 同族
  * 相似度：**掩膜内局部 NCC**，与 PMR 的 `--metric local --mask union` 一致
  * 评测：标准口径 `|lm0 + d(lm0) − lm5|`
      - DIRLAB：300 个 landmark
      - CREATIS：`.pts`（病例 0–2 为 100–113 点；**必须减 origin**，见下）
  * 确定性：`torch.backends.cudnn.benchmark = False`

🔴 两个数据集的**约定不同**，本脚本按数据集分流（`docs/36`、`tools/check_lm_frame.py`）：
  * DIRLAB：origin=(0,0,0)，landmark 即原点相对坐标；
  * CREATIS：origin=(-250,-250,-164.5)，`.pts` 是**绝对物理坐标**，必须减 origin。
  评测统一走 `pmr_v2.load_lm_dataset(cn, dataset)` 而不是各数据集自己的加载器，
  就是为了**不可能**在这里分叉出错。

两个模型
--------
  * `voxelmorph`：单尺度 U-Net，输入 concat(fixed, moving)，输出 3 通道位移
  * `lapirn`：拉普拉斯金字塔，由粗到细逐级预测**残差**位移，专治大位移

用法
----
    python dl_baseline.py --model voxelmorph --fold 1 --down 2 --epochs 200
    python dl_baseline.py --model lapirn     --fold 1 --down 2 --epochs 200
    python dl_baseline.py --model voxelmorph --dataset creatis --fold 0 --down 2
"""

# --- path shim (injected by make_repo.py; repo layout = src/ + verification/ + drivers/) ---
import os as _os, sys as _sys
_R = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_os.path.join(_R, "src"), _R):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
try:
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# --- end path shim ---
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pmr_v2 import load_imgs_v2, build_mask, load_lm_dataset   # noqa: E402
from recompute_tre_standard import sample_trilinear            # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = False          # 确定性（与 PMR 的 --cudnn-benchmark 0 对齐）
OUTD = os.path.join(HERE, 'results', 'dl_baselines')
os.makedirs(OUTD, exist_ok=True)

# 留一法的病例编号空间：DIRLAB 是 1–10，CREATIS 是 0–5
CASES_OF = {'dirlab': list(range(1, 11)), 'creatis': list(range(0, 6))}


def case_tag(model, dataset, fold, down, tag=''):
    """结果文件名。

    🔴 DIRLAB 用 `{model}_fold{k}_...`；CREATIS **必须**用别的模式，
    否则 `tools/make_main_table.py` 里 `{model}_fold*_{down}mm.json` 的 glob
    会把 CREATIS 的折当成 DIRLAB 的折，**两个数据集的数字就会混进同一行**。
    """
    stem = f'fold{fold}' if dataset == 'dirlab' else f'cr{fold}'
    return f'{model}_{stem}_{down}mm{("_" + tag) if tag else ""}.json'



# ==================== 模型 ====================
def conv_block(ci, co):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(co, co, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))


class UNet3D(nn.Module):
    """轻量 3-D U-Net：输入 2 通道（fixed, moving），输出 `out_ch` 通道位移。"""

    def __init__(self, out_ch=3, base=16, depth=3):
        super().__init__()
        self.depth = depth
        self.enc = nn.ModuleList()
        ci = 2
        chs = []
        for d in range(depth):
            co = base * (2 ** d)
            self.enc.append(conv_block(ci, co))
            chs.append(co)
            ci = co
        self.bottom = conv_block(ci, chs[-1])
        # 解码器：第 d 层的输入通道 = 上一层解码器输出 + 对应编码器 skip 的通道
        self.dec = nn.ModuleList()
        prev = chs[-1]                       # 瓶颈层输出通道数
        for d in range(depth - 1, -1, -1):
            co = base * (2 ** d)
            self.dec.append(conv_block(prev + chs[d], co))
            prev = co
        self.out = nn.Conv3d(base, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)          # 初始位移 = 0
        nn.init.zeros_(self.out.bias)
        self.pool = nn.AvgPool3d(2)

    def forward(self, x):
        # 🔴 明确的前置条件检查。`depth` 次 `AvgPool3d(2)` 要求**每个空间维 ≥ 2^depth**；
        # 否则最深一层会出现 size=1，然后 avg_pool3d 抛出
        # "input image (T: 1 H: 2 W: 1) smaller than kernel size (kT: 2 kH: 2 kW: 2)" ——
        # 这条信息**完全看不出**真实原因（哪个维度太小、需要多大）。
        # 本项目的 DL 基线在 patch=96（96 ≥ 8）与 DIRLAB 2 mm 图像上一直满足该条件，
        # 所以这个坑此前没暴露；写在这里是为了让它在**第一次**就报清楚。
        mn = min(x.shape[2:])
        need = 2 ** self.depth
        if mn < need:
            raise ValueError(
                f'UNet3D(depth={self.depth}) 要求每个空间维 ≥ {need} 体素，'
                f'实际最小维 = {mn}（输入空间尺寸 {tuple(x.shape[2:])}）。'
                f'请增大 patch，或对输入做填充/降采样后再送入。')
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for d, dec in enumerate(self.dec):
            s = skips[-1 - d]
            x = F.interpolate(x, size=s.shape[2:], mode='trilinear', align_corners=True)
            x = dec(torch.cat([x, s], dim=1))
        return self.out(x)


class LapIRN(nn.Module):
    """拉普拉斯金字塔：由粗到细逐级预测残差位移（`--model lapirn`）。"""

    def __init__(self, levels=3, base=16):
        super().__init__()
        self.levels = levels
        self.nets = nn.ModuleList([UNet3D(3, base, depth=3) for _ in range(levels)])

    def forward(self, fixed, moving, return_all=False):
        """自底向上逐级细化。返回累积位移列表（最粗在前），单位=体素。

        🔴 修过的缺陷（2026-09-24）：`cur` 来自**上一级（更粗）**的分辨率，
        但 `_warp(m, cur)` 里的基准网格是按**当前（更细）**的 `m` 建的
        ⇒ `RuntimeError: The size of tensor a (48) must match the size of tensor b (24)`。
        20 折 lapirn **全部立刻失败**（每折 ~120 s，退出码 1），
        而驱动的 `if os.path.exists(jf)` 只把失败记成一行 `!! 失败`，
        **整批实验看起来"跑完了"**（退出码 0）—— 属于本项目的静默失败家族。

        修法：进入本级时**先把 `cur` 三线性上采样到本级网格**再用。
        这一步同时把位移的单位从"粗网格体素"换算成"细网格体素"
        （`F.interpolate` 按同样的倍数放大数值），所以不需要额外乘 scale。
        """
        outs = []
        cur = None
        for lv in range(self.levels):
            scale = 2 ** (self.levels - 1 - lv)
            f = fixed if scale == 1 else F.interpolate(
                fixed, scale_factor=1.0 / scale, mode='trilinear', align_corners=True)
            m = moving if scale == 1 else F.interpolate(
                moving, scale_factor=1.0 / scale, mode='trilinear', align_corners=True)
            if cur is not None:
                # ← 关键：先对齐到本级网格，再拿去做 warp / 相加
                cur = F.interpolate(cur, size=m.shape[2:], mode='trilinear',
                                    align_corners=True)
                m = _warp(m, cur)
            d = self.nets[lv](torch.cat([f, m], dim=1))
            if cur is not None:
                d = d + cur
            cur = d
            outs.append(d)
        return outs if return_all else outs[-1]


def _base_grid(shape, device):
    D, H, W = shape[2:]
    gz, gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device), indexing='ij')
    return torch.stack([gx, gy, gz])[None]


def _warp(img, d_vox):
    """img: (N,1,D,H,W)；d_vox: (N,3,D,H,W) 体素单位。"""
    g = _base_grid(img.shape, img.device)
    S = torch.tensor([img.shape[4], img.shape[3], img.shape[2]],
                     device=img.device, dtype=img.dtype).view(1, 3, 1, 1, 1)
    return F.grid_sample(img, (g + (d_vox * 2.0 / S)).permute(0, 2, 3, 4, 1),
                         mode='bilinear', padding_mode='border', align_corners=True)


def _smooth(d):
    """位移场梯度 L2 正则（各向同性一阶差分）。"""
    return (d[:, :, 1:] - d[:, :, :-1]).pow(2).mean() + \
           (d[:, :, :, 1:] - d[:, :, :, :-1]).pow(2).mean() + \
           (d[:, :, :, :, 1:] - d[:, :, :, :, :-1]).pow(2).mean()


def masked_local_ncc(w, f, m, win):
    """掩膜内局部 NCC（与 pmr_v2 的 local 度量同形，便于公平对比）。

    🔴 聚合方式必须用**掩膜占比加权**，不能用 pmr_v2 的 `select`（只留 ≥50% 在掩膜内的窗）。
    原因：本脚本在 **32³ 的随机小块**上训练，一个块里满足 "≥50% 在掩膜内" 的窗口常常
    **不足 16 个**，`select` 会退化成 `return 0`，于是 loss 恒为 0、梯度为 0、**模型完全不学**。
    实测症状：loss ≈ −0.003 → +0.0014（只剩平滑项在涨），TRE 无改善。
    加权聚合对小掩膜/小块稳健 —— 这正是 PMR 里 `--local-mask-mode weight` 存在的理由。
    """
    pad = win // 2
    pool = lambda t: F.avg_pool3d(t, win, stride=max(1, win // 2), padding=pad,
                                  count_include_pad=False)
    wp, fp = pool(w), pool(f)
    vw = pool(w * w) - wp * wp
    vf = pool(f * f) - fp * fp
    cc = (pool(w * f) - wp * fp) / torch.sqrt(vw.clamp(min=0) * vf.clamp(min=0) + 1e-5)
    mm = pool(m)
    tot = mm.sum()
    if float(tot) < 1e-6:
        return cc.sum() * 0.0
    return -(cc * mm).sum() / tot


def predict(model_name, model, f, mv):
    """两个模型统一的**前向入口**，返回 `(N,3,D,H,W)` 的位移张量（体素单位）。

    🔴 为什么要有这个函数（2026-09-24，缺陷 C10）：原代码在两处写了

        d = model(f, mv)[-1]        # ← 错

    作者的意图是"取金字塔的最后一级"。但 `LapIRN.forward` 在
    `return_all=False`（默认）时**返回的是张量**，不是列表 ——
    于是 `[-1]` 取的是张量在**最后一维上的切片**，位移从 `(N,3,D,H,W)` 变成
    `(N,3,D,H,W-1)`，接着 `_smooth()` 立刻抛
    `IndexError: too many indices for tensor of dimension 4`。

    这个错误的隐蔽之处：`x[-1]` 对**列表**和**张量**都是合法语法，
    只是语义完全不同；而且在 C9（金字塔内部尺寸错配）修好之前，
    代码根本走不到这一行，所以它一直没暴露。
    统一入口 + `assert` 把"必须是 5 维"这件事**钉死**在这里。
    """
    if model_name == 'voxelmorph':
        d = model(torch.cat([f, mv], dim=1))
    else:
        # 显式取"金字塔最后一级"，不依赖上游返回的是张量还是列表
        d = model(f, mv, return_all=True)[-1]
    assert d.dim() == 5, (
        f'predict() 必须返回 (N,3,D,H,W)，实际 {tuple(d.shape)}。'
        f'若你把 `model(f, mv)` 的返回值再取了 `[-1]`，那是**最后一维的切片**，不是"最后一级"。')
    assert tuple(d.shape[2:]) == tuple(f.shape[2:]), (
        f'位移空间尺寸 {tuple(d.shape[2:])} 与输入 {tuple(f.shape[2:])} 不一致')
    return d


# ==================== 数据 ====================
def load_case(cn, down, dataset='dirlab'):
    """返回 (imgs[10] 归一化图, mask, spacing)。

    🔴 图像**留在 CPU**：10 个相位最大约 10.5 Mvox × 4 B ≈ 420 MB/例，
    9 个训练例全搬上 GPU 会占约 3.8 GB —— 在 8 GiB 卡上不可接受。
    训练时只把随机 patch 搬到 GPU。掩膜很小，可以留在 GPU。

    ⚠️ CREATIS 的相位只有 10 个（`00`–`90`），与 DIRLAB 相同；差别在**尺寸与 origin**：
    病例最大到 14.5 Mvox @2 mm（DIRLAB 最大约 10 Mvox），所以 CREATIS 更吃内存。
    """
    imgs, sitk_imgs = load_imgs_v2(cn, down, 'robust', dataset)
    sp = np.array(sitk_imgs[0].GetSpacing(), dtype=np.float64)
    sh = tuple(imgs[0].shape[2:])
    m = build_mask(cn, down, (1, 1) + sh, 'union', DEV, dataset)
    return [im.cpu() for im in imgs], m, sp


def mask_voxels(m, stride=4):
    """预先取出掩膜内体素的坐标（降采样以省内存），用于**以掩膜为中心**采 patch。"""
    import numpy as _np
    idx = _np.argwhere(m[0, 0].detach().cpu().numpy() > 0.5)
    if len(idx) == 0:
        return None
    return idx[::max(1, stride)]


def random_crop(shape, size, gen, center=None):
    """随机取一个 size³ 块的偏移与尺寸。

    🔴 必须**先定偏移、再对同一偏移裁多个体积**。
    早期版本对 fixed 和 moving 各自独立调用了一次随机裁剪，于是两者来自体积里
    **两个不相关的区域** —— 局部 NCC 自然恒为 0，训练完全没有信号
    （实测症状：loss 在 0 附近抖动，TRE 无改善，且很难一眼看出）。

    `center`：可选的 (z, y, x) 中心。**强烈建议传入掩膜内体素** ——
    肺只占体积的 17–35%（DIRLAB），全图均匀采样时大部分 patch 落在无有效内容处，
    训练信号被浪费。实测见 `docs/34`。
    """
    D, H, W = shape[2:]
    sz = [min(size, D), min(size, H), min(size, W)]
    if center is not None:
        cz, cy, cx = center
        zr = lambda: int(np.clip(cz - sz[0] // 2, 0, max(0, D - sz[0])))
        yr = lambda: int(np.clip(cy - sz[1] // 2, 0, max(0, H - sz[1])))
        xr = lambda: int(np.clip(cx - sz[2] // 2, 0, max(0, W - sz[2])))
        return (zr(), yr(), xr()), sz
    z = int(torch.randint(0, max(1, D - sz[0] + 1), (1,), generator=gen))
    y = int(torch.randint(0, max(1, H - sz[1] + 1), (1,), generator=gen))
    x = int(torch.randint(0, max(1, W - sz[2] + 1), (1,), generator=gen))
    return (z, y, x), sz


def crop_at(t, off, sz, size):
    """按**给定偏移**裁剪；不足 size³ 时三线性缩放到 size³。"""
    z, y, x = off
    tp = t[:, :, z:z + sz[0], y:y + sz[1], x:x + sz[2]]
    if tuple(tp.shape[2:]) != (size, size, size):
        tp = F.interpolate(tp, size=(size, size, size), mode='trilinear',
                           align_corners=True)
    return tp


# ==================== 训练 ====================
def train(model_name, fold, down, epochs, lr, patch, lam, win_mm, seed, log_every,
          iters_per_epoch=25, dataset='dirlab'):
    torch.manual_seed(seed)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    all_cases = CASES_OF[dataset]
    if fold not in all_cases:
        sys.exit(f'🔴 {dataset} 的病例编号是 {all_cases}，收到 fold={fold}')
    train_cases = [c for c in all_cases if c != fold]
    print(f'[DL {model_name}] dataset={dataset} fold={fold} 测试例={fold}，'
          f'训练例={train_cases}（{len(train_cases)} 例），{down}mm，epochs={epochs}',
          flush=True)

    t0 = time.time()
    data = {}
    for c in train_cases:
        imgs, m, sp = load_case(c, down, dataset)
        # 预取掩膜内体素坐标：patch 以肺内为中心采样，避免浪费在无内容区域
        data[c] = (imgs, m, sp, mask_voxels(m))
        print(f'  载入 {dataset} case{c}: {tuple(imgs[0].shape[2:])} '
              f'掩膜体素 {0 if data[c][3] is None else len(data[c][3])}', flush=True)

    model = (UNet3D(3, 16, 3) if model_name == 'voxelmorph' else LapIRN(3, 16)).to(DEV)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'  参数量 {n_par/1e6:.2f} M | 设备 {DEV}', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    win = max(5, int(round(win_mm / float(data[train_cases[0]][2][0]))))
    print(f'  NCC 窗 {win} 体素 | 平滑权重 {lam} | patch {patch}³ | lr {lr}', flush=True)

    best = None
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_it = iters_per_epoch
        for it in range(n_it):
            c = train_cases[int(torch.randint(0, len(train_cases), (1,), generator=gen))]
            imgs, m, _, mvox = data[c]
            ti = int(torch.randint(1, 10, (1,), generator=gen))     # 目标相位 1..9
            # ⚠️ 先定偏移（以肺内体素为中心），再对同一偏移裁 fixed / moving / mask 三者
            ctr = None
            if mvox is not None and len(mvox) > 0:
                ctr = mvox[int(torch.randint(0, len(mvox), (1,), generator=gen))]
            off, sz = random_crop(imgs[0].shape, patch, gen, center=ctr)
            f = crop_at(imgs[0], off, sz, patch)
            mv = crop_at(imgs[ti], off, sz, patch)
            mp = crop_at(m, off, sz, patch).clamp(0, 1)
            f, mv, mp = f.to(DEV), mv.to(DEV), mp.to(DEV)   # 只把 patch 搬上 GPU
            opt.zero_grad(set_to_none=True)
            d = predict(model_name, model, f, mv)
            w = _warp(mv, d)
            loss = masked_local_ncc(w, f, mp, win) + lam * _smooth(d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach())
        ep_loss /= n_it
        if ep % log_every == 0 or ep == 1:
            print(f'    ep {ep:4d}/{epochs}  loss {ep_loss:+.5f}  '
                  f'({time.time()-t0:.0f}s, 峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB)',
                  flush=True)
        best = ep_loss
    train_time = time.time() - t0
    torch.save(model.state_dict(),
               os.path.join(OUTD, case_tag(model_name, dataset, fold, down).replace('.json', '.pt')))
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    return model, train_time, peak, n_par, best


# ==================== 评测 ====================
@torch.no_grad()
def evaluate(model_name, model, fold, down, dataset='dirlab'):
    model.eval()
    imgs, m, sp = load_case(fold, down, dataset)
    # 🔴 必须走 `load_lm_dataset`：它负责 CREATIS 的 **origin 相减**
    #    （`.pts` 是绝对物理坐标）。用各数据集自己的加载器会在这里分叉出错。
    lm0, lm5 = load_lm_dataset(fold, dataset, 0, 5)
    f = imgs[0].to(DEV)
    mv = imgs[5].to(DEV)
    base = float(np.linalg.norm(lm0 - lm5, axis=1).mean())

    d = predict(model_name, model, f, mv)
    d_np = d[0].cpu().numpy() * sp[:, None, None, None]          # 体素 → mm
    d_lm = sample_trilinear(d_np, lm0, sp)
    tre = float(np.linalg.norm(lm0 + d_lm - lm5, axis=1).mean())
    print(f'  {dataset} case{fold}: 初始 {base:.3f} → TRE {tre:.3f} mm '
          f'（降幅 {(1-tre/base)*100:.1f}%，{len(lm0)} 点）', flush=True)
    return {'case': fold, 'dataset': dataset, 'down_mm': down, 'n_points': int(len(lm0)),
            'init_tre': round(base, 4), 'tre_total_STANDARD': round(tre, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['voxelmorph', 'lapirn'], default='voxelmorph')
    ap.add_argument('--dataset', choices=['dirlab', 'creatis'], default='dirlab',
                    help='DIRLAB 病例 1–10；CREATIS 病例 0–5（每例 10 相位）')
    ap.add_argument('--fold', type=int, default=1,
                    help='留作测试的病例号（DIRLAB 1–10 / CREATIS 0–5）')
    ap.add_argument('--down', type=int, default=2, choices=[1, 2, 4])
    ap.add_argument('--epochs', type=int, default=600,
                    help='训练轮数。总迭代 = epochs × iters-per-epoch。'
                         '⚠️ 必须**训到收敛**，否则就是拿一个没调好的对手比 —— '
                         '这正是 Elastix 基线犯过的错（docs/32 F4）')
    ap.add_argument('--iters-per-epoch', type=int, default=25)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--patch', type=int, default=96, help='训练随机块边长（体素）')
    ap.add_argument('--lam', type=float, default=1.0, help='位移平滑权重')
    ap.add_argument('--win-mm', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=20)
    ap.add_argument('--device', type=str, default='', help='cuda / cpu；留空则自动')
    ap.add_argument('--tag', type=str, default='',
                    help='结果文件名后缀，用于**不覆盖**地比较不同训练预算'
                         '（如 --tag ep2400）。留空 = 主结果名。')
    args = ap.parse_args()

    global DEV
    if args.device:
        DEV = args.device
    print(f'设备: {DEV}', flush=True)

    model, tt, peak, npar, final_loss = train(
        args.model, args.fold, args.down, args.epochs,
        args.lr, args.patch, args.lam, args.win_mm,
        args.seed, args.log_every, args.iters_per_epoch, args.dataset)
    res = evaluate(args.model, model, args.fold, args.down, args.dataset)
    res.update({'model': args.model, 'dataset': args.dataset, 'epochs': args.epochs,
                'lr': args.lr,
                'patch': args.patch, 'lam': args.lam, 'seed': args.seed,
                'iters_per_epoch': args.iters_per_epoch,
                'total_iters': args.epochs * args.iters_per_epoch,
                'train_time_s': round(tt, 1), 'train_peak_gb': round(peak, 3),
                'n_params': npar, 'protocol': 'leave-one-out',
                'n_train_cases': len(CASES_OF[args.dataset]) - 1,
                'deterministic_protocol': True,
                'supervision': 'unsupervised (masked local NCC + smoothness)',
                'patch_sampling': 'centred on lung mask',
                'final_epoch_loss': round(final_loss, 6) if final_loss is not None else None})
    fn = os.path.join(OUTD, case_tag(args.model, args.dataset, args.fold, args.down,
                                     args.tag))
    json.dump(res, open(fn, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n已保存 {fn}')


if __name__ == '__main__':
    main()
