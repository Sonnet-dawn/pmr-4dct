# Paper B · 复现手册

> 目标：**每一个进入稿件的数字，都能在本文件里找到产生它的确切命令与输出文件。**
> 所有命令的工作目录为项目根目录（含 `pmr_v2.py` 的那一层）。
> Python 解释器：`python`（torch 2.8.0+cu128 / SimpleITK / scipy）。

---

## 0. 环境

```powershell
python -c "import torch, SimpleITK, scipy, numpy; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

实测环境：RTX 5060 Laptop **8 GB**、24 核、31 GB RAM、Windows。

---

## 1. 数据前提

DIRLAB 放在 `../reference_4/data/DIRLAB/`，结构：

```
mha/case{1..10}/case{n}_T{p}0_R.mha          p = 0..9
points/case{n}/case{n}_300_T00_xyz_R.txt
points/case{n}/case{n}_300_T50_xyz_R.txt
```

**协议核实**（Paper B 表 B5 的来源）：

```powershell
python probe_hu.py            # -> HU = 存储值 − 1024；体素 1 mm 各向同性；FOV 14.5–84.0 L
python check_mask_volume.py   # -> 肺体积 2459–5884 mL
python verify_identity_tre.py # -> identity TRE 10 例均值 8.461 mm（与已发表初始误差一致）
```

---

## 2. 表 B1 · 2 mm 显存/耗时

```powershell
python peek_pmr.py            # 直接打印 results/fast_phase2/*_2mm.json 的关键字段
```
来源文件：`results/fast_phase2/s1f_dirlab_case{1..10}_2mm.json`（字段 `peak_mem_gb`、`time_s`、`n_vox`）。

**异质性审计**（必须一起跑，用于在稿件里声明这批数据的适用范围）：

```powershell
python audit_pmr_results.py
```
预期输出：`cases: [6, 7, 8, 9, 10]` —— 这 5 例缺少 `coef_scale`/`l2_phys`/`K` 字段，
来自早一版实现。**因此表 B1 只用于规模/显存趋势，不用于最终精度表。**

---

## 3. 表 B2 · 1 mm 探针

```powershell
python pmr_fast.py --dataset dirlab --case 1 --max-down 1 --enc-down 8 `
    --iters1 800 --iters2 800 --res-iters 200 --tag probe1mm_ws8 `
    --out results\probe_1mm --save-dvf 0
```
来源：`results/probe_1mm/probe1mm_ws8_dirlab_case1_1mm.json`
（`peak_mem_gb = 2.4`、`n_vox = 14453440`、`time_s = 1417.6`）。

对照（`enc_down=4`）：`probe1mm_ws_dirlab_case1_1mm.json`。

> ⚠️ 1 mm 下 `--save-dvf 1` 会写出 1.65 GB/例，务必显式给 `--save-dvf 0`。

---

## 4. 表 B3 · 粗网格等价性验证

```powershell
python verify_exact_basis.py         # 时间基完备性：8 函数残差 0.928 / 9 函数残差 8.9e-16, cond 3.35
python jacobian_stats.py --selftest  # 解析 Jacobian 自检 -> PASS
```
前向一致性对照在 `results/fast_validate/`。

---

## 5. 表 B4 · 验证套件

```powershell
python jacobian_stats.py --selftest     # 期望: "Jacobian self-test: PASS"
python verify_exact_basis.py            # 期望: 9x9 矩阵 rank 9, cond 3.35, 残差 8.9e-16
python lung_mask.py                     # 期望: 10 例肺占比，含绝对体积
python check_mask_volume.py             # 期望: 均值 3852 mL
```

闭合断言是**每次训练自动执行**的，无需单独命令；看运行输出末行：

```
闭合: |d(0)|max=0.00e+00  |d(2π)-d(0)|max=6.46e-07 mm
```

---

## 6. 掩膜机制的诊断证据（表 B4 / `04_claims_and_evidence.md` B-16、B-17）

```powershell
# 6.1 成对配准的收敛解是小位移解（loss 改善而 |d| 下降）
python diag_pairwise.py --cases 1 --down 2 --variants base

# 6.2 掩膜假设的跨例验证（本包与 Paper A 共用的关键证据）
python test_mask_hypothesis.py --cases 2,3,4,5,6,7,8,9,10 --iters 1200
```
输出：`results/diag_pairwise/mask_hypothesis.json`、`results/diag_pairwise/*.json`。

---

## 7. v2 变体筛选

```powershell
# 第一轮：13 个变体 × 3 例（1 易 / 5 中 / 8 难）
python run_v2_sweep.py --cases 1,5,8 --workers 3 --tag screen
# 重新汇总（不重跑）
python run_v2_sweep.py --cases 1 --workers 1 --tag screen --variants base
```
输出：`results/pmr_v2/screen_<变体>_case<n>_2mm.json`；日志 `results/pmr_v2/logs/screen_w*.log`。

---

## 8. 单例端到端（论文第 5 节 Illustrative examples）

```powershell
python pmr_v2.py --case 1 --down 2 --mask union --metric local --tag demo
python pmr_v2.py --case 8 --down 2 --mask union --metric local --phases-per-step 2 --tag demo
python pmr_v2.py --case 1 --down 1 --mask union --metric local --enc-down 8 --tag demo1mm
```

---

## 9. 重建全部表格

```powershell
python make_all_tables.py     # -> results/TABLES.md（含自动审计）
python survey_results.py      # 清点 results/ 可用数据
python paperB_gather.py       # Paper B 数据抽取
```

---

## 10. FAILED / 已废弃的命令（不要重复尝试）

| 命令 | 现象 | 原因 |
|---|---|---|
| 任何用 `urllib.request` 的下载 | `SSLError: [ASN1: NOT_ENOUGH_DATA]` | 本机系统证书库损坏；**一律改用 `requests`** |
| 含中文的 `.ps1`（非 BOM UTF-8） | `Unexpected token '}'` | PowerShell 5.1 编码问题；**改写成 `.py`** |
| 硬编码旧盘符（如 `<OLD_DRIVE>:...`）的脚本 | `The file ... does not exist` | 项目树曾换过所在盘符；已由 `fix_stale_paths.py` 修正 |
| `--save-dvf 1` @1 mm | 写出 1.65 GB | 预期行为，非错误 |
| 把 `ct < -500` 当肺阈值 | 选中 0% 体素 | 该 `.mha` 是 **HU + 1024**；应用 `probe_hu.py` 的标定 |
