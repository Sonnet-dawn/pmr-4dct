# PMR — Phase-Manifold Registration for 4D-CT

Memory-scalable deformable registration of 4D-CT, targeting consumer GPUs.

* **144 million voxels** (10 phases at 1 mm) at a measured peak of **2.4-3.3 GB** of GPU
  memory, on an 8 GB laptop card.
* Exact **loop closure** by construction: `d(x,0) = d(x,2π) = 0` is an algebraic
  identity, not a penalty.
* A verification suite that runs with every training job — and that has caught **seven**
  real defects in this codebase, including two that were completely silent
  (`docs/verification.md`).

> **Scope.** This package is about *scalability, verification and reproducibility*.
> It does **not** claim state-of-the-art registration accuracy; see `LIMITATIONS.md`.
> On DIR-Lab the 10-case mean TRE is 1.597 mm; published results reach 1.0-1.3 mm.

## Install

```bash
git clone <repo-url> && cd pmr
pip install -r requirements.txt
```

## Data

Neither dataset is redistributed (see each dataset's terms). Point the code at your
copy with the `PMR_DATA_ROOT` environment variable:

```bash
export PMR_DATA_ROOT=/path/to/data          # Linux / macOS
set PMR_DATA_ROOT=<drive>:\path	o\data     # Windows
```

Expected layout:

```
$PMR_DATA_ROOT/DIRLAB/mha/case1/case1_T00_R.mha ... case1_T90_R.mha
$PMR_DATA_ROOT/DIRLAB/points/case1/case1_300_T00_xyz_R.txt
                                   case1_300_T50_xyz_R.txt
$PMR_DATA_ROOT/CREATIS/0/00_R.mha ... 90_R.mha
$PMR_DATA_ROOT/CREATIS/0/00.pts   ... 90.pts
```

**Two datasets are supported** (`--dataset {dirlab,creatis}`). They differ in ways that
have silently broken implementations before, so both are handled explicitly:

| | DIRLAB | CREATIS |
|---|---|---|
| Cases | 1–10 | 0–5 |
| Landmarks | 300, at T00/T50 only | 100–113; **all 10 phases for cases 0–2, only T00/T50 for cases 3–5** |
| Mean displacement T00→T50 | ~2–11 mm | **5.7–14.0 mm** (max 32 mm) |
| Voxels/phase at 1 mm | 14–79 M | **60–116 M** |
| `.mha` intensity | HU **+ 1024** | **raw HU** (offset 0) |
| Image origin | (0, 0, 0) | **non-zero**, e.g. (−250, −250, −164.5) |

> See `LIMITATIONS.md` §10 for why these differences matter and
> `verification/verify_resample_geometry.py` for the regression test that locks them down.

## Quickstart

```bash
# 2 mm working resolution (~5 min/case on an 8 GB laptop GPU)
python src/pmr_v2.py --case 1 --down 2 --mask union --metric local \
    --res-reg-scale 10 --cudnn-benchmark 0

# 1 mm full resolution (only for cases whose volume fits; see LIMITATIONS)
python src/pmr_v2.py --case 1 --down 1 --mask union --metric local \
    --enc-down 8 --res-reg-scale 10 --cudnn-benchmark 0

# second dataset (CREATIS cases are 0-5)
python src/pmr_v2.py --dataset creatis --case 0 --down 2 --res-reg-scale 10 \
    --cudnn-benchmark 0
```

Every run writes a JSON beside the result containing all hyper-parameters, peak GPU
memory, wall-clock time and the closure residuals, so any reported number is traceable.

## Key options

| Flag | Default | Why it matters |
|---|---|---|
| `--dataset {dirlab,creatis}` | `dirlab` | Which 4D-CT dataset. The two differ in HU offset, image origin and landmark phase coverage — all handled explicitly. |
| `--mask {none,t00,union}` | `union` | Spatial support of the similarity. On DIR-Lab, whole-image support biases the solution toward small displacement; with a mask the 10-case mean TRE improves from 2.335 mm to 1.598 mm (**+31.6%**, all 10 cases). |
| `--norm {robust,minmax}` | `robust` | `robust` clips to the 0.5–99.5 percentile. The DIR-Lab volumes contain isolated values near 13 400 against a background near 1 000; min–max scaling compresses lung contrast and costs **+72.9%** TRE. |
| `--res-reg-scale` | 1.0 | Scale of the **residual-stage** penalty. The global `--reg-scale` does **not** reach this stage. Setting it to 10 reduces folding by **17–57×** and improves TRE by ~5.6%. |
| `--cudnn-benchmark {0,1}` | (upstream `True`) | Pass **0** for reproducibility. With the upstream default, an identical configuration varies by **5.8% (SD)** run to run; with 0 it is **0.2%**. |
| `--metric {global,local,mind,ls,lsg}` | `local` | `ls`/`lsg` replace box-window local NCC with a **shaping-regularised Gaussian** window (`σ = win/√12`, so the kernel has the same second moment as the box it replaces). `ls` additionally weights each window by the reference's local structural significance. |
| `--holdout-phases` | (none) | Comma-separated phase indices excluded from training, used to measure how well the periodic phase model predicts a phase it has never seen. |
| `--phases-per-step` | 2 | Phases sampled per optimiser step. |
| `--local-mask-mode {select,weight}` | `select` | How per-window correlations are aggregated at the mask boundary. The choice materially changes the measured benefit of masking, so it is exposed rather than fixed. |
| `--jac-weight` | 0.0 | Optional `det(J)` hinge penalty. Off by default; the residual-stage scale above is the cheaper folding control. |
| `--enc-down` | 4 | Encoder input downsampling. **Use 8 at 1 mm.** |

## Reproducibility warning

`torch.backends.cudnn.benchmark = True` (the upstream default) makes algorithm
selection non-deterministic. We measured a **5.8% standard deviation and 18.6% range**
across five runs of an *identical* configuration on the hardest DIR-Lab case.
Pass `--cudnn-benchmark 0` to reduce this to **0.2%**. **Effects below ~5% are not
interpretable without this control.**

## Verification

The suite is not decoration: **it has caught seven real defects in this codebase**, listed
in `docs/verification.md`. Two of them were silent — they produced plausible-looking output
while corrupting the result. Run it before trusting any number.

| Command | Checks |
|---|---|
| `python verification/verify_exact_basis.py` | How many temporal basis functions are required (9, not 8) |
| `python verification/verify_projection_exact.py` | The 9-term periodic basis spans anchored 10-phase data exactly |
| `python verification/jacobian_stats.py --selftest` | Analytic Jacobian self-test (scaling / shear / fold) |
| `python verification/verify_resample_geometry.py` | **Down-sampling preserves origin/direction and keeps the mask voxel-aligned** (the defect that silently corrupted the second dataset) |
| `python verification/probe_hu.py` | Intensity calibration of the input volumes |
| `python verification/verify_main_run.py <tag>` | Per-case self-consistency of a result set |

### Why the second dataset is itself a test

Every defect above that involved data handling was invisible on DIR-Lab alone, because
DIR-Lab's image origin happens to be `(0, 0, 0)` and its stored intensities happen to be
`HU + 1024`. Adding a second dataset with a non-zero origin and raw HU exposed them
immediately. **Changing the dataset is one of the most effective verification steps there
is** — more so than adding another unit test.

## Memory safety

`run_v2_sweep.py` caps concurrency from the measured per-case peak memory
(2.8 GB for the large DIR-Lab cases, 1.0 GB otherwise) and a runtime watchdog waits
for free GPU memory before each job. **Do not raise the concurrency manually** —
running four large cases concurrently exhausted an 8 GB card and destabilised the host.

## Citation

See `CITATION.cff` (DOI to be assigned on release).

## License

MIT for the code. The DIR-Lab dataset has its own terms and is not included.
