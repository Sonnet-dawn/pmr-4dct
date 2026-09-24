# PMR — Phase-Manifold Registration for 4D-CT

[![verification](https://github.com/Sonnet-dawn/pmr-4dct/actions/workflows/verify.yml/badge.svg)](https://github.com/Sonnet-dawn/pmr-4dct/actions/workflows/verify.yml)

Memory-scalable deformable registration of 4D-CT, targeting consumer GPUs.

* **144 million voxels** (10 phases at 1 mm) at a measured peak of **2.4-3.3 GiB** of GPU
  memory, on an 8 GiB laptop card.
* Exact **loop closure** by construction: `d(x,0) = d(x,2π) = 0` is an algebraic
  identity, not a penalty.
* A verification suite that runs with every training job — and that has caught **fourteen**
  documented defects in this codebase, **eleven** of which were completely silent
  (`docs/verification.md`).

> The CI badge covers the **data-free** subset of the suite only: the DIR-Lab and CREATIS
> volumes are not redistributable, so any check needing them would fail on a runner. The
> data-dependent checks are run locally and are listed in `docs/verification.md`.

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
# 2 mm working resolution (~5 min/case on an 8 GiB laptop GPU)
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
| `--mask {none,t00,union}` | `union` | Spatial support of the similarity. On DIR-Lab, whole-image support biases the solution toward small displacement; against the matched control (`--res-reg-scale 1`), the 10-case mean TRE goes from 2.335 mm without a mask to 1.692 mm with one (**+27.6%**, nine of ten cases). |
| `--norm {robust,minmax}` | `robust` | `robust` clips to the 0.5–99.5 percentile. The DIR-Lab volumes contain isolated values near 13 400 against a background near 1 000; min–max scaling compresses lung contrast and costs **+63.3%** TRE (all ten cases worse). |
| `--res-reg-scale` | 1.0 | Scale of the **residual-stage** penalty. The global `--reg-scale` does **not** reach this stage. Setting it to 10 reduces folding by **17–57×** and improves TRE by ~5.6%. |
| `--cudnn-benchmark {0,1}` | (upstream `True`) | Pass **0** for reproducibility. With the upstream default an identical configuration varies by **6.2% (SD)** run to run; with 0 the ten-case mean is stable to **0.47%**, though the hardest single cases still move by **~5%**. See the warning below. |
| `--metric {global,local,mind,ls,lsg}` | `local` | `ls`/`lsg` replace box-window local NCC with a **shaping-regularised Gaussian** window (`σ = win/√12`, so the kernel has the same second moment as the box it replaces). `ls` additionally weights each window by the reference's local structural significance. |
| `--holdout-phases` | (none) | Comma-separated phase indices excluded from training, used to measure how well the periodic phase model predicts a phase it has never seen. |
| `--tre-phases` | (none) | Extra target phases (0–5 = T00…T50) to report TRE for, written to a `tre_by_phase` block. Because the 300-point DIR-Lab set covers **only T00 and T50**, intermediate phases require `--lm-set 4d75`; asking for them with the 300-point set **raises** rather than silently evaluating the wrong points. |
| `--lm-set {300,4d75}` | `300` | Which landmark set the *extra* phases use. `4d75` is DIR-Lab's 75-point, six-phase set. It never changes `tre_total_STANDARD`, which always uses the 300-point T00/T50 set. |
| `--phases-per-step` | 2 | Phases sampled per optimiser step. |
| `--local-mask-mode {select,weight}` | `select` | How per-window correlations are aggregated at the mask boundary. The choice materially changes the measured benefit of masking, so it is exposed rather than fixed. |
| `--jac-weight` | 0.0 | Optional `det(J)` hinge penalty. Off by default; the residual-stage scale above is the cheaper folding control. |
| `--enc-down` | 4 | Encoder input downsampling. **Use 8 at 1 mm.** |

## Reproducibility warning

`torch.backends.cudnn.benchmark = True` (the upstream default) makes algorithm
selection non-deterministic. Across five runs of an *identical* configuration on the
hardest DIR-Lab case we measured a standard deviation of **6.2% of the mean** and a
range of **17.0% of the mean** (0.56 mm). **Effects below ~5% are not interpretable
without this control.**

Pass `--cudnn-benchmark 0` to remove most of that spread — but not all of it, and not
uniformly. Two complete runs of an identical configuration with the flag off (every
hyper-parameter equal, `benchmark = False` in both logs) differed by **0.5–1.6%** on the
nine easier cases and by **5.3%** / **5.4%** on the two hardest, while the ten-case mean
moved by only **0.47%**. So the flag stabilises an aggregate number to about half a
percent; it does not make the hardest single cases reproducible. Every run writes the
flag's actual value and the seed into its result file.

> All memory figures in this repository are **GiB** (2³⁰ bytes), matching what
> `torch.cuda.max_memory_allocated()` reports.

## Why this exists: a measured memory–accuracy trade-off

Mainstream pairwise tools are tuned for coarse B-spline control grids, and refining the grid
is throttled by memory rather than by compute. Measured with elastix 5.3.1 on a 1 mm DIR-Lab
volume (14.45 M voxels), with the control grid set explicitly:

| B-spline grid | Control points | Peak RAM | Wall clock | Outcome |
|---:|---:|---:|---:|---|
| **8 mm** (elastix default) | 38,148 | **0.92 GiB** | 39 s | completed |
| 4 mm | 261,950 | **5.63 GiB** (6.1×) | 352 s | completed |
| 2 mm | — | **7.54 GiB** and rising | 5 s | **ran out of memory** |
| 1 mm | — | **8.51 GiB** and rising | 6 s | **ran out of memory** |

The control-point count grows as `1/g³` and the memory follows. Two caveats: the
out-of-memory cases were terminated by *our* watchdog (the figures are lower bounds, not
elastix's own reported failure), and the machine had only ~9–10 GiB free at the time.
Reproduce with `python run_elastix_memory.py --cases 1 --grids 8,4,2,1`.

> ⚠️ This is **not** a like-for-like benchmark. elastix performs one *pairwise* registration;
> this package trains a continuous model over **all ten phases**. No ratio between the two
> should be read as "faster" or "lighter".

## Verification

The suite is not decoration: **it has caught fourteen real defects in this codebase**, listed
in `docs/verification.md`. Eleven of them were silent — they produced plausible-looking output
while corrupting the result. Run it before trusting any number.

| Command | Checks |
|---|---|
| `python run_verification_suite.py --quick` | Every data-free check, one entry point (this is what CI runs) |
| `python verification/verify_exact_basis.py` | How many temporal basis functions are required (9, not 8) |
| `python verification/verify_projection_exact.py` | The 9-term periodic basis spans anchored 10-phase data exactly |
| `python verification/verify_tre_convention.py` | The TRE convention and the point-file units, against analytic known answers |
| `python verification/verify_phase_selection.py` | Phase/landmark-set consistency (asking for intermediate phases with the two-phase landmark set **raises**) |
| `python verification/verify_lapirn_shapes.py` | The two baseline networks' forward shapes, at every pyramid level |
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
(2.8 GiB for the large DIR-Lab cases, 1.0 GiB otherwise) and a runtime watchdog waits
for free GPU memory before each job. **Do not raise the concurrency manually** —
running four large cases concurrently exhausted an 8 GiB card and destabilised the host.

## Citation

See `CITATION.cff`. If you use this software, please cite the archived release rather than
the moving `main` branch.

## Archival

The source is archived **independently of GitHub** in
[Software Heritage](https://www.softwareheritage.org/), which guarantees that the exact
revision stays retrievable even if this repository moves or disappears:

| | |
|---|---|
| **DOI (cite this)** | **[10.5281/zenodo.22899963](https://doi.org/10.5281/zenodo.22899963)** |
| Repository | https://github.com/Sonnet-dawn/pmr-4dct |
| Release v1.0.2 | https://github.com/Sonnet-dawn/pmr-4dct/releases/tag/v1.0.2 |
| Zenodo record | https://zenodo.org/record/22899963 |
| SWH snapshot | `swh:1:snp:2871fbfe6e030ec739c82f1c5bdb3c855155fdf1` |
| SWH revision (v1.0.2) | `swh:1:rev:9770ffc7ee5fd7fe6e0f45ae01b318850108892c` |
| SWH revision (v1.0.0) | `swh:1:rev:be248d6d619eec5bcf04ce7fd044fe2fc177728a` |

The source is archived in two independent places: **Zenodo** (with a DOI) and
**Software Heritage** (long-term code archive, DOI-independent).

Browse: <https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/Sonnet-dawn/pmr-4dct>

## License

MIT for the code. Neither the DIR-Lab nor the CREATIS dataset is included; both have their
own terms.
