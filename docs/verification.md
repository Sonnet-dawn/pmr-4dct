# What the verification suite has caught

This file exists because "we tested it" is not evidence. Below is every real defect the
suite has caught in *this* codebase: **thirteen, of which eleven were completely silent** —
the program ran to completion, wrote a result, and the result was wrong.

Run the suite before trusting any number produced by this code:

```bash
python run_verification_suite.py              # everything, one entry point
python run_verification_suite.py --quick      # code-only checks, no data needed
```

## The thirteen defects

| # | Defect | Consequence | Caught by | Silent? |
|---|---|---|---|---|
| 1 | Stored intensity treated as HU while the files store `HU + 1024` | A threshold of `ct < -500` selected **0%** of voxels; a published fold-localisation claim was invalidated | `probe_hu.py` | **yes** |
| 2 | The residual stage's regularisation was hard-coded and unreachable from `--reg-scale` | Folding could not be reduced by tuning the documented flag; folding was 20-57x higher than assumed | `folding_check.py` | no |
| 3 | A coarse-scale result was evaluated **on the coarse grid** | Case 8 TRE was reported as 4.02 mm; on the correct grid it is 7.59 mm | `verify_coarse_tre.py` | **yes** |
| 4 | `torch.quantile` has an input-size limit | A 1 mm run finished registration, then **crashed during evaluation and wrote no JSON** — the result was silently lost | missing result file | no |
| 5 | `torch.backends.cudnn.benchmark = True` (upstream default) | Identical configurations varied by **6.2% (SD, of the mean)**, range **17.0% (of the mean)**. Two published-effect conclusions were drawn and later reversed. Turning the flag off stabilises the ten-case mean to 0.47%, but the two hardest cases still move by ~5% | `--cudnn-benchmark 0` | **yes** |
| 6 | Four concurrent large-case jobs | Exhausted an 8 GiB card and **destabilised the host machine** | runtime watchdog added afterwards | no |
| 7 | Integer down-sampling **dropped the output origin** | On a dataset whose images do not start at `(0,0,0)`, the sampling window shifted by 250 mm; out-of-bounds fill (`0`) is *soft-tissue* HU, so the corruption was invisible. Training loss never decreased; a naive reading would have concluded "the method does not handle large displacements" | `verify_resample_geometry.py` | **yes** |
| 8 | The evaluation used a biased target-registration-error convention for this method but not for the baseline | The same transform scored 3.8748 mm instead of 2.9315 mm (**32% high**) on one case; the bias grows with displacement | `verify_tre_convention.py` | **yes** |
| 9 | `transformix -def` treats point coordinates as **voxel indices** (`physical = index x Spacing + Origin`) | Invisible at 1 mm (where index and physical value coincide); at 2 mm the points are scaled 2x and land outside the image, so the tool returns zero displacement and a case scored **414.6 mm** | `verify_tre_convention.py` | **yes** |
| 10 | `transformix` **rounds** the input indices to integers | Up to half a voxel per axis: **1 mm** at 2 mm spacing — the same order as the differences being measured | `verify_tre_convention.py` | **yes** |
| 11 | The reported baseline ran at **1 mm** while this method ran at 2 mm | Not the same protocol despite being presented as one. Correcting it *and* tuning the baseline moved its mean from 3.561 to 2.555 mm and the apparent advantage from -55% to -37.5% | `tools/summarize_elastix_baseline.py` | **yes** |
| 12 | A CREATIS landmark file that upstream does not ship was an **HTML 404 page** | `np.loadtxt` parsed it into garbage instead of failing. The loader now raises and lists the phases that do exist | `pmr_v2.load_lm_dataset` | **yes** |
| 13 | `LapIRN` pyramid: a coarse-level displacement was warped onto a finer-level image without being resampled onto that grid first | `RuntimeError: size of tensor a (48) must match tensor b (24)`. **All 10 leave-one-out folds died in ~120 s**, the driver recorded one `!! failed` line and **exited 0** — so the batch looked finished and a whole baseline had no numbers at all | resample onto the current level; `verify_lapirn_shapes.py` checks it in CI | **yes** |

## Why defect 7 is the instructive one

It was invisible for the entire lifetime of the project, because the first dataset's images
happen to have origin `(0, 0, 0)` — the bug was a no-op there. It appeared only when a
second dataset was added.

This is the general lesson: **every data-handling assumption that is accidentally satisfied
by your first dataset is untested.** Add a second dataset before you believe your pipeline.

The regression test asserts, for both datasets:

1. the output origin and direction equal the input's;
2. `resample_down(img, 2)` matches `img[::2, ::2, ::2]` voxel-for-voxel (< 1e-3 HU);
3. the down-sampled mask has the same volume fraction as the native mask;
4. the mean HU inside the mask is in the physiological lung range.

## What the suite does *not* verify

* **Not accuracy.** Accuracy is assessed against external landmarks, not by this suite.
* **Not diffeomorphism.** The suite *reports* folding; this implementation is not
  diffeomorphic (up to 0.084% of intra-pulmonary voxels have `det(J) <= 0`).
* **Not anatomical correctness.** No segmentation ground truth is used.
* **Basis-completeness results are about representation, not accuracy.** Exact closure does
  not imply a result closer to the truth.
