# What the verification suite has caught

This file exists because "we tested it" is not evidence. Below is every real defect the
suite has caught in *this* codebase. Two of them were **silent**: the program ran to
completion, wrote a result, and the result was wrong.

Run the suite before trusting any number produced by this code:

```bash
python verification/verify_exact_basis.py
python verification/verify_projection_exact.py
python verification/verify_resample_geometry.py
python verification/jacobian_stats.py --selftest
python verification/probe_hu.py
python verification/check_mask_volume.py
python verification/verify_main_run.py <tag>
python verification/folding_check.py --dvf <path> --case N --down 2
```

## The seven defects

| # | Defect | Consequence | Caught by | Silent? |
|---|---|---|---|---|
| 1 | Stored intensity treated as HU while the files store `HU + 1024` | A threshold of `ct < -500` selected **0%** of voxels; a published fold-localisation claim was invalidated | `probe_hu.py` | **yes** |
| 2 | The residual stage's regularisation was hard-coded and unreachable from `--reg-scale` | Folding could not be reduced by tuning the documented flag; folding was 20-57x higher than assumed | `folding_check.py` | no |
| 3 | A coarse-scale result was evaluated **on the coarse grid** | Case 8 TRE was reported as 4.02 mm; on the correct grid it is 7.59 mm | `verify_main_run.py` | **yes** |
| 4 | `torch.quantile` has an input-size limit | A 1 mm run finished registration, then **crashed during evaluation and wrote no JSON** — the result was silently lost | missing result file | no |
| 5 | `torch.backends.cudnn.benchmark = True` (upstream default) | Identical configurations varied by **5.8% (SD)**, range 18.6%. Two published-effect conclusions were drawn and later reversed | `--cudnn-benchmark 0` | **yes** |
| 6 | Four concurrent large-case jobs | Exhausted an 8 GB card and **destabilised the host machine** | runtime watchdog added afterwards | no |
| 7 | Integer down-sampling **dropped the output origin** | On a dataset whose images do not start at `(0,0,0)`, the sampling window shifted by 250 mm; out-of-bounds fill (`0`) is *soft-tissue* HU, so the corruption was invisible. Training loss never decreased; a naive reading would have concluded "the method does not handle large displacements" | `verify_resample_geometry.py` | **yes** |

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
