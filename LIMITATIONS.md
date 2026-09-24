# Limitations

1. **No accuracy-SOTA claim.** On DIR-Lab (10 cases, standard protocol) this
   implementation reaches 1.597 mm mean TRE. Published results reach 1.0-1.3 mm.
2. **Not diffeomorphic.** In the reported configuration (`--res-reg-scale 10`), the mean over
   the ten phases of the intra-pulmonary non-positive `det(J)` fraction is
   **0.0006% (case 1) to 0.084% (case 8)**, with the worst voxel reaching
   `det(J) = -1.79` on the hardest case. Both figures are per case and per the ten-phase
   mean, from `results/folding_v2_resreg10_case*.json`; the older, under-regularised
   configuration folded up to **1.7%** on case 8, which is why the residual-stage scale is
   documented rather than left at its default.
3. **1 mm is only feasible for smaller volumes.** The large DIR-Lab cases (6-10) have
   ~79 million voxels per phase at 1 mm, requiring roughly 20 GiB; this does not fit on
   an 8 GiB card. The 1 mm results cover cases 1-5 only. The CREATIS volumes are larger
   still (up to 116 million voxels per phase at 1 mm) and are reported at 2 mm.
4. **The coarse-grid formulation is not itself novel** — control-grid upsampling is
   standard (B-spline transforms). What is contributed here is an exact algebraic
   statement of the decomposition, a minimal-memory implementation, and the verification
   suite.
5. **Lung masking is standard practice**, not a contribution of this work.
6. **The naive memory subtotals are arithmetic, not measurements.** We did not instrument a
   naive per-voxel implementation. The coefficient tensor a naive implementation *would*
   need (1.29 GiB) and its optimiser state (3.88 GiB) are tensor-size arithmetic. What is
   measured is this implementation's own peak on the same case (3.25 GiB) and the card's
   capacity (7.96 GiB); the conclusion that a naive version would not fit follows from
   those measurements plus the arithmetic.
7. **Run-to-run variability** is 6.2% (SD, of the mean) unless `--cudnn-benchmark 0` is used.
   Passing the flag removes most of it but, importantly, not uniformly: two complete runs of
   an identical configuration with the flag off differed by 0.5-1.6% on the nine easier
   DIR-Lab cases but by **5.3%** and **5.4%** on the two hardest, while the ten-case mean
   moved by only 0.47%. The flag stabilises an aggregate number; it does not make the hardest
   single cases reproducible, and single-case differences of a few percent there should not be
   interpreted.
8. **Landmark phase coverage is uneven.** On CREATIS, only cases 0-2 carry expert
   landmarks at all ten phases; cases 3-5 carry them only at end-exhale (T00) and
   end-inhale (T50). Requiring a phase that does not exist raises an error rather than
   returning a silently wrong number — see `creatis_valid_phases()`.
9. **Landmark annotation is itself uncertain.** The original CREATIS protocol reports a
   mean inter-observer distance of 0.5 mm (SD 0.93 mm). Results near 1.1 mm are therefore
   within roughly a factor of two of the ground truth's own precision, and differences
   below that scale should not be over-interpreted.
10. **Two dataset-specific conventions must not be conflated.** DIR-Lab stores
    `HU + 1024`; CREATIS stores raw HU. DIR-Lab images have origin `(0, 0, 0)`; CREATIS
    images do not. Assuming either convention globally is silent corruption, not an error:
    it produces plausible images and a training loss that simply never decreases.
