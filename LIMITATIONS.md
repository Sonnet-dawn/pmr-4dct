# Limitations

1. **No accuracy-SOTA claim.** On DIR-Lab (10 cases, standard protocol) this
   implementation reaches 1.597 mm mean TRE. Published results reach 1.0-1.3 mm.
2. **Not diffeomorphic.** 0.009%-0.003% of voxels have `det(J) <= 0` depending on the
   case (minimum `det(J) = -1.79` on the hardest case).
3. **1 mm is only feasible for smaller volumes.** The large DIR-Lab cases (6-10) have
   ~79 million voxels per phase at 1 mm, requiring roughly 20 GB; this does not fit on
   an 8 GB card. The 1 mm results cover cases 1-5 only. The CREATIS volumes are larger
   still (up to 116 million voxels per phase at 1 mm) and are reported at 2 mm.
4. **The coarse-grid formulation is not itself novel** — control-grid upsampling is
   standard (B-spline transforms). What is contributed here is an exact algebraic
   statement of the decomposition, a minimal-memory implementation, and the verification
   suite.
5. **Lung masking is standard practice**, not a contribution of this work.
6. **Derived numbers are labelled as such.** The memory a naive per-voxel
   parameterisation *would* require is computed arithmetically; it was not measured.
7. **Run-to-run variability** is 5.8% (SD) unless `--cudnn-benchmark 0` is used.
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
