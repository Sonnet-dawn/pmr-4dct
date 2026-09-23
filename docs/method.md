# Method

## The representation

A 4D-CT series is a set of images indexed by breathing phase. Write the displacement that
maps phase θ onto the reference phase as `d(x, θ)`. PMR constrains that field to lie on a
low-dimensional manifold:

```
d(x, θ) = Σ_{k=1..K} [ a_k(x) (cos kθ − 1) + b_k(x) sin kθ ]  +  d_aff(x, θ)
```

Two facts follow **by construction**, at every voxel, for every parameter setting:

```
d(x, 0)  = 0              the reference phase is the identity
d(x, 2π) = d(x, 0)        loop closure
```

Every basis function vanishes at `θ = 0` (because `cos 0 − 1 = 0` and `sin 0 = 0`), and
every one is `2π`-periodic. Neither property is a penalty term: there is no weight to tune
and no residual to monitor. `--K` sets the number of harmonics.

## How many basis functions are required?

This has a definite answer, and it is not the obvious one. For `N` anchored phase samples,
the design matrix has a zero first row in every column, so its range is contained in
`{v : v₀ = 0}`, a space of dimension `N − 1`. The representation is therefore exactly
lossless if and only if the basis has `N − 1` independent columns.

For `N = 10` phases:

| Basis | Columns | Rank | Condition | Max projection residual |
|---|---|---|---|---|
| `{cos kθ − 1, sin kθ}` for `k = 1..4` | 8 | 8 | 3.00 | **1.325 (lossy)** |
| the same **plus `cos 5θ − 1`** | **9** | **9** | **3.35** | **3.3e−15 (exact)** |
| `{cos kθ − 1, sin kθ}` for `k = 1..5` | 10 | 9 | 2.9e16 | numerically singular |

The missing mode of the 8-term basis is the **Nyquist mode**: the first singular direction
has inner product **1.000000** with the alternating vector `(−1)^i`. This is why the
implementation uses a 9-term basis internally even when `--K 4` is requested.

Run `verification/verify_projection_exact.py` to reproduce the table.

**What this does and does not mean.** The basis is lossless *as a representation of anchored
phase samples*. That is a statement about representation, not about accuracy: exact closure
does not imply a displacement field closer to the truth.

## Exact coarse-grid decomposition

The coefficients are stored on a coarse isotropic grid, and the phase weights are composed
**before** upsampling. With `U` the trilinear upsampling operator and `w_k(θ)` the
phase-only scalar weights,

```
U( Σ_k c_k · w_k(θ) )  ≡  Σ_k U(c_k) · w_k(θ)          (exact)
```

because trilinear interpolation is linear in its input and its weights depend only on the
output coordinate, not on the channel index `k`.

The implementation consequence is that 24 channels are reduced to 3 **on the coarse grid**,
and only a 3-channel field is upsampled. For a 1 mm DIR-Lab volume the coefficient tensor
is **2.55 MiB rather than 1.29 GiB** (**519×**), and the upsampling temporaries shrink 8×.

## Optimisation

1. **Manifold stage.** A CNN predicts the coarse coefficients from the reference image; the
   periodic field is composed, upsampled and used to warp sampled phases; the similarity
   loss is back-propagated. Runs coarsely then finely.
2. **Residual stage.** A second, finer periodic field refines the result. It has its **own**
   regularisation scale, exposed as `--res-reg-scale`; the global `--reg-scale` does not
   reach this stage. (That separation was itself a defect found by the verification suite —
   before it was exposed, folding could not be reduced by tuning the documented flag.)

Phases are sampled a few per optimiser step (`--phases-per-step`), which is what keeps the
image buffers small.

## Similarity and its support

Similarity is evaluated inside a lung mask derived from the images themselves. Whole-image
support biases the solution toward small displacement, because the stationary majority of
the field of view contributes to the loss while carrying no information about the motion.
On DIR-Lab this costs **+31.6%** in mean TRE across all ten cases.

Masking is standard practice and is *not* claimed as a contribution here. It is exposed as a
flag because the measured benefit depends on how per-window correlations are aggregated at
the mask boundary (`--local-mask-mode`), and that choice should be reportable rather than
buried.

## Memory accounting

For the 1 mm DIR-Lab case 1 (14,453,440 voxels, 24 coefficient channels at 4 bytes):

| Quantity | Naive per-voxel | PMR |
|---|---|---|
| Coefficient tensor | 1.29 GiB | **2.55 MiB** (519× smaller) |
| Optimiser state (Adam `m`, `v` + grad) | 3.88 GiB | negligible |
| **Subtotal** | **5.17 GiB** | ≈ 0 |
| Upsampling temporaries | 24 channels | 3 channels |

This implementation's measured peak on that same case is **3.25 GiB**, against a card with
**7.96 GiB**. A naive implementation would need ≈ **8.4 GiB** — it would not fit.

The naive subtotals are arithmetic, not measurements; a naive implementation was not
instrumented. See `LIMITATIONS.md`. All memory figures are GiB (2³⁰ bytes), matching what
`torch.cuda.max_memory_allocated()` reports.

## Determinism

`torch.backends.cudnn.benchmark` defaults to `True`, letting cuDNN choose convolution
algorithms at runtime. On the hardest DIR-Lab case, five runs of an *identical*
configuration gave TRE of 3.023, 3.216, 3.320, 3.354 and 3.584 mm: standard deviation
**0.20 mm (6.2% of the mean)**, range **0.56 mm (17.0% of the mean)**. (Relative to the
lowest of the five the range is 18.6%; we quote it against the mean so that it uses the
same basis as the standard deviation.) Passing `--cudnn-benchmark 0` reduces this to **0.2%**
(1.13048 vs 1.13305 mm on case 1). Effects below ~5% are not interpretable without this
control.

## Two datasets, two sets of conventions

| | DIR-Lab | CREATIS |
|---|---|---|
| Cases | 1–10 | 0–5 |
| Stored intensity | `HU + 1024` | raw HU |
| Image origin | `(0, 0, 0)` | non-zero |
| Landmarks | 300, T00/T50 only | 100–113; all phases for cases 0–2 only |
| Mean displacement T00→T50 | ~2–11 mm | 5.7–14.0 mm (max 32 mm) |

Assuming either convention globally fails **silently**: the images look normal and the
training loss simply never decreases. `verification/verify_resample_geometry.py` locks both
down. See `docs/verification.md` for how these were found.
