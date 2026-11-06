# Paper A v07: anchored periodic phase-manifold registration

Source and derived numerical results for **Anchored periodic phase-manifold registration for respiratory 4D-CT**. Snapshot prepared 6 November 2026 for the BMC Medical Imaging manuscript; this is not a claim of publication or acceptance.

## Start here

- `Source_to_Table.json` maps all nine main and seven supplementary tables to unrounded CSV files and their original SHA-256 hashes.
- `experiments/paperA_advisor_20261023/analysis/core_three_seed_runs.csv`: 90 fits, with seeds, endpoint mean/p95 TRE, ROI folding, parameter counts, update counts, runtime and peak GPU allocation.
- `core_three_seed_cases.csv` and `core_three_seed_effects.csv` in the same directory: within-patient seed averaging and paired patient-level effects.
- `experiments/paperA_components_20261022/analysis/component_case_results.csv`: seed-0 structural ablations, including two-harmonic and coarse-only variants.
- `experiments/paperA_strengthening_20261021/analysis/completed_runs.csv`: protocol contrasts, the six-case CREATIS cohort, and comparator records.
- `verify_public_results.py` verifies package checksums, table provenance, the 90-fit design, and recorded means without CT data or a GPU.

## Numerical scope

Primary seed-0 DIR-Lab TRE is 1.585525596093 mm; CREATIS TRE is approximately 1.128 mm. Three-seed cohort means are approximately Full 1.585, NoAffine 1.676 and NoResidual 2.069 mm. The ten patients are the resampling units after within-patient averaging of three seeds. Full/NoAffine have 1600 encoder and 300 residual updates; NoResidual has 1600 encoder updates. CoarseOnly reuses the NoAffine pre-residual state and has no independent pipeline-time/peak-memory measurement. Missing values remain missing. The ROI folding fraction uses the dilated union of the unwarped T00/T50 masks, not the reference mask alone.

Root-level documentation and the historical Zenodo DOI describe earlier releases. This folder is the manuscript-specific snapshot. Analytic periodic endpoint anchoring does not establish zero error for compositions of adjacent transforms. The historical defect summary and its individually classified silent defects do not coincide; the supplied ledger retains the per-defect classification.

## Using the code

Use the preserved **versioned experiment implementations**, not the older root `src` model, for this manuscript. The directory hierarchy inside this folder mirrors the research project. Set `PMR_DATA_ROOT` to an authorised local data directory containing `DIRLAB` and `CREATIS`.

Recorded environment: Python 3.9.25, PyTorch 2.8/CUDA 12.8, NumPy, SciPy and SimpleITK. See `requirements.txt`. Install the appropriate CUDA build of PyTorch for your driver. Training is subject-specific, and expert landmarks are used for evaluation only.

```sh
cd paperA_v07
python verify_public_results.py
python prepare_runtime_dirs.py
# Set PMR_DATA_ROOT before a data-dependent job:
python experiments/paperA_components_20261022/run_component.py --job <id-from-manifest.json>
```

Read the manifest and run_queue.py before launching a queue. Run one GPU job at a time. The training scripts retain their original Windows-oriented resource guards and path/hash conventions; cross-platform execution may need path adaptation. Do not change frozen files silently: their hashes are deliberately checked. New runtime directories are empty; original dense fields/checkpoints and raw run folders are not in this public package. Full source-linked audits need those retained assets; the lightweight verifier works directly from the published aggregate records. This release was not retrained on another machine.

## Data access and publication boundary

Original DIR-Lab CT images/landmarks: https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html

CREATIS/POPI images/landmarks and attribution conditions: https://github.com/open-vv/popi-model/blob/master/dir_validation_data.md

No raw CT, original landmark coordinates, dense displacement fields, checkpoints, patient identifiers, downloaded papers, institutional fonts or private correspondence are redistributed here. Case IDs are benchmark indices. The CSV files contain derived evaluation measurements. Re-evaluating full fields or rerunning registration requires data obtained from the providers and, for retrospective field audits, the corresponding saved arrays. Cite the original data papers and acknowledge the Leon Berard Cancer Center/CREATIS as required. Repository source code and the author-produced derived tables are distributed under the repository's existing MIT license; third-party source and dataset terms remain applicable.

Code and text preparation involved OpenAI Codex assistance, as disclosed in the manuscript. Authors remain responsible for the work. This folder and the table mapping supersede old README performance figures for Paper A v07.
