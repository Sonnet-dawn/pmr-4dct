"""Run exactly one prespecified experiment and its same-field evaluation."""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('PMR_DATA_ROOT', str(ROOT.parent / 'reference_4' / 'data'))
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
sys.path.insert(0, str(HERE / 'frozen'))
import numpy as np
import torch
import SimpleITK as sitk
from study import evaluate, write_json, sha, image_path, reference_image, landmarks


def pmr(job, out):
    import pmr_extended as model
    from checkpoint_export import export_checkpoint
    checkpoint = out / 'registration_checkpoint.pt'
    flags = dict(final=[], control=['--res-reg-scale', '1'],
                 minmax=['--res-reg-scale', '1', '--norm', 'minmax'],
                 whole=['--res-reg-scale', '1', '--mask', 'none'],
                 nyquist=['--nyquist', '1'])
    sys.argv = ['pmr_extended.py', '--dataset', job['dataset'], '--case', str(job['case']),
                '--down', '2', '--mask', 'union', '--metric', 'local', '--res-metric', 'local',
                '--norm', 'robust', '--phases-per-step', '2', '--iters1', '800', '--iters2', '800',
                '--res-iters', '300', '--res-reg-scale', '10', '--cudnn-benchmark', '0',
                '--seed', str(job['seed']), '--save-dvf', '0', '--export-state', str(checkpoint),
                '--tag', 'run', '--out', str(out)]
    sys.argv += flags[job['arm']]
    write_json(out / 'command.json', sys.argv)
    resumed = checkpoint.is_file()
    if resumed:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['schema'] == 1 and state['argv'] == sys.argv, 'Checkpoint configuration mismatch'
        assert state['model_sha256'] == sha(HERE/'frozen'/'pmr_extended.py'), 'Checkpoint model mismatch'
        result = state['result']
        print('RESUME verified coefficient checkpoint; skip training', flush=True)
    else:
        result = model.main()
        gc.collect(); torch.cuda.empty_cache()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cn = f"case{job['case']}" if job['dataset'] == 'dirlab' else f"cr{job['case']}"
    path = out / f'run_{cn}_2mm_dvf.npy'
    gc.collect(); torch.cuda.empty_cache()
    export_checkpoint(state, path)
    del state
    result['resumed_from_checkpoint'] = resumed
    result['field_export_protocol'] = 'same_cuda_formula_one_phase_at_a_time_v1'
    assert path.is_file()
    gc.collect(); torch.cuda.empty_cache()
    return result, path


def elastix(job, out):
    from elastix_bspline import ElastixBSpline, validate_against_dvf
    import psutil
    exe = ROOT / 'tools' / 'elastix' / 'elastix.exe'
    transformix = exe.with_name('transformix.exe')
    fixed = reference_image(job['dataset'], job['case'])
    raw = sitk.ReadImage(str(image_path(job['dataset'], job['case'], 5)))
    moving = sitk.Resample(raw, fixed, sitk.Transform(), sitk.sitkLinear, 0.0)
    for name, image in [('fixed', fixed), ('moving', moving)]:
        sitk.WriteImage(image, str(out / f'{name}.mha'))
    text = '''(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(Registration "MultiResolutionRegistration")
(Interpolator "LinearInterpolator")
(ResampleInterpolator "FinalLinearInterpolator")
(Resampler "DefaultResampler")
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Optimizer "AdaptiveStochasticGradientDescent")
(MaximumNumberOfIterations 2000 2000 2000 2000 2000)
(NumberOfSpatialSamples 4096)
(NewSamplesEveryIteration "true")
(ImageSampler "RandomCoordinate")
(Metric "AdvancedMattesMutualInformation")
(NumberOfHistogramBins 32 32 32 32 32)
(Transform "BSplineTransform")
(NumberOfResolutions 5)
(FinalGridSpacingInPhysicalUnits 6 6 6)
(HowToCombineTransforms "Compose")
(DefaultPixelValue 0)
(WriteResultImage "false")
'''
    (out / 'parameters.txt').write_text(text, encoding='utf-8')
    cmd = [str(exe), '-f', str(out/'fixed.mha'), '-m', str(out/'moving.mha'),
           '-p', str(out/'parameters.txt'), '-out', str(out), '-threads', '4']
    write_json(out/'command.json', cmd)
    t0 = time.monotonic()
    with (out/'elastix_console.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        peak = 0
        while child.poll() is None:
            try:
                peak = max(peak, psutil.Process(child.pid).memory_info().rss)
            except psutil.Error:
                pass
            if psutil.virtual_memory().available < 5*2**30:
                child.terminate(); child.wait()
                raise RuntimeError('Stopped elastix: less than 5 GiB RAM available')
            if time.monotonic()-t0 > 3600:
                child.terminate(); child.wait()
                raise RuntimeError('Elastix exceeded prespecified one-hour job limit')
            time.sleep(.5)
        assert child.returncode == 0, 'Elastix failed; inspect log'
    runtime = time.monotonic()-t0
    write_json(out/'registration_resources.json', dict(registration_wall_s=runtime,
                                                      peak_rss_gib=peak/2**30))
    transform = out/'TransformParameters.0.txt'
    with (out/'transformix_console.log').open('w', encoding='utf-8') as log:
        subprocess.run([str(transformix), '-def', 'all', '-tp', str(transform), '-out', str(out),
                        '-threads', '4'], stdout=log, stderr=subprocess.STDOUT, check=True)
    candidates = [out/'deformationField.mhd', out/'deformationField.nii', out/'deformationField.nii.gz']
    dense = next(p for p in candidates if p.exists())
    bspline = ElastixBSpline(str(transform))
    validation = validate_against_dvf(bspline, str(dense), verbose=False)
    assert validation['max_mm'] < 1e-3, 'Continuous B-spline evaluator disagrees with transformix'
    field = np.moveaxis(sitk.GetArrayFromImage(sitk.ReadImage(str(dense))), -1, 0).astype(np.float32)
    path = out/'endpoint_dvf.npy'
    np.save(path, field)
    a, b, _ = landmarks(job['dataset'], job['case'])
    origin = np.array(fixed.GetOrigin())
    exact = np.linalg.norm(bspline.apply(a+origin) - (b+origin), axis=1)
    from recompute_tre_standard import sample_trilinear
    error = np.linalg.norm(a+sample_trilinear(field, a, fixed.GetSpacing())-b, axis=1)
    return dict(impl='elastix', tre_exact_mm=float(exact.mean()), tre_dense_mm=float(error.mean()),
                dense_vs_exact_mean_abs_error_mm=float(np.abs(error-exact).mean()),
                time_s=runtime, runtime_scope='elastix_registration_only_wall',
                peak_rss_gib=peak/2**30, transform_node_validation=validation,
                bins_effective=32, executable_sha256=sha(exe)), path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    job = next(j for j in manifest['jobs'] if j['id'] == args.job)
    out = HERE/'runs'/job['id']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        raise RuntimeError('Completed result already exists; do not overwrite')
    torch.set_num_threads(4)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)
    environment = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                       numpy=np.__version__, sitk=sitk.Version_VersionString(),
                       platform=platform.platform(), gpu=torch.cuda.get_device_name(0),
                       torch_threads=4, sitk_threads=4,
                       worker_sha256=sha(Path(__file__)),
                       model_sha256=sha(HERE/'frozen'/'pmr_extended.py'),
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    write_json(out/'environment.json', environment)
    start = time.monotonic()
    result, field = pmr(job, out) if job['kind'] == 'pmr' else elastix(job, out)
    diagnostics = evaluate(field, job, result, out)
    result['study_job'] = job
    result['total_job_wall_s'] = time.monotonic()-start
    result['diagnostics'] = 'diagnostics.json'
    write_json(out/'result.json', result)
    # Completion is committed only after registration AND independent field evaluation.
    write_json(out/'complete.json', dict(job=job, result_sha256=sha(out/'result.json'),
                                        diagnostics_sha256=sha(out/'diagnostics.json'),
                                        field_shape=list(np.load(field, mmap_mode='r').shape),
                                        tre_mm=diagnostics['tre_mean_mm']))
    # A committed, independently checked field replaces the temporary training checkpoint.
    (out/'registration_checkpoint.pt').unlink(missing_ok=True)
    print(f"COMPLETE {job['id']} TRE={diagnostics['tre_mean_mm']:.6f} mm", flush=True)


if __name__ == '__main__':
    main()
