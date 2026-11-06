"""Reaggregate means and bootstrap intervals from independently sampled errors."""
from pathlib import Path
import csv,json
import numpy as np
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'paperA_strengthening_20261021';COMP=HERE.parent/'paperA_components_20261022'
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def main():
    data=read(HERE/'analysis/independent_existing_tre_audit.csv')
    source={(r['study'],r['job']):r for r in data};checks=[]
    def value(study,case,seed,arm):return float(source[(study.name,f'dirlab_c{case:02d}_s{seed}_{arm}')]['tre_mm'])
    def check(label,computed,archived):
        tolerance=1e-5 if label.endswith('_pct') else 3e-7
        error=abs(float(computed)-float(archived));assert error<tolerance,(label,error)
        checks.append(dict(quantity=label,independent_value=float(computed),archived_value=float(archived),absolute_difference=error,tolerance=tolerance))
    rng=np.random.default_rng(20261021);original={r['contrast']:r for r in read(OLD/'analysis/paired_contrast_summary.csv')}
    for name,left,right in [('robust_minus_minmax','control','minmax'),('lung_minus_whole','control','whole'),('res10_minus_res1','final','control'),('nine_minus_eight','nyquist','final')]:
        v=np.array([np.mean([value(OLD,c,s,left)-value(OLD,c,s,right) for s in range(3)]) for c in range(1,11)])
        bs=v[rng.integers(0,10,(10000,10))].mean(axis=1);lo,hi=np.quantile(bs,[.025,.975]);o=original[name]
        for metric,x,key in [('mean',v.mean(),'mean_paired_difference_mm'),('CI_low',lo,'exploratory_case_bootstrap_low_mm'),('CI_high',hi,'exploratory_case_bootstrap_high_mm')]:check(name+' '+metric,x,o[key])
        assert int((v<0).sum())==int(o['cases_with_lower_seed_averaged_tre'])
    rng=np.random.default_rng(20261022);original={r['model']:r for r in read(COMP/'analysis/component_effects.csv')}
    for model in ['noaffine','noresidual','twoharmonics','coarseonly']:
        v=np.array([value(COMP,c,0,model)-value(OLD,c,0,'final') for c in range(1,11)])
        bs=v[rng.integers(0,10,(10000,10))].mean(axis=1);lo,hi=np.quantile(bs,[.025,.975]);o=original[model]
        for metric,x,key in [('mean',v.mean(),'mean_difference_mm'),('CI_low',lo,'exploratory_low_mm'),('CI_high',hi,'exploratory_high_mm')]:check(model+' '+metric,x,o[key])
    for r in read(COMP/'analysis/repeats.csv'):
        c=int(r['case']);v=np.array([value(OLD,c,s,'final') for s in range(3)])
        for k,x in [('mean_mm',v.mean()),('sd_mm',v.std(ddof=1)),('relative_range_pct',100*np.ptp(v)/v.mean())]:check(f'case{c} '+k,x,r[k])
    with (HERE/'analysis/independent_statistics_audit.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    report=dict(status='pass',quantities_checked=len(checks),maximum_absolute_difference=max(r['absolute_difference'] for r in checks),bootstrap_seed_correction='Existing protocol contrasts use20261021; structural primary and new core contrasts use20261022. Manuscript previously gave only20261022; correct Methods without changing results.')
    (HERE/'reports/statistics_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
