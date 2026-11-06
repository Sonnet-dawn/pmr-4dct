from pathlib import Path
import csv,json,hashlib,statistics
r=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for f in json.loads((r/'MANIFEST_SHA256.json').read_text(encoding='utf-8')):
 assert sha(r/f['path'])==f['sha256'],f['path']
for t in json.loads((r/'Source_to_Table.json').read_text(encoding='utf-8')):
 assert sha(r/t['source'])==t['sha256'],t['table']
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
a=r/'experiments/paperA_advisor_20261023/analysis'
rows=read(a/'core_three_seed_runs.csv')
assert len(rows)==90
assert len({(x['model'],int(x['case']),int(x['seed'])) for x in rows})==90
for model,expected in [('full',1.584989673153),('noaffine',1.676447508403),('noresidual',2.068771853811)]:
 rs=[x for x in rows if x['model']==model]
 assert len(rs)==30 and {int(x['case']) for x in rs}==set(range(1,11))
 mean=statistics.mean(statistics.mean(float(x['tre_mm']) for x in rs if int(x['case'])==c) for c in range(1,11))
 assert abs(mean-expected)<1e-9,(model,mean)
 print(model,format(mean,'.12f'),'mm')
print('PASS: published file hashes, 16 table sources, 90-fit design and three-seed means')
