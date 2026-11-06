from pathlib import Path
r=Path(__file__).resolve().parent
for name in ['paperA_strengthening_20261021','paperA_components_20261022','paperA_advisor_20261023']:
 for folder in ['runs','logs','analysis','reports']:(r/'experiments'/name/folder).mkdir(parents=True,exist_ok=True)
print('Runtime directories ready; no training was launched.')
