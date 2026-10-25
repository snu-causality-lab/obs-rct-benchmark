from pathlib import Path
import hashlib,json,traceback
from obs_rct_benchmark.runner import load_config,_generate,method_seed
from obs_rct_benchmark.methods import make_method
from obs_rct_benchmark.evaluation import cate_metrics
root=Path('/shared/s2/lab05/sanghack/work/obsrct-public-catalog-20261025-r01')
name='group-common-seed-20014'
config=load_config(root/'configs-v1'/f'{name}.toml')
records=root/'fits-v1'/name/'records.jsonl'
failed=[json.loads(line) for line in records.read_text().splitlines() if json.loads(line)['status']=='failed']
scenarios={s.key:s for s in config.scenarios};checks=[]
for row in failed:
 repeats=[]
 for k in range(2):
  stage='simulation'
  try:
   problem=_generate(scenarios[row['scenario']],row['seed'])
   stage='method_or_metric'
   estimator=make_method(row['method'],**dict(config.method_options.get(row['method'],{})))
   pred=estimator.fit_predict(problem.training_view(),problem.eval_X,seed=method_seed(row['scenario'],row['seed'],row['method']))
   result={'status':'ok','metrics':cate_metrics(pred,problem.eval_tau)}
  except Exception as exc:
   result={'status':'failed','stage':stage,'error_type':type(exc).__name__,'message':str(exc),'frames':[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno} for f in traceback.extract_tb(exc.__traceback__)]}
  repeats.append(result)
 checks.append({'record_id':row['record_id'],'scenario':row['scenario'],'seed':row['seed'],'method':row['method'],'original_stage':row['stage'],'original_error_type':row['error_type'],'repeats':repeats,'failure_reproduced':all(r['status']=='failed' and r['stage']==row['stage'] and r['error_type']==row['error_type'] for r in repeats)})
value={'scope':'post-hoc failure diagnosis of seed20014; original records preserved, no retry replaces a failed score','config_hash':config.config_hash,'records_sha256':hashlib.sha256(records.read_bytes()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':checks}
(root/'failure-diagnosis-v1.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
print(json.dumps(value,indent=2))
assert all(c['failure_reproduced'] for c in checks)
