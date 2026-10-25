"""Post-hoc terminal-failure reconstruction; never substitutes benchmark scores."""
from pathlib import Path
import argparse,hashlib,json,traceback
from obs_rct_benchmark.runner import load_config,_generate,method_seed,source_provenance,environment_provenance,external_method_provenance
from obs_rct_benchmark.methods import make_method
from obs_rct_benchmark.evaluation import cate_metrics
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args();root=args.root
out=root/'failure-diagnosis-v2.json';assert not out.exists()
checks=[];inputs=[]
for statuspath in sorted((root/'status-v1').glob('*.json')):
 status=json.loads(statuspath.read_text())
 if not status['coverage']['failed']: continue
 name=Path(status['config']).stem
 config=load_config(root/'configs-v1'/status['config'])
 records=root/'fits-v1'/name/'records.jsonl'
 manifestpath=records.with_name('run_manifest.json');manifest=json.loads(manifestpath.read_text())
 assert hashlib.sha256(records.read_bytes()).hexdigest()==status['records_sha256']
 assert manifest['config_hash']==config.config_hash
 assert manifest['source']==source_provenance()
 assert manifest['environment']==environment_provenance()
 assert manifest['external_methods']==external_method_provenance(config.methods)
 failed=[row for line in records.read_text().splitlines() if (row:=json.loads(line))['status']=='failed']
 assert len(failed)==status['coverage']['failed']
 inputs.append({'run_name':name,'config_hash':config.config_hash,'records_sha256':status['records_sha256'],'manifest_sha256':hashlib.sha256(manifestpath.read_bytes()).hexdigest(),'failed':len(failed)})
 scenarios={s.key:s for s in config.scenarios}
 for row in failed:
  repeats=[]
  for repeat in range(2):
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
  check={'run_name':name,'record_id':row['record_id'],'scenario':row['scenario'],'seed':row['seed'],'method':row['method'],'original_stage':row['stage'],'original_error_type':row['error_type'],'repeats':repeats,'failure_reproduced':all(r['status']=='failed' and r['stage']==row['stage'] and r['error_type']==row['error_type'] for r in repeats)}
  checks.append(check)
  print(json.dumps({k:check[k] for k in ('scenario','seed','method','failure_reproduced')}),flush=True)
value={'scope':'post-hoc diagnosis of ALL terminal failures in the frozen full catalog; original records preserved, no retry substitutes a score','inputs':inputs,'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checks':checks,'failed_cells_checked':len(checks),'fresh_refits':2*len(checks),'all_failure_stages_types_reproduced':all(c['failure_reproduced'] for c in checks)}
out.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
assert len(list((root/'status-v1').glob('*.json')))==128
assert value['all_failure_stages_types_reproduced']
