#!/usr/bin/env python3
"""Run retained experiments with configurable paths."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from satc import unpack_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['quality','throughput','network','rtt'])
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',action='store_true',help='Quality: inspect inputs and planned run order')
    args,extra=p.parse_known_args()
    config=json.loads(args.config.read_text())
    required=['prepared_dir','london_video','sdk']
    for key in required:
        if key not in config: p.error(f'Config requires {key}')
    env=os.environ.copy()
    for key,value in {'SATC_PREPARED':config['prepared_dir'],'SATC_LONDON':config['london_video'],
                      'SATC_SDK':config['sdk'],'SATC_MODEL':str(unpack_model()),
                      'SATC_PYTHON':config.get('python',sys.executable),
                      'SATC_METRICS_PYTHON':config.get('metrics_python',sys.executable)}.items():
        env[key]=str(Path(value).expanduser().resolve())
    python=env['SATC_PYTHON']
    research=ROOT/'experiments/research'
    if args.experiment=='quality':
        command=[python,str(research/'final_quality_matrix.py'),'--plan' if args.plan else '--run',
                 '--master',str(research/'all_experiments_v6.py'),'--base-script',str(research/'allocation_candidates.py'),
                 '--london-video',env['SATC_LONDON'],'--no-reuse-london','--output',str(args.output.resolve())]
    else:
        if args.plan: p.error('--plan is supported for quality only')
        command=[python,str(research/'all_experiments_v6.py'),'--experiment',args.experiment,
                 '--london-video',env['SATC_LONDON'],'--output',str(args.output.resolve())]
    if config.get('shared_mat'):command+=['--shared-mat',str(Path(config['shared_mat']).expanduser().resolve())]
    return subprocess.call(command+extra,env=env)
if __name__=='__main__':raise SystemExit(main())
