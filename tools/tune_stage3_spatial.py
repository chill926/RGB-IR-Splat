"""RoadBlock C spatial-prior sweep, then matched R/K at the selected strength."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from utils.thermal_stage3 import file_sha256, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, default=ROOT/'output/RoadBlock_stage3_D_pilot_run1_common/thermal_stage3_common.pt')
    parser.add_argument("--hyperparameter_search", type=Path, default=ROOT/'output/RoadBlock_stage3_tune_run1')
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--evaluate_selected", action="store_true",
                        help="After validation selection and matched R/K training, evaluate selected C/R/K all splits")
    args = parser.parse_args()
    if min(args.steps, args.eval_every) <= 0:
        parser.error("Positive budget/interval required")
    args.output, args.common = args.output.resolve(), args.common.resolve()
    checkpoint = torch.load(args.common, map_location='cpu')
    verify_inputs(checkpoint)
    metadata = checkpoint['metadata']
    for protected in (args.common.parent, Path(metadata['source_path']).resolve(),
                      Path(metadata['geometry_model']).resolve(),
                      Path(metadata['stage3_common']['source_checkpoint']).resolve().parent,
                      args.hyperparameter_search.resolve()):
        if args.output == protected or args.output in protected.parents or protected in args.output.parents:
            raise ValueError('Output overlaps protected experiments/inputs')
    rows = json.loads((args.hyperparameter_search/'trial_results.json').read_text())
    selection = json.loads((args.hyperparameter_search/'selection.json').read_text())
    by_id = {row['id']: row for row in rows}
    chosen = {branch: by_id[identifier] for branch,identifier in selection['selected_trial_ids'].items()}
    if any(row['summary']['protocol']['common_sha256'] != file_sha256(args.common) for row in chosen.values()):
        raise ValueError('Previous hyperparameter search used a different common')
    if chosen['R']['regularization'] != chosen['K']['regularization']:
        raise ValueError('R/K need one matched material regularization value')
    shared = chosen['C']['summary']['protocol']
    settings = {'temperature_lr':chosen['C']['temperature_lr'],
                'temperature_lr_final':shared['shared_training_protocol']['temperature_lr_final'],
                'environment_lr':chosen['C']['environment_lr'],
                'environment_lr_final':shared['shared_training_protocol']['environment_lr_final'],
                'material_lr':shared['material_lr'], 'material_lr_final':shared['material_lr_final'],
                'lr_hold_fraction':shared['shared_training_protocol']['lr_hold_fraction'],
                'lambda_k':chosen['K']['regularization'], 'lambda_delta_epsilon':chosen['R']['regularization']}
    grid = [0.,1e-4,1e-3,1e-2]
    plan = {'kind':'RoadBlock_spatial_C_sweep_matched_R_K_v1', 'common':str(args.common),
            'common_sha256':file_sha256(args.common), 'steps':args.steps,'eval_every':args.eval_every,
            'seed':args.seed, 'settings':settings, 'lambda_spatial_grid':grid,
            'previous_search_plan_sha256':file_sha256(args.hyperparameter_search/'search_plan.json'),
            'selection_uses_validation_only':True,'evaluate_selected_after_training':args.evaluate_selected,
            'signal_handoff':'float_checkpoint_to_signal_render_without_pseudo_color_inverse',
            'runtime_source_sha256':{str(path.relative_to(ROOT)):file_sha256(path) for path in
                [Path(__file__).resolve(),ROOT/'train_thermal_stage3.py',ROOT/'utils/thermal_spatial.py',
                 ROOT/'utils/thermal_stage3.py',ROOT/'utils/thermal_joint.py',ROOT/'utils/thermal_physics.py',
                 ROOT/'utils/thermal_observations.py',ROOT/'utils/thermal_training_utils.py',
                 ROOT/'scene/dataset_readers.py',ROOT/'scene/__init__.py',ROOT/'gaussian_renderer/__init__.py']}}
    if args.dry_run:
        print(json.dumps(plan,indent=2)); return
    if args.output.exists():
        if not args.resume or not (args.output/'search_plan.json').is_file():
            raise ValueError('Output exists; use a new name or --resume matching search')
        if json.loads((args.output/'search_plan.json').read_text()) != plan:
            raise ValueError('Resume plan/settings/source mismatch')
    else:
        args.output.mkdir(parents=True)
        (args.output/'search_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (args.output/'search.lock').open('a') as guard:
        try: fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This search is already running')
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        trials=[]
        def execute(identifier,branch,strength,lock):
            directory=args.output/identifier
            command=[sys.executable,'-B','-u',str(ROOT/'train_thermal_stage3.py'),'--common',str(args.common),
                     '--branch',branch,'--comparison_protocol',str(lock),'-m',str(directory),
                     '--steps',str(args.steps),'--eval_every',str(args.eval_every),'--save_every','1000',
                     '--seed',str(args.seed),'--lambda_spatial',str(strength)]
            for name,value in settings.items():command.extend(['--'+name,str(value)])
            summary_path=directory/'training_summary.json'
            if not summary_path.is_file():
                # Preserve incomplete trials/logs and restart the same trial
                # from the common state; this is not optimizer-state resume.
                stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                if directory.exists():directory.rename(args.output/(identifier+'.interrupted_'+stamp))
                log=args.output/(identifier+'.log')
                if log.exists():log.rename(args.output/(identifier+'.interrupted_'+stamp+'.log'))
                print('[spatial start] '+json.dumps({'id':identifier,'branch':branch,'lambda_spatial':strength}),flush=True)
                with log.open('x') as handle:
                    subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                   stdout=handle,stderr=subprocess.STDOUT,check=True)
            summary=json.loads(summary_path.read_text())
            protocol=summary['protocol']
            shared_values=protocol['shared_training_protocol']
            if (summary['stopped_step']!=args.steps or summary['branch']!=branch
                    or not summary['frozen_state_verified'] or protocol['common_sha256']!=plan['common_sha256']
                    or protocol['lambda_spatial']!=strength or protocol['seed']!=args.seed
                    or any((shared_values.get(key) if key in ['temperature_lr','temperature_lr_final',
                        'environment_lr','environment_lr_final','lr_hold_fraction'] else protocol.get(key))!=value
                        for key,value in settings.items())):
                raise ValueError('Trial protocol mismatch: '+identifier)
            row={'id':identifier,'branch':branch,'lambda_spatial':strength,'summary':summary}
            trials.append(row)
            (args.output/'trial_results.json').write_text(json.dumps(trials,indent=2,allow_nan=False)+'\n')
            print('[spatial done] '+json.dumps({'id':identifier,'best_step':summary['best_step'],
                'best_validation':summary['best_validation_objective'],
                'final_validation':summary['final_validation_metrics']['selection_loss']}),flush=True)
            return row
        c_rows=[]
        for index,strength in enumerate(grid):
            identifier='C_spatial_{:02d}'.format(index)
            c_rows.append(execute(identifier,'C',strength,args.output/'locks'/(identifier+'.json')))
        selected=min(c_rows,key=lambda row:(row['summary']['best_validation_objective'],
                         row['summary']['final_validation_metrics']['selection_loss'],row['id']))
        lock=args.output/'locks'/(selected['id']+'.json')
        strength=selected['lambda_spatial']
        selected_rows={'C':selected}
        for branch in ['R','K']:
            selected_rows[branch]=execute(branch+'_selected_spatial',branch,strength,lock)
        result={'selected_lambda_spatial':strength,'selected_trial_ids':{b:r['id'] for b,r in selected_rows.items()},
                'best_steps':{b:r['summary']['best_step'] for b,r in selected_rows.items()},
                'shared_settings':settings,'same_hyperparameters_C_R_K':True,
                'selection_uses_validation_only':True,'C_all_best_steps_zero':all(r['summary']['best_step']==0 for r in c_rows)}
        (args.output/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
        exports={'GS_WORK':str(ROOT),'THERMAL_PYTHON':sys.executable,'STAGE3_COMMON':str(args.common),
                 'STAGE3_D_CHECKPOINT':metadata['stage3_common']['source_checkpoint'],
                 'STAGE3_LEARN_MATERIALS':','.join(metadata['stage3_common']['enabled_materials']),
                 'STAGE3_RUN':args.output.name+'_full','STAGE3_STEPS':'10000','STAGE3_EVAL_EVERY':'100',
                 'STAGE3_SAVE_EVERY':'1000','STAGE3_SEED':str(args.seed),'STAGE3_SPATIAL':str(strength)}
        names={'temperature_lr':'STAGE3_T_LR','temperature_lr_final':'STAGE3_T_LR_FINAL',
               'environment_lr':'STAGE3_E_LR','environment_lr_final':'STAGE3_E_LR_FINAL',
               'material_lr':'STAGE3_MATERIAL_LR','material_lr_final':'STAGE3_MATERIAL_LR_FINAL',
               'lr_hold_fraction':'STAGE3_HOLD','lambda_k':'STAGE3_LAMBDA_K','lambda_delta_epsilon':'STAGE3_LAMBDA_DELTA'}
        exports.update({names[key]:str(value) for key,value in settings.items()})
        (args.output/'best_settings.env').write_text(''.join('export '+key+'='+shlex.quote(value)+'\n' for key,value in exports.items()))
        if args.evaluate_selected:
            for branch,row in selected_rows.items():
                directory=args.output/row['id']
                checkpoint_path=directory/('thermal_'+branch+'_best.pt')
                for modality in ['thermal','rgb']:
                    command=[sys.executable,'-B',str(ROOT/'tools/evaluate_thermal_physics.py'),
                             '--checkpoint',str(checkpoint_path),'--modality',modality,'--split','all']
                    if modality=='thermal':command.extend(['--thermal_output','pseudocolor'])
                    with (args.output/(branch+'_evaluation_'+modality+'.log')).open('a') as handle:
                        subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                       stdout=handle,stderr=subprocess.STDOUT,check=True)
        print('[spatial complete] '+json.dumps(result),flush=True)


if __name__=='__main__':
    main()
