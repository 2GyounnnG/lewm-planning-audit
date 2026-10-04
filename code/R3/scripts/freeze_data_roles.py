"""Freeze actual official source identities, normalization, and metadata-only R3 roles."""
from pathlib import Path
import argparse, hashlib, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.preprocessing import StandardScaler
from r3 import common
from r3.data import RawH5,IMAGE_CONTRACT,admissible_windows,verified_file
from r3.roles import build_roles,canonical
ROOT=common.ROOT

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def freeze(path,data):
    p=ROOT/path
    if p.exists():
        if common.read_json(p)!=data:raise RuntimeError('Existing metadata freeze differs; retained unchanged: '+path)
    else:common.atomic_json(path,data)

def run(task,h5_path,family_column=None,initial_family=False,initial_family_columns=None):
    unpack=common.read_json(f'manifests/{task}_data_unpacked.json')
    matches=[x for x in unpack['files'] if x['path']==h5_path]
    if len(matches)!=1:raise ValueError('Selected HDF5 must be an actually unpacked and verified official member')
    asset=matches[0];path=verified_file(asset);source_sha=asset['sha256']
    source_assets=common.read_json(f'manifests/{task}_data_assets.json')
    if len([x for x in unpack['files'] if Path(x['path']).suffix in ('.h5','.hdf5')])!=1:
        raise RuntimeError('Multiple real HDF5 members require explicit author train/eval source audit; do not guess one pool')
    initial_family_columns=initial_family_columns or []
    if sum((bool(family_column),bool(initial_family),bool(initial_family_columns)))>1:raise ValueError('Only one explicit metadata family rule')
    prior=None;prior_selected={}
    if task=='pusht':
        prior_path=ROOT/'state/PUSHT_PRIOR_SOURCE_ALIGNMENT_V2.json'
        if prior_path.exists():
            prior=common.read_json(prior_path)
            if prior['status']!='EXACT_METADATA_ACTION_AND_INITIAL_INPUT_MATCH' or prior['R3_source']!=asset:raise RuntimeError('Prior source alignment does not bind actual source')
            prior_selected={x['source_episode_idx']:x for x in prior['old_selected_episodes']}
    map_doc={'task':task,'assets':{source_sha:asset},'official_dataset':source_assets['repo'],'official_revision':source_assets['revision'],'source_role':'author_model_card_original_train_source','independent_official_evaluation_source_verified':False,'selection_based_on_model_outputs':False}
    freeze(f'manifests/{task}_source_map.json',map_doc)
    reset_keys=['state'] if task=='pusht' else ['qpos','qvel']
    with RawH5(path,keys=['pixels','action']) as raw:
        available=list(raw.file.keys())
        missing=set(reset_keys)-set(available)
        if missing:raise RuntimeError('Actual source cannot reset task: missing '+str(sorted(missing)))
        if family_column and family_column not in available:raise RuntimeError('Specified actual family column absent')
        if any(k not in available for k in initial_family_columns):raise RuntimeError('Specified initial-condition column absent')
        actions=np.asarray(raw.file['action'][:]);finite=np.isfinite(actions).all(axis=1)
        scaler=StandardScaler().fit(actions[finite])
        valid=actions[finite].astype(np.float64);n=int(finite.sum())
        if n<2:raise RuntimeError('No finite original-source actions')
        normal={'task':task,'convention':'official_eval_StandardScaler_population_ddof0','image':IMAGE_CONTRACT,'action':{'mean':scaler.mean_.tolist(),'variance':scaler.var_.tolist(),'scale':scaler.scale_.tolist(),'finite_rows':n,'excluded_nonfinite_rows':int((~finite).sum()),'raw_dim':2,'macro_block':5,'sample_ddof1_std_for_disclosure':valid.std(axis=0,ddof=1).tolist()},'source_assets':{source_sha:asset},'population':'all finite raw actions in pinned original author training source, including the source episodes subsequently held out from R3 refit','R3_EVAL_physical_goals_used':False,'official_train_vs_eval_difference':'LeWM train utils uses torch sample std; selected evaluation pipeline uses sklearn population std. H0 and all refits share this one fixed eval-domain action transform.'}
        freeze(f'manifests/{task}_normalization.json',normal)
        records=[]
        for idx,length in enumerate(raw.lengths):
            length=int(length);offset=int(raw.offsets[idx]);aa=actions[offset:offset+length]
            reset={k:raw.array(idx,k) for k in reset_keys}
            for k,a in reset.items():
                if a.ndim!=2 or a.shape[0]!=length:raise RuntimeError('Actual reset column shape differs: '+k)
            valid_reset=np.logical_and.reduce([np.isfinite(a).all(axis=1) for a in reset.values()])
            planning=np.arange(max(0,length-25),dtype=np.int64)
            planning=planning[valid_reset[planning]&valid_reset[planning+25]]
            offline=admissible_windows(aa,length,horizon=5)+10
            source_seed=None;seed_note='SOURCE_SEED_COLUMN_ABSENT'
            if 'seed' in available:
                ss=raw.array(idx,'seed').reshape(length,-1)
                if ss.shape[1]!=1 or not np.isfinite(ss).all() or not np.all(ss==ss[0,0]) or ss[0,0]!=int(ss[0,0]) or not 0<=int(ss[0,0])<2**32:
                    raise RuntimeError('Source seed is not a constant uint32 per episode; requires explicit reset audit')
                source_seed=int(ss[0,0]);seed_note='ACTUAL_SOURCE_COLUMN_CONSTANT_WITHIN_EPISODE'
            family=None;evidence='FAMILY_UNKNOWN; no independence assertion'
            if family_column:
                values=raw.array(idx,family_column)
                if not np.all(values==values[0]):raise RuntimeError('Family column varies within an episode')
                value=values[0].tolist()
                if isinstance(value,bytes):value=value.decode('utf-8')
                family='source_family:'+digest(value);evidence='EXPLICIT_SOURCE_COLUMN:'+family_column
            elif initial_family:
                if not all(np.isfinite(a[0]).all() for a in reset.values()):raise RuntimeError('Nonfinite initial reset state cannot define exact initial-condition family')
                family='initial_reset_exact:'+digest({k:a[0].astype(np.float64).tolist() for k,a in reset.items()})
                evidence='CONSERVATIVE_EXACT_INITIAL_RESET_STATE_IDENTITY; does not prove different hashes statistically independent'
            elif initial_family_columns:
                initial={k:raw.array(idx,k,0,1)[0].astype(np.float64) for k in initial_family_columns}
                if not all(np.isfinite(a).all() for a in initial.values()):raise RuntimeError('Nonfinite initial-condition input')
                family='initial_input_exact:'+digest({k:a.tolist() for k,a in initial.items()})
                evidence='CONSERVATIVE_EXACT_INITIAL_INPUT_COLUMNS:'+','.join(initial_family_columns)+'; metadata-based repeated initial conditions, not proven original demo lineage'
            identity={'source_asset_sha256':source_sha,'source_episode_idx':idx,'raw_offset':offset,'length':length}
            eid=task+':'+source_sha[:16]+':'+str(idx)
            records.append({'episode_id':eid,'source_episode_idx':idx,'source_asset_sha256':source_sha,'episode_sha256':digest(identity),'episode_hash_definition':'SHA256 canonical(source asset SHA256, episode index, raw offset, length); asset SHA binds all original bytes; selected raw pixel/action content digest also saved in cache','length':length,'raw_offset':offset,'family_id':family,'family_evidence':evidence,'source_seed':source_seed,'source_seed_evidence':seed_note,'planning_starts':planning.tolist(),'open_loop_starts':offline.tolist(),'reset_metadata':{'column_shapes':{k:list(a.shape[1:]) for k,a in reset.items()}},'exposure':{'OFFICIAL_PRETRAIN_EXPOSURE':'SOURCE_EXPLICITLY_LINKED_AS_ORIGINAL_TRAIN; EXACT_EPISODE_EXPOSURE_UNVERIFIED','R3_REFIT_EXPOSURE':'PENDING_METADATA_ROLE_ASSIGNMENT','PRIOR_USER_STUDY_EXPOSURE':'UNVERIFIED_ACROSS_G1_R2_SOURCE_FORMATS'}})
        result=build_roles(task,records,history_size=3,frameskip=5)
        if prior is not None:
            result['prior_user_study_source_alignment']={'path':str(prior_path.relative_to(ROOT)),'sha256':common.sha256(prior_path),'exact_episode_values_actions_initial_proprio':True,'pixel_byte_identity_verified':False,'original_demonstration_lineage_verified':False}
            for row in result['episodes']:
                old=prior_selected.get(row['source_episode_idx'])
                row['exposure']['PRIOR_USER_STUDY_EXPOSURE']={'prior_G1_source_metadata_audited':True,'selected_in_G1_896':old is not None,'G1_role':old['G1_role'] if old else None,'G1_family_id':old['family_id'] if old else None,'R2_used_G1_source':True,'R2_per_episode_label_access_not_reconstructed':True,'pixel_byte_identity_verified':False}
        for row in result['episodes']:row['exposure']['R3_REFIT_EXPOSURE']=row['role']=='REFIT_TRAIN'
        for cases in result['cases'].values():
            for case in cases:
                idx=case['source_episode_idx'];start=case['start_raw_index'];goal=case['goal_raw_index']
                case['reset_metadata'].update({'start':{k:raw.array(idx,k,start,start+1)[0].tolist() for k in reset_keys},'goal':{k:raw.array(idx,k,goal,goal+1)[0].tolist() for k in reset_keys},'source_start_pixel_sha256':hashlib.sha256(raw.array(idx,'pixels',start,start+1).tobytes()).hexdigest(),'source_goal_pixel_sha256':hashlib.sha256(raw.array(idx,'pixels',goal,goal+1).tobytes()).hexdigest()})
        result['source_map_sha256']=common.sha256(ROOT/f'manifests/{task}_source_map.json')
        result['normalization_sha256']=common.sha256(ROOT/f'manifests/{task}_normalization.json')
        result['reader_source_sha256']=common.sha256(Path(__file__))
        freeze(f'manifests/{task}_data_roles.json',result)
        print(json.dumps({'status':result['status'],'task':task,'episodes':len(records),'roles':{k:len(v) for k,v in result['roles'].items()},'groups':{k:v for k,v in result['groups'].items() if k.endswith('count')},'formal_cases':len(result['cases']['EVAL']),'missing_reset_seed_cases':len(result['reset_validation_pending_case_ids'])}),flush=True)
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=common.TASKS);p.add_argument('--h5-path',required=True);p.add_argument('--family-column');p.add_argument('--initial-state-exact-family',action='store_true');p.add_argument('--initial-family-column',action='append');a=p.parse_args();run(a.task,a.h5_path,a.family_column,a.initial_state_exact_family,a.initial_family_column)
