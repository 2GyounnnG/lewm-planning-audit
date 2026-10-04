"""Pure supplied-metadata R3 split contracts; no source observations or models."""
import copy
import hashlib
import unittest
from r3 import roles as r


def records(groups=10,per_group=20,unknown=False):
    return [{'episode_id':f'ep{i:04}', 'source_episode_idx':i,
             'source_asset_sha256':'a'*64,'episode_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
             'length':80,'family_id':None if unknown else f'f{i//per_group:03}',
             'source_seed':123,'planning_starts':[0,1,2,3,4,10,11,12,13,14],
             'open_loop_starts':[10,11,12,13,14,15,16,17,18,19,20],
             'reset_metadata':{'source_receipt':'synthetic_only'}}
            for i in range(groups*per_group)]


class RolesContracts(unittest.TestCase):
    def test_deterministic_input_order_and_metadata_immutability(self):
        source=records();before=copy.deepcopy(source)
        first=r.build_roles('pusht',source,history_size=3,frameskip=5)
        self.assertEqual(first,r.build_roles('pusht',list(reversed(source)),history_size=3,frameskip=5))
        self.assertEqual(source,before)
        self.assertEqual(first['status'],'METADATA_ROLES_FROZEN')

    def test_ceil_rounding_and_disjoint_split_families(self):
        result=r.build_roles('pusht',records(11,20),history_size=3,frameskip=5)
        groups=result['groups'];self.assertEqual(groups['eval_pool_count'],3);self.assertEqual(groups['monitor_count'],1)
        train,monitor,evaluation=[set(map(tuple,groups[k])) for k in ['REFIT_TRAIN','MONITOR','EVAL_POOL']]
        self.assertFalse(train&monitor or train&evaluation or monitor&evaluation)
        self.assertEqual(result['technical_eval_episode_overlap'],[])
        self.assertTrue(result['technical_eval_known_family_overlap'])

    def test_round_robin_first_four_and_fewer_than100(self):
        result=r.build_roles('pusht',records(10,20),history_size=3,frameskip=5)
        self.assertEqual(len(result['cases']['TECH']),4);self.assertEqual(len(result['cases']['EVAL']),36)
        self.assertEqual(result['sample_scope'],'LIMITED_EVALUATION_SAMPLE')
        families=[x['family_id'] for x in result['cases']['TECH']]
        self.assertEqual(families[0],families[2]);self.assertEqual(families[1],families[3]);self.assertNotEqual(families[0],families[1])
        self.assertEqual(result['planned_formal_trajectories'],144)

    def test_case_cap_unique_episode_and_unused_pool_retained(self):
        result=r.build_roles('reacher',records(10,80),history_size=3,frameskip=5)
        self.assertEqual(len(result['cases']['EVAL']),100);self.assertEqual(len(result['roles']['EVAL_POOL_UNUSED']),56)
        self.assertEqual(len({c['episode_id'] for a in result['cases'].values() for c in a}),104)

    def test_all_raw_phases_preserved_and_offline_indices(self):
        source=records()
        for i,row in enumerate(source):
            row['planning_starts']=[i%5];row['open_loop_starts']=[10+i%5]
        result=r.build_roles('pusht',source,history_size=3,frameskip=5)
        for row in result['episodes']:
            self.assertEqual(row['planning_starts'],source[row['source_episode_idx']]['planning_starts'])
        for case in result['cases']['TECH']+result['cases']['EVAL']:
            phase=case['source_episode_idx']%5
            self.assertEqual(case['start_raw_index'],phase)
            self.assertEqual(case['goal_raw_index'],phase+25)
            self.assertEqual(case['open_loop_anchor_raw'],10+phase)
            self.assertEqual(case['open_loop_window_start_raw'],phase)
            self.assertEqual(case['open_loop_target_raw']['5'],35+phase)
            self.assertEqual(case['open_loop_anchor_relation'],'OPEN_LOOP_ANCHOR_NOT_PLANNING_START')

    def test_same_legal_anchor_reused(self):
        source=records()
        for row in source:row['planning_starts']=[12]
        result=r.build_roles('pusht',source,history_size=3,frameskip=5)
        self.assertTrue(all(c['open_loop_anchor_raw']==12 and c['open_loop_anchor_relation']=='SAME_AS_PLANNING_START' for c in result['cases']['EVAL']))

    def test_missing_seed_stays_missing_and_unknown_family_not_invented(self):
        source=records(10,20,unknown=True)
        for row in source:row.pop('source_seed')
        result=r.build_roles('reacher',source,history_size=3,frameskip=5)
        self.assertEqual(result['groups']['all_count'],200)
        self.assertEqual(len(result['reset_validation_pending_case_ids']),40)
        self.assertTrue(all(c['reset_seed'] is None and c['family_id'] is None for a in result['cases'].values() for c in a))

    def test_insufficient_data_is_explicit_block_not_repartition(self):
        result=r.build_roles('pusht',records(3,2),history_size=3,frameskip=5)
        self.assertEqual(result['status'],'BLOCKED_DATA')
        self.assertIn('FEWER_THAN_20_LEGAL_EVAL_EPISODES_AFTER_RESERVED_TECH',result['blocked_reasons'])
        tiny=r.build_roles('pusht',records(2,30),history_size=3,frameskip=5)
        self.assertIn('EMPTY_REFIT_TRAIN_AFTER_FIXED_GROUP_SPLIT',tiny['blocked_reasons'])

    def test_reject_bad_admissibility_scores_and_duplicate_source(self):
        for field,value in [('planning_starts',[55]),('open_loop_starts',[9]),('open_loop_starts',[51]),
                            ('planning_starts',[0,0]),('source_seed',-1),('success',1)]:
            source=records();source[0][field]=value
            with self.subTest(field=field,value=value):
                with self.assertRaises(ValueError):r.build_roles('pusht',source,history_size=3,frameskip=5)
        source=records();source[1]['source_episode_idx']=0
        with self.assertRaisesRegex(ValueError,'Duplicate source'):r.build_roles('pusht',source,history_size=3,frameskip=5)

    def test_empty_admissibility_retained_with_exclusion_and_no_model_selection(self):
        source=records();source[0]['planning_starts']=[]
        result=r.build_roles('pusht',source,history_size=3,frameskip=5)
        row=next(x for x in result['episodes'] if x['episode_id']=='ep0000')
        self.assertEqual(row['role'],'EXCLUDED_METADATA_INELIGIBLE')
        self.assertFalse(result['selection_used_model_outputs'])

    def test_monitor_windows_are_authorized_history_starts(self):
        result=r.build_roles('pusht',records(),history_size=3,frameskip=5)
        by={x['episode_id']:x for x in result['episodes']}
        for key,role in [('monitor_windows','MONITOR'),('technical_monitor_windows','TECH')]:
            self.assertLessEqual(len(result[key]),256)
            self.assertTrue(result[key])
            for ep,start in result[key]:
                self.assertEqual(by[ep]['role'],role);self.assertIn(start+10,by[ep]['open_loop_starts'])
                self.assertLessEqual(start+40,by[ep]['length'])


if __name__=='__main__':unittest.main(verbosity=2)
