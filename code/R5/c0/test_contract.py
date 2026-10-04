import unittest
from types import SimpleNamespace
import numpy as np
from .common import comparisons
from .reset import clear_internal,apply_symmetric,MAIN_CLEAR,IK_CLEAR

class Rotation:
    @staticmethod
    def identity():return Rotation()
class Pose:
    def rotation(self):return Rotation()
    @classmethod
    def from_rotation_and_translation(cls,**kwargs):return cls()
class Tests(unittest.TestCase):
    def test_bitwise_gate_rejects_ulp_and_signed_zero(self):
        a={'x':np.array([0.,1.])}
        self.assertTrue(comparisons(a,a)[0]['bitwise_equal'])
        for b in (np.array([-0.,1.]),np.array([0.,np.nextafter(1.,2.)])):
            self.assertFalse(comparisons(a,{'x':b})[0]['bitwise_equal'])
    def test_nonfinite_rejected(self):
        self.assertFalse(comparisons({'x':np.array([np.nan])},{'x':np.array([np.nan])})[0]['bitwise_equal'])
    def test_intervention_before_official_callables(self):
        data=SimpleNamespace(**{key:np.ones(2) for key in MAIN_CLEAR})
        ik=SimpleNamespace(_model='IK_ONLY',_data='IK_DATA',**{key:np.ones(2) for key in IK_CLEAR})
        env=SimpleNamespace(_data=data,_ik=ik,_target_effector_pose=Pose(),_prev_qpos=np.ones(2),_prev_qvel=np.ones(2),_prev_ob_info={'qpos':np.ones(2)})
        events=[]
        def reset(seed):events.append('official_reset')
        def resetdata(model,state):self.assertEqual(model,'IK_ONLY');events.append('clear_ik')
        def callables(e,cfg,state):
            self.assertTrue(all(np.all(getattr(data,k)==0) for k in MAIN_CLEAR))
            self.assertTrue(all(np.all(getattr(ik,k)==0) for k in IK_CLEAR))
            self.assertEqual(state['qpos'].tolist(),[2.,3.]);events.append('set_state_and_goal')
        world=SimpleNamespace(reset=reset,envs=SimpleNamespace(envs=[SimpleNamespace(unwrapped=env)]),infos={'pixels':np.zeros((1,1,2,2,3),dtype=np.uint8)})
        apply_symmetric(world,SimpleNamespace(_apply_callables=callables),{'callables':[]},{'qpos':np.array([[2.,3.]]),'pixels':np.ones((1,2,2,3),dtype=np.uint8)}, {'goal':np.zeros((1,2,2,3),dtype=np.uint8)},SimpleNamespace(mj_resetData=resetdata))
        self.assertEqual(events,['official_reset','clear_ik','set_state_and_goal'])
        self.assertTrue(np.all(world.infos['pixels']==1))

if __name__=='__main__':unittest.main()
