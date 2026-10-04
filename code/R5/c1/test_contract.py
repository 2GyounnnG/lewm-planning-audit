import unittest,types
from .common import install_reset
class TestContract(unittest.TestCase):
    def test_reset_order_and_seed(self):
        calls=[];env=object();world=types.SimpleNamespace(envs=types.SimpleNamespace(envs=[types.SimpleNamespace(unwrapped=env)]))
        def original(seed=None,options=None):calls.append(('official_reset',seed));return 'same'
        world.reset=original
        install_reset(world,None,lambda e,m:calls.append(('clear',e)))
        self.assertEqual(world.reset(),'same');calls.append(('set_state',env))
        self.assertEqual([x[0] for x in calls],['official_reset','clear','set_state'])
        with self.assertRaises(ValueError):world.reset(seed=0)
    def test_three_mean_is_not_ensemble(self):
        import numpy as np
        outcomes=np.array([[1,0,0],[0,1,1]],float)
        self.assertTrue(np.array_equal(outcomes.mean(1),[1/3,2/3]))
    def test_case_stream_inherits(self):
        import hashlib
        cid='R3_cube_abc';i=1
        expected=int.from_bytes(hashlib.sha256(f'R3_CEM_CASE_20261002/cube/{cid}/{i}'.encode()).digest()[:8],'big')
        from x1.evaluate import case_seed
        self.assertEqual(expected,case_seed('cube',cid,i))
if __name__=='__main__':unittest.main()
