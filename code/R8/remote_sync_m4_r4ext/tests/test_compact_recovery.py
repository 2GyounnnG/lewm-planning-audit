import json,tempfile,unittest
from pathlib import Path
import numpy as np
from r4.common import atomic_json,atomic_npz,file_record
from r4.compact_recovery import compact_case,frame_hash

class Tests(unittest.TestCase):
    def test_lossless_nonpixels_and_exact_frame_hashes_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'original';dst=root/'derived';src.mkdir()
            pixels=np.arange(3*2*2*3,dtype=np.uint8).reshape(3,2,2,3)
            arrays={'raw_pixels':pixels,'raw_actions':np.array([[.1,.2],[.3,.4]],np.float32),'raw_latent':np.arange(12,dtype=np.float32).reshape(3,4),'goal_pixels':pixels[-1],'raw_dynamical_state':np.ones((3,7)),'returned_plans_normalized':np.ones((1,5,10),np.float32)}
            atomic_npz(src/'trajectory.npz',**arrays);atomic_json(src/'result.json',{'task':'pusht','case_id':'tiny'})
            atomic_json(src/'attempt_000/STARTED.json',{'identity_sha256':'test','identity':{'version':'test'}})
            atomic_json(src/'COMPLETE.json',{'identity_sha256':'test','files':{name:file_record(src/name) for name in ('trajectory.npz','result.json')}})
            before={str(p.relative_to(src)):file_record(p) for p in src.rglob('*') if p.is_file()}
            rec=compact_case(src,dst)
            self.assertEqual(rec['status'],'DERIVED_NOT_ORIGINAL')
            with np.load(dst/'trajectory.compact.npz') as compact:
                self.assertNotIn('raw_pixels',compact.files)
                self.assertEqual(list(compact['raw_pixel_sha256']),[frame_hash(p) for p in pixels])
                for key,value in arrays.items():
                    if key!='raw_pixels':np.testing.assert_array_equal(compact[key],value)
            self.assertEqual(before,{str(p.relative_to(src)):file_record(p) for p in src.rglob('*') if p.is_file()})
            self.assertEqual(compact_case(src,dst),rec)
            atomic_npz(src/'trajectory.npz',raw_pixels=pixels+1)
            with self.assertRaises(RuntimeError):compact_case(src,dst)

if __name__=='__main__':unittest.main()
