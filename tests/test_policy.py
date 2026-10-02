import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from policy import PaperPolicy,tile_qp

class PolicyTests(unittest.TestCase):
    def test_ranking_and_ties(self):
        offsets=tile_qp(np.arange(45))
        self.assertEqual(np.flatnonzero(offsets==-2).tolist(),list(range(39,45)))
        self.assertEqual(np.flatnonzero(offsets==-1).tolist(),list(range(33,39)))
        ties=tile_qp(np.zeros(45))
        self.assertEqual(np.flatnonzero(ties==-2).tolist(),list(range(6)))
    def test_codec_block_maps(self):
        scores=np.arange(45).reshape(5,9)
        for codec,shape in [('h264',(128,256)),('hevc',(64,128)),('av1',(32,64))]:
            policy=PaperPolicy(codec)
            result=policy.make_map(scores,None,codec,0)
            self.assertEqual(result.shape,shape)
            self.assertEqual(result.dtype,np.int8)
            self.assertEqual(set(np.unique(result)),{-2,-1,0})
    def test_reject_invalid_scores(self):
        for values in [np.zeros(44),np.full(45,np.nan),np.full(45,-1)]:
            with self.assertRaises(ValueError):tile_qp(values)
if __name__=='__main__':unittest.main()
