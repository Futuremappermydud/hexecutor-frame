import unittest
import numpy as np
from stereo_spatial_support import spatial_track_folds

class SpatialSupportTests(unittest.TestCase):
    def test_track_integrity_and_no_right_or_later_pixel_selection(self):
        rows=np.array([[f,t,70+128*t+100*f,70+100*f,80,90] for t in range(6) for f in range(4)],float)
        labels,report=spatial_track_folds(rows)
        self.assertEqual(set(labels),{0,1})
        for t in range(6):self.assertEqual(len(set(labels[rows[:,1]==t])),1)
        altered=rows.copy();altered[:,4:]+=10000;altered[altered[:,0]>0,2:4]-=10000
        other,second=spatial_track_folds(altered)
        np.testing.assert_array_equal(labels,other);self.assertEqual(report,second)
        shuffled=np.random.default_rng(2211).permutation(len(rows))
        np.testing.assert_array_equal(spatial_track_folds(rows[shuffled])[0],labels[shuffled])
        with self.assertRaises(ValueError):spatial_track_folds(np.vstack((rows,rows[:1])))

if __name__=='__main__':unittest.main()
