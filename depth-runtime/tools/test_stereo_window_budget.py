import unittest
from evaluate_stereo_temporal_lens import validate_window_budget

class WindowBudgetTests(unittest.TestCase):
    def test_original_limits_remain_and_continuous_is_explicit(self):
        validate_window_budget(12,200)
        validate_window_budget(60,400)
        validate_window_budget(100,1000,True)
        for args in [(61,400),(60,401),(100,1000),(121,1000,True),(100,1001,True),(True,200),(12,True),(100,1000,1)]:
            with self.subTest(args=args),self.assertRaises(ValueError):validate_window_budget(*args)

if __name__=='__main__':unittest.main()
