from KalmanFilter import KalmanFilter
import numpy as np
import unittest

class TestKF(unittest.TestCase):
    def test_with_x_v(self):
        x = 0.2
        v = 2.3
        kf = KalmanFilter(initial_x = x, initial_v = v, accel_variance=1.2)
        self.assertAlmostEqual(kf.pos, x)
        self.assertAlmostEqual(kf.vel, v)

    def test_predict(self):
        x = 0.2
        v = 2.3
        kf = KalmanFilter(initial_x = x, initial_v = v, accel_variance=1.2)
        kf.predict(dt=0.1)
        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2,))
    
    def test_predict_state_uncertainty(self):
        x = 0.2
        v = 2.3
        kf = KalmanFilter(initial_x = x, initial_v = v, accel_variance=1.2)
        for i in range(10):
            determinant_before = np.linalg.det(kf.cov)
            kf.predict(dt = 0.1)
            determinant_after = np.linalg.det(kf.cov)
            self.assertGreater(determinant_after, determinant_before)
            print(determinant_before, determinant_after)
    
    def test_update(self):
        x = 0.2
        v = 2.3
        kf = KalmanFilter(initial_x = x, initial_v = v, accel_variance=1.2)
        kf.update(meas_value=0.1, meas_variance=0.1)

