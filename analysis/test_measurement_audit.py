import unittest
import numpy as np
from measurement_audit import ordered_design, separation


class MeasurementAuditTests(unittest.TestCase):
    def test_order_model_is_invariant_to_common_nonlinear_increasing_transform(self):
        scores=np.array([[5,8,8,0],[6,5,10,0]])
        available=np.array([[1,1,1,0],[1,1,1,0]],dtype=bool)
        np.testing.assert_array_equal(ordered_design(scores,available,'combined'),
                                      ordered_design(scores**3,available,'combined'))
        np.testing.assert_array_equal(ordered_design(scores,available,'order')[0,:,0],[0,1,1,0])

    def test_separation_diagnostic_distinguishes_finite_and_unbounded_likelihood(self):
        x=np.array([[[0.],[1.]],[[0.],[1.]]]);available=np.ones((2,2),bool)
        self.assertTrue(separation(x,available,np.array([1,1]))['separating_direction'])
        self.assertFalse(separation(x,available,np.array([0,1]))['separating_direction'])


if __name__=='__main__':unittest.main()
