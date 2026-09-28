"""Synthetic checks of choice likelihood, gradients, and candidate masking."""
import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from selection_audit import design, likelihood, fit

class ChoiceLikelihoodTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(5107)
        self.scores=rng.uniform(0,10,(2000,4))
        self.x=design(self.scores,'combined')
        self.available=rng.random((2000,4))>.3
        self.available[:,:2]=True
        self.beta=np.array([.4,-.6,.3,.8])
        probabilities=likelihood(self.beta,self.x,self.available,np.zeros(2000,dtype=int))[3]
        self.y=np.array([rng.choice(4,p=p) for p in probabilities])

    def test_gradient_and_hessian(self):
        nll=lambda b: likelihood(b,self.x,self.available,self.y)[0]
        grad=lambda b: likelihood(b,self.x,self.available,self.y)[1]
        analytic=likelihood(self.beta,self.x,self.available,self.y)
        np.testing.assert_allclose(analytic[1],approx_derivative(nll,self.beta).ravel(),atol=1e-5)
        np.testing.assert_allclose(analytic[2],approx_derivative(grad,self.beta),rtol=1e-6,atol=1e-5)

    def test_record_shift_invariance_and_mask(self):
        shifted=design(self.scores+np.arange(len(self.scores))[:,None],'combined')
        original=likelihood(self.beta,self.x,self.available,self.y)
        alternative=likelihood(self.beta,shifted,self.available,self.y)
        np.testing.assert_allclose(original[3],alternative[3],atol=1e-12)
        self.assertTrue((original[3][~self.available]==0).all())
        np.testing.assert_allclose(original[3].sum(axis=1),1)

    def test_fit_recovery_and_ineligible_choice(self):
        b,_,cov,_=fit(self.x,self.available,self.y)
        self.assertTrue((np.abs(b-self.beta)<4*np.sqrt(np.diag(cov))).all())
        unavailable=self.available.copy(); unavailable[0,self.y[0]]=False
        with self.assertRaises(ValueError):
            fit(self.x,unavailable,self.y)

if __name__=='__main__':
    unittest.main()
