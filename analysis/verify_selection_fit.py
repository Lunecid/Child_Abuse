#!/usr/bin/env python3
"""Verify the frozen choice fit through an independent statsmodels likelihood.

Requires authorized source access. Writes no individual data or predictions.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from statsmodels.discrete.conditional_models import ConditionalLogit
from selection_audit import read_minimal, verify_corpus, design


def verify(source, joint, summary_path):
    scores, labels, _ = read_minimal(source)
    available = verify_corpus(scores, labels, joint)
    mask=(available.sum(axis=1)>=2) & (labels<4)
    eligible=available[np.arange(len(labels)),np.minimum(labels,3)]
    mask &= eligible
    x=design(scores[mask],'combined'); a=available[mask]; y=labels[mask]
    row,alternative=np.where(a)
    long_x=x[row,alternative]
    long_y=(alternative==y[row]).astype(int)
    model=ConditionalLogit(long_y,long_x,groups=row)
    opt=minimize(lambda b:-model.loglike(b),np.zeros(4),
                 jac=lambda b:-model.score(b),method='BFGS',
                 options={'gtol':1e-7,'maxiter':200})
    frozen=json.loads(summary_path.read_text())['models']['combined']
    np.testing.assert_allclose(opt.x,frozen['parameters'],atol=3e-6,rtol=0)
    assert abs(model.loglike(opt.x)-frozen['log_likelihood'])<1e-8
    assert np.max(np.abs(model.score(opt.x)))<1e-5
    assert len(y)==frozen['n']
    print('PASS: independent statsmodels likelihood and score; coefficient error <3e-6, log-likelihood error <1e-8; eligible N='+str(len(y)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--joint-table',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    args=p.parse_args()
    verify(args.source_dir,args.joint_table,args.summary)
