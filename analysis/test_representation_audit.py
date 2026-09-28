"""Synthetic checks of estimands, without participant-level source data."""
import unittest

import numpy as np

from reproduce_representation_audit import statistics


class RepresentationMeasures(unittest.TestCase):
    def test_capacity_and_no_match_are_distinct(self):
        # Four single-domain profiles and one two-domain profile; the last
        # category does not match either of its two positive domains.
        domains = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0]])
        matched = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]])
        weights = np.array([1,1,1,1,1])
        rates, metrics = statistics(weights, domains, matched)
        np.testing.assert_allclose(rates[0], [.5,.5,0,0])
        np.testing.assert_allclose(metrics[0], [.2,.8,4/6,5/6,.8])

    def test_aggregation_preserves_each_statistic(self):
        # Compare two encodings of the same small synthetic distribution.
        d=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0]])
        m=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,1,0,0]])
        counts=np.array([2,3,1,1,2])
        aggregated=statistics(counts,d,m)
        expanded=statistics(np.ones(counts.sum()),np.repeat(d,counts,axis=0),np.repeat(m,counts,axis=0))
        for a,b in zip(aggregated,expanded):
            np.testing.assert_allclose(a,b)

    def test_cell_order_does_not_change_estimands(self):
        d=np.eye(4,dtype=int)
        counts=np.array([1,2,3,4])
        first=statistics(counts,d,d)
        second=statistics(counts[::-1],d[::-1],d[::-1])
        for a,b in zip(first,second):
            np.testing.assert_allclose(a,b)


if __name__ == '__main__':
    unittest.main()
