"""Checks that support diagnostics separate rotation, nesting, and empty spans."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gaussian_rate.support import support_statistics, load_saved_run


def covariance(u, bits):
    u = np.asarray(u, float)
    return np.eye(len(u))+(2**(-2*bits)-1)*np.outer(u, u)


class SupportTests(unittest.TestCase):
    def test_rotation_at_fixed_rank(self):
        u = np.array([1., 0., 0.]); v = np.array([np.sqrt(3)/2, .5, 0.])
        s = support_statistics([1, 2], [covariance(u, 1), covariance(v, 2)], [2, 1])
        np.testing.assert_array_equal(s['ranks'], [1, 1])
        self.assertAlmostEqual(s['overlap'][0, 1], .75)
        self.assertAlmostEqual(s['max_principal_angle'][0, 1], 30)

    def test_empty_nested_full_and_deactivation(self):
        K = [np.eye(3), np.diag([.25, 1, 1]), np.diag([.125, .5, 1]),
             np.diag([.125, .25, .5]), np.diag([.03125, 1, 1])]
        s = support_statistics([0, 1, 2, 3, 4], K, [5, 4, 3, 2, 1])
        self.assertTrue(np.isnan(s['overlap'][0]).all())
        self.assertEqual(s['max_principal_angle'][1, 2], 0)
        self.assertAlmostEqual(s['previous_support_retained'][2], 1)
        self.assertAlmostEqual(s['previous_support_retained'][4], 1/3)
        self.assertEqual(s['changes'][-1]['kind'], 'deactivation')
        self.assertEqual(s['changes'][1]['old_rank'], 1)
        self.assertEqual(s['changes'][1]['new_rank'], 2)

    def test_full_support_internal_rotation_is_invisible(self):
        a = .45; V = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        s = support_statistics([2, 3], [np.diag([.125, .5]), V@np.diag([.03125, .5])@V.T], [2, 1])
        self.assertAlmostEqual(s['overlap'][0, 1], 1)
        self.assertEqual(s['max_principal_angle'][0, 1], 0)

    def test_threshold_in_bits_and_sorting(self):
        s = support_statistics([2, 1], [np.diag([.1, 2**(-2e-7)]), np.diag([.2, 1])], [1, 2])
        np.testing.assert_array_equal(s['rates'], [1, 2])
        np.testing.assert_array_equal(s['ranks'], [1, 1])
        with self.assertRaises(ValueError): support_statistics([1, 1], [np.eye(2)]*2, [1, 1])

    def test_load_rate_selection_and_seed_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/'gaussian_tasks.npz', H=np.array([np.diag([2., 0.])]))
            rows = []
            for seed in [0, 1]:
                cell = root/f'seed_{seed}'; cell.mkdir()
                np.savez(cell/'channel.npz', K=np.diag([.25, 1.]))
                rows.append(dict(mode='gaussian', candidate='task_aware', budget=1, directory=cell.name))
            (root/'results.json').write_text(json.dumps(rows))
            data = load_saved_run(root)
            self.assertEqual(len(data['rates']), 1)
            self.assertAlmostEqual(data['D'][0], .5*np.log2(1.5))
            np.savez(root/'seed_1/channel.npz', K=np.diag([.5, 1.]))
            with self.assertRaises(ValueError): load_saved_run(root)


if __name__ == '__main__': unittest.main()
