"""Real separate-grid comparison; no production configuration edits or sweep."""
from pathlib import Path
import os
import tempfile
import unittest

from validate_mo_parallel import run_check, available_cpus


class ParallelIntegration(unittest.TestCase):
    # Params: self; returns: None.
    def test_sequential_and_concurrent_grid_retain_contracts_and_recover(self):
        capacity = available_cpus()
        if capacity is not None and capacity < 6:
            self.skipTest('Concurrent smoke check requires six allocated/visible CPUs')
        retained = os.environ.get('MO_PARALLEL_SMOKE_OUTPUT')
        if retained:
            evidence = run_check(Path(retained))
        else:
            with tempfile.TemporaryDirectory(prefix='mo_parallel_integration_') as temporary:
                evidence = run_check(Path(temporary) / 'check')
        self.assertEqual(evidence['status'], 'passed')
        self.assertEqual(evidence['contract']['complete_policies'], 4)
        self.assertEqual(evidence['contract']['requested_total'], 5)
        self.assertEqual(evidence['contract']['effective_total'], 8)
        self.assertEqual(len(evidence['contract']['evaluation_contracts']), 12)
        self.assertTrue(evidence['duplicate_claim_refused'])
        self.assertTrue(evidence['incomplete_finalizer_refused'])


if __name__ == '__main__':
    unittest.main()
