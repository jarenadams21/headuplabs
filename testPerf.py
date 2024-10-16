# test_quantum_grant_search.py

import unittest
from qstate import QuantumGrantSearcher, grants

class TestQuantumGrantSearcher(unittest.TestCase):
    def setUp(self):
        self.searcher = QuantumGrantSearcher(grants)

    def test_exact_match(self):
        query = "renewable energy projects in Boston"
        matching_indices, partial_match_scores = self.searcher.encode_query(query)
        # Assuming grants at indices 0, 4, 8, 28, 36, 40 are relevant based on sample data
        expected_indices = [0, 4, 8, 28, 36, 40]
        self.assertEqual(set(matching_indices), set(expected_indices))

    def test_partial_match(self):
        query = "AI research funding"
        matching_indices, partial_match_scores = self.searcher.encode_query(query)
        # Assuming grants at indices 18, 37, 55 are partial matches based on sample data
        expected_partial = [18, 37, 55]
        self.assertEqual(set([idx for idx, score in partial_match_scores]), set(expected_partial))

    def test_no_match(self):
        query = "non-existent grant category"
        matching_indices, partial_match_scores = self.searcher.encode_query(query)
        self.assertEqual(len(matching_indices), 0)
        self.assertEqual(len(partial_match_scores), 0)

    def test_search_output(self):
        query = "quantum computing research in Boston"
        # To test the search method, you might need to mock the quantum circuit execution
        # Here, we can test if the search method runs without errors
        try:
            self.searcher.search(query)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Search method raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()