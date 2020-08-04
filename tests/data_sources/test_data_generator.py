from unittest import TestCase
from data_sources.data_generator import ExamplesGenerator


class TestExamplesGenerator(TestCase):
    def test_init(self):
        """can we build the damn thing?"""
        _ = ExamplesGenerator()

    def test_call_seed(self):
        """How does this seed set up at  init works if we call the random generator later"""
        eg = ExamplesGenerator(seq_len=10, seed=456)
        examples = [next(eg()) for _ in range(10)]
        self.assertEqual(list(examples[0][0:3]), [27, 43, 89])

    def test_insert_pattern(self):
        eg = ExamplesGenerator(seq_len=10, seed=7357, pattern=[(5, 123), (2, 456)])
        examples_pat = eg.insert_pattern(list(range(10)))
        self.assertEqual(examples_pat[4:6], [123, 456])

    def test_has_pattern(self):
        eg = ExamplesGenerator(seed=7357, pattern=[(10, 123), (10, 456), (10, 789)])
        example_pat = eg.insert_pattern(list(range(100)))
        self.assertTrue(eg.has_pattern(example_pat))
        self.assertFalse(eg.has_pattern(list(range(100))))

        """ test recursive find"""

        eg = ExamplesGenerator(seed=7357, pattern=[(3, 123), (3, 456), (3, 789)])
        example = [1, 123, 123, 1, 456, 456, 1, 1, 789]
        self.assertTrue(eg.has_pattern(example))

    def test_pattern_indices(self):
        eg = ExamplesGenerator(seed=7357, pattern=[(10, 123), (10, 456), (10, 789)])
        example_pat = eg.insert_pattern(list(range(100)))
        self.assertEqual(eg.find_pattern_indices(example_pat), [8, 14, 18])

    def test_remove_pattern(self):
        eg = ExamplesGenerator(seed=7357, pattern=[(10, 123), (10, 456), (10, 789)])
        example_pat = eg.insert_pattern(list(range(100)))
        self.assertTrue(eg.has_pattern(example_pat))
        example_pat = eg.remove_pattern(example_pat)
        self.assertFalse(eg.has_pattern(example_pat))

    def test_proportions(self):
        eg = ExamplesGenerator(seed=7357, pattern=[(10, 123), (10, 456), (10, 789)])
        examples = [next(eg()) for _ in range(10000)]
        has_pattern = [eg.has_pattern(e) for e in examples]
        self.assertEqual(sum(has_pattern), 4953)
