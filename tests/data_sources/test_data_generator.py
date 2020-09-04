from unittest import TestCase
from data_sources.data_generator import ExamplesGenerator


class TestExamplesGenerator(TestCase):
    def test_init(self):
        """can we build the damn thing?"""
        _ = ExamplesGenerator()

    def _get_eg(self):
        eg = ExamplesGenerator(seed=7357, pattern=[(10, 123), (10, 456), (10, 789)])
        example_pat = eg.insert_pattern(list(range(100)), eg.pattern)
        self.assertTrue(eg.has_pattern(example_pat, eg.pattern))
        return eg, example_pat

    def test_call_seed(self):
        """How does this seed set up at  init works if we call the random generator later"""
        eg = ExamplesGenerator(seq_len=10, seed=456)
        examples = [next(eg())[0] for _ in range(10)]
        self.assertEqual(list(examples[0][0:3]), [27, 43, 89])

    def test_insert_pattern(self):
        eg = ExamplesGenerator(seq_len=10, seed=7357, pattern=[(5, 123), (2, 456)])
        examples_pat = eg.insert_pattern(list(range(10)), eg.pattern)
        self.assertEqual(examples_pat[4:6], [123, 456])

    def test_has_pattern(self):
        eg, _ = self._get_eg()
        self.assertFalse(eg.has_pattern(list(range(100)), eg.pattern))

        """ test recursive find"""

        eg = ExamplesGenerator(seed=7357, pattern=[(3, 123), (3, 456), (3, 789)])
        example = [1, 123, 123, 1, 456, 456, 1, 1, 789]
        self.assertTrue(eg.has_pattern(example, eg.pattern))

    def test_pattern_indices(self):
        eg = ExamplesGenerator(seed=7357, pattern=[(10, 123), (10, 456), (10, 789)])
        example_pat = eg.insert_pattern(list(range(100)), eg.pattern)
        self.assertEqual(eg.find_pattern_indices(example_pat, eg.pattern), [8, 14, 18])

    def test_remove_pattern(self):
        eg, example_pat = self._get_eg()
        example_pat = eg.remove_pattern(example_pat, eg.pattern)
        self.assertFalse(eg.has_pattern(example_pat, eg.pattern))

    def test_proportions(self):
        eg = ExamplesGenerator(seed=7357, pattern=[(10, 123), (10, 456), (10, 789)])
        examples = [next(eg()) for _ in range(10000)]
        has_pattern = [eg.has_pattern(e[0], eg.pattern) for e in examples]
        self.assertEqual(sum(has_pattern), 4953)

    def test_multiple_patterns(self):
        eg = ExamplesGenerator(seed=7357, multiple_patterns=[
            [
                [(10, 123), (10, 456), (10, 789)],
                [(10, 789), (10, 456), (10, 123)]
            ],
            [0.8, 0.2]
        ])
        examples = [next(eg()) for _ in range(10000)]
        has_pattern_0 = [eg.has_pattern(e[0], eg.all_patterns[0]) for e in examples]
        has_pattern_1 = [eg.has_pattern(e[0], eg.all_patterns[1]) for e in examples]
        self.assertTrue(sum(has_pattern_0) + sum(has_pattern_1) > 4000)
        self.assertTrue(sum(has_pattern_0) > sum(has_pattern_1) * 3)

    def test_fp_rate(self):
        pattern = [(10, 123), (10, 456), (10, 789)]
        eg = ExamplesGenerator(seed=7357, pattern=pattern, fp_rate=0.5)

        examples = [next(eg()) for _ in range(1000)]
        positive_examples = [e[0] for e in examples if e[1]]
        has_pattern = [eg.has_pattern(e, pattern) for e in positive_examples]
        self.assertTrue(sum(has_pattern) < 0.6 * len(positive_examples))
