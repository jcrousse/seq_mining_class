import numpy as np
import random


class ExamplesGenerator:
    """ExampleGenerator generates fixed length random sequences of tokens (integers).
    A percentage of the generated sequences have a pre-defined 'pattern' which is a sequence of tokens,
    and the number of tokens between them.
    The pattern may also occur randomly.
     """
    def __init__(self, seq_len: int = 100, vocab_size: int = 100, seed=None, pos_pct: float = 0.5, pattern: list = None,
                 multiple_patterns: list = None, fp_rate: float = 0.0, fn_rate: float = 0.0):
        """
        :param seq_len Length of the sequences
        :param vocab_size Number of different values the tokens can take
        :param seed Random seed
        :param pos_pct Percentage of examples where the pattern is "enforced"
        :param: pattern: A list of tuples (num_elements <int>, token_id<int>): Number of elements between two tokens
        and token_id, the expected token to be found.
        :param multiple_patterns: list containing two lists: A list of patterns (see pattern),
        and a list of weights (float)
        :param fp_rate: float, false positive rate
        :param fn_rate: float, false negative rate
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pos_pct = pos_pct
        self.pattern = pattern or []
        np.random.seed(seed)
        self.count_p = 0
        if multiple_patterns:
            self.all_patterns = multiple_patterns[0]
            self.patterns_weight = multiple_patterns[1]
        else:
            self.all_patterns = [self.pattern]
            self.patterns_weight = [1.0]

        self.fp_rate = fp_rate
        self.fn_rate = fn_rate

    def __call__(self):
        """main call to generate an example"""
        while True:
            example_nopattern = np.random.randint(0, self.vocab_size, self.seq_len)
            example_pattern, label = self.random_pattern(example_nopattern)
            yield example_pattern, label

    def random_pattern(self, example):
        """ tosses a coin to decide whether to modify the example to add the pattern or to remove it"""
        label = 0
        add_pattern = False
        remove_pattern = False
        pattern = []

        if np.random.rand() > self.pos_pct:
            self.count_p += 1
            label = 1
            pattern = random.choices(self.all_patterns, weights=self.patterns_weight, k=1)[0]
            if random.random() > self.fp_rate:
                add_pattern = True
            else:
                remove_pattern = True
        else:
            if random.random() > self.fn_rate:
                remove_pattern = True

        if add_pattern:
            if not self.has_pattern(example, pattern):
                example = self.insert_pattern(example, pattern)
        if remove_pattern:
            for p in self.all_patterns:
                if self.has_pattern(example, p):
                    example = self.remove_pattern(example, p)
        return example, label

    def insert_pattern(self, examples, pattern):
        if self.can_fit_pattern(examples, pattern):
            idx_replace = 0
            for token_data in pattern:
                max_spaces = token_data[0]
                token_value = token_data[1]
                idx_replace += np.random.randint(1, max_spaces)
                examples[idx_replace] = token_value
        return examples

    def remove_pattern(self, example, pattern):
        """ while a pattern is found, randomly replace a random element from the pattern"""
        pattern_indices = self.find_pattern_indices(example, pattern)
        while pattern_indices:
            replace_idx = random.choice(pattern_indices)
            example[replace_idx] = np.random.randint(0, self.vocab_size)
            pattern_indices = self.find_pattern_indices(example, pattern)
        return example

    @staticmethod
    def can_fit_pattern(examples, pattern):
        """ check that the pattern fits in the example. If not, we should throw a warning (only once).
        Currently the only constraint is that the length of the examples is longer or equal to the pattern length
        (as there is currently no minimal amount of tokens between two pattern token).
        Change here if we want to introduce minimal number of tokens between two patten tokens"""
        return len(pattern) <= len(examples)

    @staticmethod
    def find_pattern_indices(example, pattern):
        """ return the index location of pattern tokens.
        empty list if no pattern found"""
        if len(pattern) == 0:
            return []
        else:
            max_spaces = pattern[0][0]
            token_value = pattern[0][1]
            token_indices = [i for i, t in enumerate(example[0:max_spaces]) if t == token_value]
            for idx in token_indices:
                tail_indices = ExamplesGenerator.find_pattern_indices(example[idx + 1:], pattern[1:])
                tail_indices_adjusted = [i + idx + 1 for i in tail_indices]
                token_indices = [idx] + tail_indices_adjusted
                if len(token_indices) == len(pattern):
                    return token_indices
            return []

    @staticmethod
    def has_pattern(example, pattern):
        pattern_idx = ExamplesGenerator.find_pattern_indices(example, pattern)
        return True if len(pattern_idx) == len(pattern) else False

    @staticmethod
    def seq_to_pattern(seq, gap_size=1):
        """Convert a sequence (list of tokens) to a patter (list of tuples (gap_size, token))
        """
        return [(gap_size, t) for t in seq]


def get_multiple_patterns(n=2):
    return [
        [
            [(n, 10), (n, 11), (n, 12), (n, 13), (n, 14)],
            [(n, 1), (n, 2), (n, 3), (n, 4), (n, 5)],
            [(n, 7), (n, 8), (n, 9), (n, 10), (n, 11)]
        ],
        [0.4, 0.3, 0.3]
    ]
