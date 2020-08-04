import numpy as np
import random


class ExamplesGenerator:
    """ExampleGenerator generates fixed length random sequences of tokens (integers).
    A percentage of the generated sequences have a pre-defined 'pattern' which is a sequence of tokens,
    and the number of tokens between them.
    The pattern may also occur randomly.
     """
    def __init__(self, seq_len=100, vocab_size=100, seed=None, pos_pct=0.5, pattern=None):
        """
        :param seq_len Length of the sequences
        :param vocab_size Number of different values the tokens can take
        :param seed Random seed
        :param pos_pct Percentage of examples where the pattern is "enforced"
        :param: pattern
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pos_pct = pos_pct
        self.pattern = pattern or []
        np.random.seed(seed)
        self.count_p = 0

    def __call__(self):
        """main call to generate an example"""
        while True:
            example_nopattern = np.random.randint(0, self.vocab_size, self.seq_len)
            example_pattern = self.random_pattern(example_nopattern)
            yield example_pattern

    def random_pattern(self, example):
        """ tosses a coin to decide whether to modify the example to add the pattern or to remove it"""
        if np.random.rand() > self.pos_pct:
            self.count_p += 1
            if not self.has_pattern(example):
                example = self.insert_pattern(example)
        else:
            if self.has_pattern(example):
                example = self.remove_pattern(example)
        return example

    def insert_pattern(self, examples):
        if self.can_fit_pattern(examples):
            idx_replace = 0
            for token_data in self.pattern:
                max_spaces = token_data[0]
                token_value = token_data[1]
                idx_replace += np.random.randint(1, max_spaces)
                examples[idx_replace] = token_value
        return examples

    def remove_pattern(self, example):
        """ while a pattern is found, randomly replace a random element from the pattern"""
        pattern_indices = self.find_pattern_indices(example)
        while pattern_indices:
            replace_idx = random.choice(pattern_indices)
            example[replace_idx] = np.random.randint(0, self.vocab_size)
            pattern_indices = self.find_pattern_indices(example)
        return example

    def can_fit_pattern(self, examples):
        """ check that the pattern fits in the example. If not, we should throw a warning (only once).
        Currently the only consraint is that the length of the examples is longer or equal to the pattern length
        (as there is currently no minimal amount of tokens between two pattern token).
        Change here if we want to introduce minimal number of tokens between two patten tokens"""
        return len(self.pattern) <= len(examples)

    def find_pattern_indices(self, example, sub_pattern=None):
        """ return the index location of pattern tokens.
        empty list if no pattern found"""
        if sub_pattern is None:
            sub_pattern = self.pattern
        if len(sub_pattern) == 0:
            return []
        else:
            max_spaces = sub_pattern[0][0]
            token_value = sub_pattern[0][1]
            token_indices = [i for i, t in enumerate(example[0:max_spaces]) if t == token_value]
            for idx in token_indices:
                tail_indices = self.find_pattern_indices(example[idx + 1:], sub_pattern[1:])
                tail_indices_adjusted = [i + idx + 1 for i in tail_indices]
                token_indices = [idx] + tail_indices_adjusted
                if len(token_indices) == len(sub_pattern):
                    return token_indices
            return []

    def has_pattern(self, example):
        pattern_idx = self.find_pattern_indices(example)
        return True if len(pattern_idx) == len(self.pattern) else False

