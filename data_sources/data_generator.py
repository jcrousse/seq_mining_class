import numpy as np


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

    def __call__(self):
        """main call to generate an example"""
        while True:
            examples_nopattern = np.random.randint(0, self.vocab_size, self.seq_len)
            examples_pattern = self.random_pattern(examples_nopattern)
            yield examples_pattern

    def random_pattern(self, examples):
        """ tosses a coin to decide whether to modify the example to add the pattern"""
        if np.random.rand() > self.pos_pct:
            examples = self.insert_pattern(examples)
        return examples

    def insert_pattern(self, examples):
        if self.can_fit_pattern(examples):
            idx_replace = 0
            for token_data in self.pattern:
                max_spaces = token_data[0]
                token_value = token_data[1]
                idx_replace += np.random.randint(0, max_spaces)
                examples[idx_replace] = token_value
        return examples

    def can_fit_pattern(self, examples):
        """ check that the pattern fits in the example. If not, we should throw a warning (only once).
        Currently the only consraint is that the length of the examples is longer or equal to the pattern length
        (as there is currently no minimal amount of tokens between two pattern token).
        Change here if we want to introduce minimal number of tokens between two patten tokens"""
        return len(self.pattern) <= len(examples)

    def has_pattern(self, example):
        idx_min = 0
        for token_data in self.pattern:
            max_spaces = token_data[0]
            token_value = token_data[1]
            search_window = example[idx_min: idx_min + max_spaces]
            if token_value not in search_window:
                return False
            idx_min = idx_min + search_window.index(token_value)
        return True
