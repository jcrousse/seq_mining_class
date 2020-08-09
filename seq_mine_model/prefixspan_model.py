from prefixspan import PrefixSpan
from data_sources.data_generator import ExamplesGenerator


VOCAB_SIZE = 30
SEQ_LEN = 20
PATTERN = [(2, 10), (2, 11), (2, 12), (2, 13), (2, 14)]

data_generator = ExamplesGenerator(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=111, pattern=PATTERN)

data_sequences = [next(data_generator()) for _ in range(10000)]
positive_sequences = [s[0] for s in data_sequences if s[1] == 1]
negative_sequences = [s[0] for s in data_sequences if s[1] == 0]


long_seq = [s for s in PrefixSpan(positive_sequences).frequent(2000) if len(s[1]) >= 4]
freq_seq = sorted(long_seq, key=lambda x: len(x[1]), reverse=True)

print(freq_seq[0:10])
