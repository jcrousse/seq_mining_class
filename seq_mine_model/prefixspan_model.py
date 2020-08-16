from prefixspan import PrefixSpan
from data_sources.data_generator import ExamplesGenerator


VOCAB_SIZE = 1000
SEQ_LEN = 1000
PATTERN = [(100, 10), (100, 11), (100, 12), (100, 13), (100, 14)]

data_generator = ExamplesGenerator(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=111, pattern=PATTERN)

data_sequences = [next(data_generator()) for _ in range(100)]
positive_sequences = [s[0] for s in data_sequences if s[1] == 1]
negative_sequences = [s[0] for s in data_sequences if s[1] == 0]


long_seq = [s for s in PrefixSpan(positive_sequences).frequent(30) if len(s[1]) >= 4]
freq_seq = sorted(long_seq, key=lambda x: len(x[1]), reverse=True)
seq_by_freq = sorted(long_seq, key=lambda x: x[0], reverse=True)
print(freq_seq[0:10])
print(seq_by_freq[0:10])
