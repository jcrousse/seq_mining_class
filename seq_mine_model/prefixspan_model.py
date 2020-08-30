from prefixspan import PrefixSpan

from data_sources.data_generator import ExamplesGenerator


VOCAB_SIZE = 100
SEQ_LEN = 100
PATTERN = [(2, 10), (2, 11), (2, 12), (2, 13), (2, 14)]

data_generator = ExamplesGenerator(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=111, pattern=PATTERN)

data_sequences = [next(data_generator()) for _ in range(100)]
positive_sequences = [s[0] for s in data_sequences if s[1] == 1]
negative_sequences = [s[0] for s in data_sequences if s[1] == 0]


positive_seq = PrefixSpan(positive_sequences).frequent(30)
long_seq = [s for s in positive_seq if len(s[1]) >= 4]
freq_seq = sorted(long_seq, key=lambda x: len(x[1]), reverse=True)
seq_by_freq = sorted(long_seq, key=lambda x: x[0], reverse=True)

# ignore sub-sequences if no more frequent. and ignore longer sequences with same prefix if less frequent
keep_seq = [False] * len(long_seq)
for idx, seq1 in enumerate(long_seq):
    freq_1, s1 = seq1[0], seq1[1]
    for seq2 in long_seq:
        freq_2, s2 = seq2[0], seq2[1]
        if len(s1) > len(s2) and freq_1 < freq_2 and s2 == s1[0:len(s2)]:
            keep_seq[idx] = False
            break
        if len(s2) > len(s1) and freq_2 >= freq_1:
            p = data_generator.seq_to_pattern(s1)
            if data_generator.has_pattern(s2, p):
                keep_seq[idx] = False
                break


print(freq_seq[0:10])
print(seq_by_freq[0:10])
