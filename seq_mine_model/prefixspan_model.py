from prefixspan import PrefixSpan

from data_sources.data_generator import ExamplesGenerator, get_multiple_patterns

VOCAB_SIZE = 1000
SEQ_LEN = 250
multiple_patterns = get_multiple_patterns(10)

NUM_EXAMPLES = 200
MIN_FREQ = 25
MIN_LEN = 5
MIN_DIST = 3

data_generator = ExamplesGenerator(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, seed=111,
                                   multiple_patterns=multiple_patterns)

data_sequences = [next(data_generator()) for _ in range(NUM_EXAMPLES)]
positive_sequences = [s[0] for s in data_sequences if s[1] == 1]
negative_sequences = [s[0] for s in data_sequences if s[1] == 0]

positive_seq = PrefixSpan(positive_sequences).frequent(MIN_FREQ)
long_seq = [s for s in positive_seq if len(s[1]) >= MIN_LEN]
seq_by_freq = sorted(long_seq, key=lambda x: x[0], reverse=True)


def distance_from_seqs(s, s_list: list):
    """return distance (in terms of number of different tokens) between the sequence s
    and the list of sequence s_list"""
    if not s_list:
        s_list = [[]]
    dist_per_seq = [len(set(s) - set(s2)) for s2 in s_list]
    return min(dist_per_seq)


most_freq_seq = []
for s in seq_by_freq:
    if distance_from_seqs(s[1], most_freq_seq) >= MIN_DIST:
        most_freq_seq.append(s[1])

print(most_freq_seq[0:10])
