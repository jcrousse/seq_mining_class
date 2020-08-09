import tensorflow as tf
from data_sources.data_generator import ExamplesGenerator
# TODO:
#  -Parameter search to try different dataset sizes, sequence sizes and pattern sizes. find NN breaking point.
#  -Tensorboard to illustrate where NN breaks (epochs and test accuracy)
#  -Test how Seq mining works where NN breaks
#  -How to simulate sub sequences that are positive or negative within the pattern? Is it necessary?
#       Or just mention it?


VOCAB_SIZE = 30
SEQ_LEN = 20
PATTERN = [(2, 10), (2, 11), (2, 12), (2, 13), (2, 14)]

actual_vocab_size = max([VOCAB_SIZE] + [e[1] for e in PATTERN])


def get_dataset(seq_len, vocab_size, seed, pattern):
    data_generator = ExamplesGenerator(seq_len=seq_len, vocab_size=vocab_size, seed=seed, pattern=pattern)
    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int64, tf.int64),
                                             output_shapes=(tf.TensorShape([seq_len]), tf.TensorShape([]))).batch(1024)
    return dataset


train_dataset = get_dataset(SEQ_LEN, VOCAB_SIZE, 111, PATTERN)
test_dataset = get_dataset(SEQ_LEN, VOCAB_SIZE, 222, PATTERN)


def get_model(vocab_size, embed_size=4):
    lstm_units = 16
    inputs = tf.keras.layers.Input(shape=(None,), name="input")
    embedded = tf.keras.layers.Embedding(vocab_size, embed_size)(inputs)
    seq_model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(embedded)
    output = tf.keras.layers.Dense(1, name="output")(seq_model)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


model = get_model(actual_vocab_size + 1)
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              )

model.fit(train_dataset, epochs=15, steps_per_epoch=10)
model.evaluate(test_dataset.take(10), verbose=2)
