import tensorflow as tf
from data_sources.data_generator import ExamplesGenerator
# TODO:
#  -Model to identify presence of patterns in random dataset.
#       - To ensure model works: Seq lent 10 and pattern with values above vocab_size.
#  -Parameter search to try different dataset sizes, sequence sizes and pattern sizes.

train_generator = ExamplesGenerator(seq_len=20, vocab_size=5, seed=111, pattern=[(2, 10), (2, 15), (2, 20)])
test_generator = ExamplesGenerator(seq_len=20, vocab_size=5, seed=222, pattern=[(2, 10), (2, 15), (2, 20)])

train_dataset = tf.data.Dataset.from_generator(train_generator,
                                               output_types=(tf.int64, tf.int64),
                                               output_shapes=(tf.TensorShape([20]), tf.TensorShape([]))).batch(526)


def get_model(vocab_size, embed_size):
    lstm_units = 16
    inputs = tf.keras.layers.Input(shape=(None,), name="input")
    embedded = tf.keras.layers.Embedding(vocab_size, embed_size)(inputs)
    seq_model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(embedded)
    output = tf.keras.layers.Dense(1, name="output")(seq_model)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


model = get_model(21, 8)
model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              )

model.fit(train_dataset, epochs=100, steps_per_epoch=10)
