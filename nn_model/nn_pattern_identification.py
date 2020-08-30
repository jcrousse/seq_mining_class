import tensorflow as tf
from data_sources.data_generator import ExamplesGenerator


# TODO:
#  -Adjust seq mining to find sequences that are over-represented in on label.
#       -Remove sub-sequences that are not higher in predictive power.
#       -One single var model per sequence? Groups of sequences?
#  -How to simulate sub sequences that are positive or negative within the pattern? Is it necessary?
#       Or just mention it?
#  -More realistic: More than one possible pattern?

def get_dataset(seq_len, vocab_size, seed, pattern, batch_size=200):
    data_generator = ExamplesGenerator(seq_len=seq_len, vocab_size=vocab_size, seed=seed, pattern=pattern)
    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int64, tf.int64),
                                             output_shapes=(tf.TensorShape([seq_len]), tf.TensorShape([]))
                                             ).batch(batch_size)
    return dataset


def get_model(vocab_size, embed_size=4):
    lstm_units = 16
    inputs = tf.keras.layers.Input(shape=(None,), name="input")
    embedded = tf.keras.layers.Embedding(vocab_size, embed_size)(inputs)
    seq_model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units))(embedded)
    output = tf.keras.layers.Dense(1, name="output")(seq_model)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def run_experiment(name, seq_len, vocab_size, pattern, data_limit=None):
    train_dataset = get_dataset(seq_len, vocab_size, 111, pattern)
    validation_dataset = get_dataset(seq_len, vocab_size, 222, pattern, batch_size=256)
    test_dataset = get_dataset(seq_len, vocab_size, 333, pattern, batch_size=256)

    if isinstance(data_limit, int):
        train_dataset = get_dataset(seq_len, vocab_size, 111, pattern, batch_size=data_limit).take(1).repeat()

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"tb_logs/{name}",
                                                 histogram_freq=10)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.05,
        patience=20)

    actual_vocab_size = max([vocab_size] + [e[1] for e in pattern])

    model = get_model(actual_vocab_size + 1)
    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  )

    model.fit(train_dataset,
              validation_data=validation_dataset,
              epochs=40,
              steps_per_epoch=50,
              validation_steps=1,
              callbacks=[tensorboard, early_stop])
    model.evaluate(test_dataset.take(1000), verbose=2)


if __name__ == '__main__':
    tokens1 = [10, 11, 12, 13, 14]
    experiments = {
        "alpha": {
            'seq_len': 20,
            'vocab_size': 20,
            'pattern': [(2, t) for t in tokens1]
        },
        "bravo": {
            'seq_len': 50,
            'vocab_size': 50,
            'pattern': [(5, t) for t in tokens1]
        },
        "charlie": {
            'seq_len': 100,
            'vocab_size': 100,
            'pattern': [(20, t) for t in tokens1]
        },
        "delta": {
            'seq_len': 250,
            'vocab_size': 500,
            'pattern': [(30, t) for t in tokens1]
        },
        "echo": {
            'seq_len': 1000,
            'vocab_size': 1000,
            'pattern': [(100, t) for t in tokens1]
        },
        "alpha_200": {
            'seq_len': 20,
            'vocab_size': 20,
            'pattern': [(2, t) for t in tokens1],
            'data_limit': 200
        },
        "bravo_200": {
            'seq_len': 50,
            'vocab_size': 50,
            'pattern': [(5, t) for t in tokens1],
            'data_limit': 200
        },
        "charlie_200": {
            'seq_len': 100,
            'vocab_size': 100,
            'pattern': [(20, t) for t in tokens1],
            'data_limit': 200
        },
        "delta_200": {
            'seq_len': 250,
            'vocab_size': 500,
            'pattern': [(30, t) for t in tokens1],
            'data_limit': 200
        },
        "echo_200": {
            'seq_len': 1000,
            'vocab_size': 1000,
            'pattern': [(100, t) for t in tokens1],
            'data_limit': 200
        },
        "forxtrot_200": {
            'seq_len': 2000,
            'vocab_size': 2000,
            'pattern': [(100, t) for t in tokens1],
            'data_limit': 200
        },
        "golf_200": {
            'seq_len': 5000,
            'vocab_size': 10000,
            'pattern': [(100, t) for t in tokens1],
            'data_limit': 200
        },
    }
    for exp_name, exp_parameters in experiments.items():
        run_experiment(exp_name, **exp_parameters)
