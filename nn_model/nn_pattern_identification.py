import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from data_sources.data_generator import ExamplesGenerator, get_multiple_patterns

hyperparameter_defaults = dict(
    dropout=0.5,
    rnn_size1=16,
    rnn_size2=32,
    dense_size1=64,
    dense_size2=32,
    learning_rate=0.001,
    )

wandb.init(config=hyperparameter_defaults, project="seq_mining")
config = wandb.config


def get_dataset(seq_len, vocab_size, seed, pattern=None, batch_size=200, multiple_patterns=None, **kwargs):
    data_generator = ExamplesGenerator(seq_len=seq_len, vocab_size=vocab_size, seed=seed,
                                       pattern=pattern, multiple_patterns=multiple_patterns, **kwargs)
    dataset = tf.data.Dataset.from_generator(data_generator,
                                             output_types=(tf.int64, tf.int64),
                                             output_shapes=(tf.TensorShape([seq_len]), tf.TensorShape([]))
                                             ).batch(batch_size)
    return dataset


def get_model(vocab_size, embed_size=4):
    inputs = tf.keras.layers.Input(shape=(None,), name="input")
    embedded = tf.keras.layers.Embedding(vocab_size, embed_size)(inputs)
    seq_model = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(config.rnn_size1))(embedded)
    seq_model = tf.keras.layers.Dropout(config.dropout)(seq_model)

    dense1 = tf.keras.layers.Dense(config.dense_size1, activation='relu')(seq_model)
    dense1 = tf.keras.layers.Dropout(config.dropout)(dense1)

    dense2 = tf.keras.layers.Dense(config.dense_size2, activation='relu')(dense1)
    dense2 = tf.keras.layers.Dropout(config.dropout)(dense2)

    output = tf.keras.layers.Dense(1, name="output")(dense2)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def run_experiment(name, seq_len, vocab_size, pattern=None, batch_size=32, data_limit=None, multiple_patterns=None,
                   **kwargs):

    train_dataset = get_dataset(seq_len, vocab_size, 111,
                                pattern=pattern, multiple_patterns=multiple_patterns, **kwargs)
    validation_dataset = get_dataset(seq_len, vocab_size, 222, batch_size=batch_size,
                                     pattern=pattern, multiple_patterns=multiple_patterns, **kwargs)
    test_dataset = get_dataset(seq_len, vocab_size, 333, batch_size=batch_size,
                               pattern=pattern, multiple_patterns=multiple_patterns, **kwargs)

    if isinstance(data_limit, int):
        num_batches = data_limit // batch_size
        train_dataset = train_dataset.take(num_batches).repeat()
        validation_dataset = validation_dataset.take(num_batches).repeat()

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f"tb_logs/{name}",
                                                 histogram_freq=10)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.05,
        patience=15)

    if multiple_patterns:
        all_patterns = multiple_patterns[0]
    else:
        all_patterns = [pattern]

    actual_vocab_size = max([vocab_size] + [e[1] for p in all_patterns for e in p])

    model = get_model(actual_vocab_size + 1)
    model.compile(optimizer='rmsprop',
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
        # "alpha": {
        #     'seq_len': 20,
        #     'vocab_size': 20,
        #     'pattern': [(2, t) for t in tokens1]
        # },
        # "bravo": {
        #     'seq_len': 50,
        #     'vocab_size': 50,
        #     'pattern': [(5, t) for t in tokens1]
        # },
        # "charlie": {
        #     'seq_len': 100,
        #     'vocab_size': 100,
        #     'pattern': [(20, t) for t in tokens1]
        # },
        # "delta": {
        #     'seq_len': 250,
        #     'vocab_size': 500,
        #     'pattern': [(30, t) for t in tokens1]
        # },
        # "alpha_200": {
        #     'seq_len': 50,
        #     'vocab_size': 1000,
        #     'pattern': [(10, t) for t in tokens1],
        #     'data_limit': 200
        # },
        # "bravo_200": {
        #     'seq_len': 100,
        #     'vocab_size': 1000,
        #     'pattern': [(20, t) for t in tokens1],
        #     'data_limit': 200
        # },
        # "charlie_200": {
        #     'seq_len': 200,
        #     'vocab_size': 1000,
        #     'pattern': [(40, t) for t in tokens1],
        #     'data_limit': 200
        # },
        # "delta_200": {
        #     'seq_len': 300,
        #     'vocab_size': 500,
        #     'pattern': [(30, t) for t in tokens1],
        #     'data_limit': 200
        # },
        "alpha_200_mp": {
            'seq_len': 50,
            'vocab_size': 1000,
            'multiple_patterns': get_multiple_patterns(10),
            'fp_rate': 0.05,
            'fn_rate': 0.05,
            'data_limit': 200
        },
        # "bravo_200_mp": {
        #     'seq_len': 100,
        #     'vocab_size': 1000,
        #     'multiple_patterns': get_multiple_patterns(5),
        #     'fp_rate': 0.05,
        #     'fn_rate': 0.05,
        #     'data_limit': 200
        # },
    }
    for exp_name, exp_parameters in experiments.items():
        run_experiment(exp_name, **exp_parameters)
