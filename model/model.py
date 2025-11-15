import tensorflow as tf
import numpy as np
import chess
import pandas as pd

def parse(file):
    chess_data = pd.read_csv('model/chess_evaluations/chessData.csv')

    first_column = chess_data.iloc[:, 0].tolist()
    second_column = chess_data.iloc[:, 1].tolist()

    return first_column, second_column

def reshape(data):
    red = data[:, 0:1024].reshape(-1, 32, 32)
    green = data[:, 1024:2048].reshape(-1, 32, 32)
    blue = data[:, 2048:3072].reshape(-1, 32, 32)

    images = np.stack([red, green, blue], axis=-1)

    return images


def main():

    x_train = []
    y_train = []

    for i in range(1, 6):
        data = unpickle(f'cifar-10-batches-py/data_batch_{i}')
        x_train.extend(data[b'data'])
        y_train.extend(data[b'labels'])

    test_data = unpickle('cifar-10-batches-py/test_batch')
    x_test = np.array(test_data[b'data'])
    y_test = np.array(test_data[b'labels'])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    x_train = reshape(x_train)
    x_test = reshape(x_test)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),

        #tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same"),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),

        #tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding="same"),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.4),


        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    initial_learning_rate = 0.001

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True)

    model.compile(optimizer=optimizer,
                  loss=loss_fn,
                  metrics=['accuracy'],
                  )

    model.fit(x_train, y_train, epochs=5, validation_split=0.1, callbacks=[early_stopping])

    model.evaluate(x_test, y_test, verbose=2)
