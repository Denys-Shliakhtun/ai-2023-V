from wsgiref import validate
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_feed_forward(neurons):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_cascade_forward(neurons, layers=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    if layers == 2:
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_elman(neurons, layers=1):
    layers_array = [tf.keras.layers.Dense(1)]
    for _ in range(layers):
        layers_array.append(tf.keras.layers.SimpleRNN(neurons, activation='relu', return_sequences=True))
        layers_array.append(tf.keras.layers.Dense(1))
    model = tf.keras.Sequential(layers_array)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_data():
    values_x = np.arange(1, 2, 0.01)
    values_y = np.sin(abs(values_x)) * np.cos(values_x/2)
    values_z = values_y * np.sin(values_x)
    input_data = np.column_stack((values_x, values_y))
    output_data = values_z.reshape(-1, 1)
    return input_data, output_data

def train(model, X_train, y_train, X_val, y_val, epochs=100):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), verbose=0)
    return history.history

def display(history, name):
    print(f"Model name: {name}")
    print(f"Final training loss: {history['loss'][-1]:.10f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.10f}\n\n")
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label=f'{name} (training loss)')
    plt.plot(history['val_loss'], label=f'{name} (validation loss)')
    plt.title('Training & validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
input_data, output_data = generate_data()
input_rnn = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, shuffle=False)
x_train_rnn, x_test_rnn = x_train.reshape(x_train.shape[0], 1, 2), x_test.reshape(x_test.shape[0], 1, 2)

models = {
    'a) feed forward - 10 neurons': create_feed_forward(10),
    'b) feed forward - 20 neurons': create_feed_forward(20),
    'a) cascade forward - 20 neurons': create_cascade_forward(20),
    'b) cascade forward - 2 layers, 10 neurons each': create_cascade_forward(10, 2),
    'a) elman - 15 neurons': create_elman(15),
    'b) elman - 3 layers, 5 neurons each': create_elman(5, 3)
}

# plot graph
def plot_graph(x_train, z_train, x_test, z_test, z_predicted, label):
    plt.plot(x_train, z_train, label="Training value")
    plt.plot(x_test, z_test, label="True value")
    plt.plot(x_test, z_predicted, label="Predicted value")
    plt.title(label)
    plt.legend()
    plt.grid(True)
    plt.show()
    return

for name, model in models.items():
    if 'elman' in name:
        history = train(model, x_train_rnn, y_train, x_test_rnn, y_test, epochs=100)
        x_data, y_true = x_test_rnn, y_test
        y_pred = model.predict(x_test_rnn)
        display(history, name)
        plot_graph(x_train[:, 0], y_train, x_test[:, 0], y_true, y_pred[:, 0], f'{name} Predictions')
    else:
        history = train(model, x_train, y_train, x_test, y_test, epochs=100)
        x_data, y_true = x_test, y_test
        y_pred = model.predict(x_test)
        display(history, name)
        plot_graph(x_train[:, 0], y_train, x_data[:, 0], y_true, y_pred, f'{name} Predictions')

    

