import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_y(x):
    return x*np.cos(2*x) + np.sin(x/2)

def get_z(x,y):
    return np.sin(y) + np.cos(x/2)

def create_feed_forward_model(neurons_per_layer):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_cascade_forward_model(neurons_per_layer, hidden_layers=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu'))
    for _ in range(hidden_layers - 1):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_elman_model(neurons_per_layer, hidden_layers=1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu', input_shape=(None, 2)))
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.SimpleRNN(neurons_per_layer, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.Dense(1))    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_graph(x_train, x_test, z_train, z_test, z_predicted, label):
    plt.plot(x_train, z_train, 'o', label="Training value", markersize=1)
    plt.plot(x_test, z_test, label="True value")
    plt.plot(x_test, z_predicted, label="Predicted value")
    plt.title(label+"\nMSE: "+str(MSE(z_test, z_predicted)))
    plt.legend()
    plt.grid(True)
    plt.show()
    return

def MSE(true, pred):
    return np.mean((true - pred) ** 2)

x_train = np.concatenate((np.arange(3.4, 3.5, 0.001), np.arange(3.6, 3.7, 0.001)))
x_test = np.arange(3.5, 3.6, 0.001)

y_train, y_test  = get_y(x_train), get_y(x_test)
z_train, z_test = get_z(x_train, y_train), get_z(x_test, y_test)
train_values = np.vstack((x_train, y_train)).T
test_values = np.vstack((x_test, y_test)).T
train_values_elman = train_values.reshape(train_values.shape[0], 1, 2) 
test_values_elman = test_values.reshape(test_values.shape[0], 1, 2)

# Feed forward backprop, 1 внутрішній шар з 10 нейронами
model = create_feed_forward_model(10)
model.fit(train_values, z_train, epochs=500, verbose=0)
predicted_values = model.predict(test_values)
plot_graph(x_train, x_test, z_train, z_test, predicted_values, 
           "Feed forward backprop, 1 внутр. шар з 10 нейронами")

# Feed forward backprop, 1 внутрішній шар з 20 нейронами
model = create_feed_forward_model(20)
model.fit(train_values, z_train, epochs=500, verbose=0)
predicted_values = model.predict(test_values)
plot_graph(x_train, x_test, z_train, z_test, predicted_values, 
           "Feed forward backprop, 1 внутр. шар з 20 нейронами")

# Cascade - forward backprop, 1 внутрішній шар з 20 нейронами
model = create_cascade_forward_model(20)
model.fit(train_values, z_train, epochs=500, verbose=0)
predicted_values = model.predict(test_values)
plot_graph(x_train, x_test, z_train, z_test, predicted_values, 
           "Cascade - forward backprop, 1 внутр. шар з 20 нейронами")

# Cascade - forward backprop, 2 внутрішніх шари по 10 нейронів у кожному
model = create_cascade_forward_model(10, 2)
model.fit(train_values, z_train, epochs=500, verbose=0)
predicted_values = model.predict(test_values)
plot_graph(x_train, x_test, z_train, z_test, predicted_values, 
           "Cascade - forward backprop, 2 внутр. шари по 10 нейронів")

# Elman backprop, 1 внутрішній шар з 15 нейронами
model = create_elman_model(15)
model.fit(train_values_elman, z_train, epochs=500, verbose=0)
predicted_values = model.predict(test_values_elman)
plot_graph(x_train, x_test, z_train, z_test, predicted_values[:,0,:], 
           "Elman backprop, 1 внутр. шар з 15 нейронами")

# Elman backprop, 3 внутрішніх шари по 5 нейронів у кожному
model = create_elman_model(5, 3)
model.fit(train_values_elman, z_train, epochs=500, verbose=0)
predicted_values = model.predict(test_values_elman)
plot_graph(x_train, x_test, z_train, z_test, predicted_values[:,0,:], 
           "Elman backprop, 3 внутр. шари по 5 нейронів")