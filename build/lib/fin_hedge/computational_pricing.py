import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


""" LONG-SHORT TERM MEMORY NN"""
def lstm(df):
    # Convert data to numpy array
    dataset = df.values
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    # Split data
    training_data_len = int(np.ceil(len(dataset) * .8))
    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []
    # Train the data
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    # Convert x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    # Reshape the data to 3D for LSTM model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences= False))
    model.add(Dense(25))
    model.add(Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    # Create testing dataset
    test_data = scaled_data[training_data_len - 60: , :]
    # Create datasets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    # Convert x_test to a numpy array and reshape it to 3D
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # Get predicted scaled price
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    # Get RMSE (root mean squared error)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    print('Root Mean Squared Error: ', rmse)
    return prediction


if __name__ == '__main__':
    # Download historical prices
    df = yf.download('AAPL', start='2012-01-01', end='2023-06-07')
    df = df.filter(['Close'])

    # LSTM
    prediction = lstm(df)