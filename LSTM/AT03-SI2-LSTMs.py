import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense, LSTM

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()

# For time stamps
from datetime import datetime

#J&J, NVIDIA, COSTCO WHOLESALE, EXXOL MOBIL
tech_dict = {'COST': (0,0), 'JNJ': (0,1), 'NVDA': (1,0), 'XOM': (1,1)}

fig, axs = plt.subplots(2, 2)

pd.options.mode.chained_assignment = None  # Desabilita o aviso

for ativo in tech_dict.keys():
    df = pdr.get_data_yahoo(ativo, start='2012-01-01', end=datetime.now())

    #baseline = média móvel
    baseline = df['Adj Close'].rolling(10).mean()

    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rms = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"RMS ({ativo}): {rms}")
    print(f"MAE ({ativo}): {mae}")

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid.loc[:,"Predictions"] = predictions
    # Visualize the data
    axs[tech_dict[ativo]].plot(baseline)
    axs[tech_dict[ativo]].plot(train['Close'])
    axs[tech_dict[ativo]].plot(valid[['Close', 'Predictions']])
    axs[tech_dict[ativo]].set_title(ativo)
    axs[tech_dict[ativo]].set(xlabel='Data', ylabel='Preço estimado (USD)')
    axs[tech_dict[ativo]].legend(['Train', 'Val', 'Predictions'], loc='upper left')
    #plt.figure(figsize=(16,6))
    #plt.title('Model')
    #plt.xlabel('Date', fontsize=18)
    #plt.ylabel('Close Price USD ($)', fontsize=18)
    #plt.plot(train['Close'])
    #plt.plot(valid[['Close', 'Predictions']])
    #plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

plt.show()