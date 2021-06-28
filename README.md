# Stock market price forecast
The name of this project is pretty self explanatory so i won't go in details on that.
### LSTM architecture
I used Long Short-term memory artificial recurrent neural network (RNN) architecture in this  project.
LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series.
# Coding
### Imports

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas_datareader as web
    import math
    from keras.models import Sequential
    from tensorflow.keras import layers
    from sklearn.preprocessing import MinMaxScaler
### Data gathering
For this example we are going to predict prices of Crude Oil

    day = int(input('Какой Сегодня день?', ))
    month = int(input('Какого месяца?', ))
    year = int(input('Какого года?', ))
    today = f'{day}-{month}-{year}'
    print('Сегодня ', today)
    
    data = web.DataReader('CL=F', data_source='yahoo', start='1980-01-01', end=today)
    
### Preprocessing and train data creation

    df = data.filter(['Close'])
    df_val = df.values
    train_len = math.ceil(len(df)*0.8)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_val)
    
    train_data = scaled_data[:train_len, :]

    X_train=[]
    y_train=[]

    for i in range(60, len(train_data)):
      X_train.append(train_data[i-60:i, 0])
      y_train.append(train_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)    
    X_train = np.expand_dims(X_train, 1)
### Model    
    
    model = Sequential([
        layers.Dense(16, activation='relu', input_shape=(X_train.shape[1:])),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='elu'),
        layers.Dense(1)
        ])
        
     model.compile(optimizer='adam', loss='mse')
### Training and creating test data
    model.fit(X_train, y_train, batch_size=128, epochs=100)
    
    
    test_data = scaled_data[train_len - 60: , :]

    X_test = []
    y_test = df_val[train_len:, :]
    for i in range(60, len(test_data)):
        X_test.append(test_data[i-60:i, 0])
        
    X_test = np.array(X_test)
    X_test = np.expand_dims(X_test, 1)
     
    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
### Metrics
    train = data[:train_len]
    valid = data[train_len:]
    valid['Predictions'] = pred
    valid
    fig = plt.figure()
    plt.xlabel('Дата', fontsize = 18)
    plt.ylabel('USD', fontsize = 18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.grid(True)
 ![Figure 2](https://user-images.githubusercontent.com/82718776/123694476-0cea2c00-d862-11eb-8d4a-f94677e89e1d.png)
    
    def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
    A = valid['Close']
    F = valid['Predictions']
    print('sMAPE = ',smape(A, F))
sMAPE = 3.21
    
