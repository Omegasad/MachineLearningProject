import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import  LSTM

"""
Used the github as a base
https://github.com/surajr/Stock-Predictor-using-LSTM/blob/master/Stock-Predictor-using-LSTM.ipynb
"""
# data = pd.read_csv("BTCtoUSD.csv",header=0,usecols=['Date','Price'],index_col=['Price'])

# Read CSV
# data = pd.read_csv("./csv/BTCtoUSD.csv")
data = pd.read_csv("./csv/ETHtoUSD.csv")
#data = pd.read_csv("./csv/LTCtoUSD.csv")

# Drop data
data = data.drop(['Date','Change %','Low'],1)
print(data.head())

#Load data
def load_data(data, seq_len):
    amount_of_features = len(data.columns)
    data = data.as_matrix()  # pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]

#Build model
def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

def build_model2(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model

# Divide as training and test set
# X,Y_Train being the training set
# X,Y_Test being the test set
window = 22
X_train, y_train, X_test, y_test = load_data(data[::-1], window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)

model = build_model2([3,window,1])

#Execute trained model
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=500,
    validation_split=0.1,
    verbose=1)

# Print scores
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))



# Prediction vs Real results
# Need to figure out what is prediction and how to plot it
plt.plot(y_train,color='red', label='prediction')

plt.plot(y_test,color='blue', label='y_test')
plt.legend(loc='upper left')
plt.show()
