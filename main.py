import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math, time
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM

# Read CSV
# data = pd.read_csv("./csv/BTCtoUSD.csv")
data = pd.read_csv("./csv/ETHtoUSD.csv")
#data = pd.read_csv("./csv/LTCtoUSD.csv")

# Drop data
data = data.drop(['Date'],1)


#Normalize data
def normalize_data(df):
    min_max_scaler = preprocessing.MinMaxScaler()
    df['Price'] = min_max_scaler.fit_transform(df['Price'].values.reshape(-1,1))
    df['Open'] = min_max_scaler.fit_transform(df['Open'].values.reshape(-1,1))
    df['High'] = min_max_scaler.fit_transform(df['High'].values.reshape(-1,1))
    df['Change %'] = min_max_scaler.fit_transform(df['Change %'].values.reshape(-1,1))
    df['Low'] = min_max_scaler.fit_transform(df['Low'].values.reshape(-1,1))
    return df
df = normalize_data(data)

#Load data
def load_data(data, seq_len):
    amount_of_features = len(data.columns)
    data = data.as_matrix()  # pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []

    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length]) # index : index + 22days

    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split
    train = result[:int(row), :] # 90% date, all features

    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]

    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]

def build_model(layers):
    d = 0.3
    model = Sequential()

    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(256, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(32, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(1, kernel_initializer="uniform", activation='linear'))

    # adam = keras.optimizers.Adam(decay=0.2)

    start = time.time()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

#Train Model
window = 22
X_train, y_train, X_test, y_test = load_data(data, window)
print (X_train[0], y_train[0])

model = build_model([5,window,1])

model.fit(X_train,y_train,batch_size=512,epochs=90,validation_split=0.1,verbose=1)

# print(X_test[-1])
diff=[]
ratio=[]
p = model.predict(X_test)
print (p.shape)
# for each data index in test data
for u in range(len(y_test)):
    # pr = prediction day u
    pr = p[u][0]
    # (y_test day u / pr) - 1
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))
    # Last day prediction
    # print(p[-1])


# Bug fixed at here, please update the denormalize function to this one
def denormalize(df, normalized_value):
    df = df['Price'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)

    # return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new

data2 = pd.read_csv("./csv/ETHtoUSD.csv")

newp = denormalize(data2, p)
newy_test = denormalize(data2, y_test)

def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore[0], testScore[0]


model_score(model, X_train, y_train, X_test, y_test)

plt.plot(newp,color='red', label='Prediction')
plt.plot(newy_test,color='blue', label='Actual')
plt.legend(loc='best')
plt.show()