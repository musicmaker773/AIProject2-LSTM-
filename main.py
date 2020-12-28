#import packages
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('COVID_DATA.csv')
r = csv.reader(open('COVID_DATA.csv'))
lines = list(r)


#print the head
df.head()

#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

n = 0

while n < 1 or n > 2:
    print("1 - Los Angeles")
    print("2 - Orange")
    num = input("Enter an input: ")
    n = int(num)



if n == 1:
    county = 'Los Angeles'
if n == 2:
    county = 'Orange'

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', county])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data[county][i] = data[county][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:102,:]
valid = dataset[102:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data


for b in range (0,10):
    temp = b + 103
    valid = dataset[102:temp,:]

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)


    train = new_data[:102]
    valid = new_data[102:temp]
    valid['Predictions'] = closing_price

    lines[temp][n] = valid['Predictions'][b]
    writer = csv.writer(open('COVID_DATA.csv','w',newline=''))
    writer.writerows(lines)

    


# print(valid[['Orange','Predictions']])
print(train[county])
print(valid[['Predictions']])

plt.title("COVID-19 Cases and Predictions")

plt.xticks(rotation=90)
plt.xlabel("Dates")
plt.ylabel("Number Of Cases")

plt.plot(train[county])
plt.plot(valid[['Predictions']])
plt.show()



