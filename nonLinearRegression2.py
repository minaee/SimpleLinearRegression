import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


df = pd.read_csv("china_gdp.csv")
# plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)

# msk = np.random.rand(len(df)) < 0.8
# train = df[msk]
# test = df[~msk]

# x_data_train, y_data_train = (train["Year"].values, train["Value"].values)
# x_data_test, y_data_test = (test["Year"].values, test["Value"].values)

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

msk = np.random.rand(len(df)) < 0.8
x_data_train = xdata[msk]
x_data_test = xdata[~msk]
y_data_train = ydata[msk]
y_data_test = ydata[~msk]


# plt.plot(x_data_test, y_data_test, 'ro')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()



def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

#logistic function
# Y_pred = sigmoid(x_data_train, beta_1 , beta_2)
# # plot initial prediction against datapoints
# plt.plot(x_data_train, Y_pred*15000000000000)
# plt.plot(x_data_train, y_data_train, 'ro')
# plt.show()





popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
# print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# x = np.linspace(1960, 2015, 55)
# x = x/max(x)
# plt.figure(figsize=(8,5))
# y = sigmoid(x, *popt)

y_pred = sigmoid(x_data_test, *popt)

print(y_pred)
print(y_data_test)
# plt.plot(xdata, ydata, 'ro', label='data')
# plt.plot(x,y, linewidth=3.0, label='fit')
# plt.legend(loc='best')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y_data_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y_data_test) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_pred , y_data_test) )