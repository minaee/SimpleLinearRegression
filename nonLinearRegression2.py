import numpy as np
import pandas as pd

df = pd.read_csv("china_gdp.csv")

x_data, y_data = (df["Year"].values, df["Value"].values)


def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y


beta_1 = 0.10
beta_2 = 1990.0

# logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

# plot initial prediction against datapoints
# plt.plot(x_data, Y_pred*15000000000000.)
# plt.plot(x_data, y_data, 'ro')
# plt.show()
# Lets normalize our data
xdata = x_data / max(x_data)
ydata = y_data / max(y_data)

from scipy.optimize import curve_fit

popt, pcov = curve_fit(sigmoid, xdata, ydata)
# print the final parameters
# print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
x = np.linspace(1960, 2015, 55)
x = x / max(x)
# plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
# plt.plot(xdata, ydata, 'ro', label='data')
# plt.plot(x,y, linewidth=3.0, label='fit')
# plt.legend(loc='best')
# plt.ylabel('GDP')
# plt.xlabel('Year')
# plt.show()
from sklearn.metrics import r2_score

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_, test_y))
