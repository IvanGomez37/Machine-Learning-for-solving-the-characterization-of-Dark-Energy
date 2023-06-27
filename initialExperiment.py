
import pandas
import numpy
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split # Splitting data to training and test data
from sklearn.linear_model import ( # Linear regression model regressors
    LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor,
)
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# To use a different data set just change the txt file name to any of: hz_com, hz_da, hz_dl
data = pandas.read_csv('DataBases\hz_dl.txt', sep = ' ', comment = '#') # Reading from a txt file, each column separated by a single blank space
del data[data.columns[-1]] # Deleting last column (trash column)

print(data)

# Data splited by 90% training and 10% testing. To change percentage modify test_size parameter
xTrain, xTest, yTrain, yTest = train_test_split(data['x'], data['y'], test_size = 0.1, random_state = 2) 

print(xTrain.shape, yTrain.shape)
print(xTest.shape, yTest.shape)

regLin = LinearRegression()
regLin.fit(xTrain.to_numpy().reshape(-1, 1), yTrain)

regTS = TheilSenRegressor()
regTS.fit(xTrain.to_numpy().reshape(-1, 1), yTrain)

regRan = RANSACRegressor()
regRan.fit(xTrain.to_numpy().reshape(-1, 1), yTrain)

regHub = HuberRegressor()
regHub.fit(xTrain.to_numpy().reshape(-1, 1), yTrain)

yPredLin = regLin.predict(xTest.to_numpy().reshape(-1, 1))

yPredTS = regTS.predict(xTest.to_numpy().reshape(-1, 1))

yPredRan = regRan.predict(xTest.to_numpy().reshape(-1, 1))

yPredHub = regHub.predict(xTest.to_numpy().reshape(-1, 1))

print(r2_score(yPredLin, yTest))

print(r2_score(yPredTS, yTest))

print(r2_score(yPredRan, yTest))

print(r2_score(yPredHub, yTest))

plt.scatter(xTrain, yTrain, color = "black", alpha = 0.2, label = "Training data")
plt.scatter(xTest, yTest, color = "red", alpha = 0.5, label = "Expected testing results")

plt.plot(xTest, yPredLin, color = "orange", linestyle = "dashed", alpha = 0.7, linewidth = 1, label = 'Linear')
plt.plot(xTest, yPredTS, color = "green", linestyle = "dashed", alpha = 0.7, linewidth = 1, label = 'Theil Sen')
plt.plot(xTest, yPredRan, color = "purple", linestyle = "dashed", alpha = 0.7, linewidth = 1, label = 'RANSAC')
plt.plot(xTest, yPredHub, color = "olive", linestyle = "dashed", alpha = 0.7, linewidth = 1, label = 'Huber')

plt.legend()
plt.title("Linear Regression models")
plt.xlabel("Redshift")
plt.ylabel("Hubble's parameter")

plt.show()

polyFeatures = PolynomialFeatures(degree = 6, include_bias = False) # To change degree of polynomial function change degree parameter

xPolyTrain = polyFeatures.fit_transform(xTrain.to_numpy().reshape(-1,1)) # xTrain data polynomial transformation
xPolyTest = polyFeatures.fit_transform(xTest.to_numpy().reshape(-1,1)) # xTest data polynomial transformation

regPoly = LinearRegression()
regPoly.fit(xPolyTrain, yTrain)

yPredPoly = regPoly.predict(xPolyTest)

print(r2_score(yPredPoly, yTest))
print(regPoly.coef_)

plt.scatter(xTrain, yTrain, color = "black", alpha = 0.2, label = "Training data")
plt.scatter(xTest, yTest, color = "red", alpha = 0.5, label = "Expected testing data")

# Predicted data transformation
polyRes = pandas.DataFrame()
polyRes['x'] = xTest
polyRes['y'] = yPredPoly
polyRes = polyRes.sort_values(by = ['x'])

plt.plot(polyRes['x'], polyRes['y'], color = "Blue", linestyle = "dashed", alpha = 0.7, linewidth = 1, label = "Polynomial")

plt.legend()
plt.title("Polynomial 7 - hz_dl")
plt.xlabel("Redshift")
plt.ylabel("Hubble's parameter")

plt.show()
