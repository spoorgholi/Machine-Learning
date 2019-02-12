#simple linear regression

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import datasets

#importing the dataset
data = datasets.load_boston()
x = data.data[:, 5:6]

#nd array ro 1d array
x = x.ravel()
y = data.target

#np.array to pd.datafram
df = pd.DataFrame({'RM': x, 'Price': y })

#print a description of the data
print('\n Stats for the price is \n ', stats.describe(y))
price_mean = y.mean()

#remove crazy data
for rows in range(0, len(df)):
    if (df['Price'][rows] > (1.9* price_mean) or df['Price'][rows] < (0.1 * price_mean)) :
        print('\n Deleted lines are : ', rows)
        df.drop(rows, inplace = True)

#plotting the dataset for more info
plt.figure(1)
plt.scatter(df['RM'], df['Price'] ,color = 'red', marker = '.')
plt.figure(2)
sns.distplot(df['Price'], color = 'green', bins = 100)
plt.show()

x = df['RM'].values
y = df['Price'].values

#simple linear eregressionfunction
def linear_regression(df, x, y):

    x_hat = x.mean()
    y_hat = y.mean()
    
    naminator = 0
    denaminator = 0
    
    for rows in range(0, len(df)):
        naminator += (x[rows] - x_hat) * (y[rows]- y_hat)
        
    for rows in range(0, len(df)):
        denaminator += ((x[rows] - x_hat)** 2)
    
    b1 = naminator / denaminator
    b0 = y_hat - b1* x_hat
    
    return b0, b1

b0, b1 = linear_regression(df, x, y)

#plotting the refressor
plt.figure(3)
plt.scatter(x, y, color = 'red', marker = '.', label = 'Housing Data')
plt.plot(x, b0 + b1 * x, color = 'blue', label = 'Regression line')
plt.xlabel('RM')
plt.ylabel('Price')
plt.legend()
plt.show()
