import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv("VideoGamesSales.csv")
data.head()
data.columns
data.isnull().sum()
data = data.fillna(data.mean())
data["User_Score"]
type(data["Year_of_Release"][0])
data["Year_of_Release"] = data["Year_of_Release"].fillna(0).astype(int)
data["Year_of_Release"]
data["Year_of_Release"].mode()

data["Year_of_Release"]=data["Year_of_Release"].replace([0], 2008)
freq = data.groupby(["Year_of_Release"]).count() 
freq
plt.figure(figsize=(16, 8))
plt.scatter(
    data['Global_Sales'],
    data['NA_Sales'],
    c='black'
)
plt.xlabel("global sales")
plt.ylabel("Ventas ($)")
plt.show()
X = data['Global_Sales'].values.reshape(-1,1)
y = data['NA_Sales'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)
print(reg.coef_[0][0])
print(reg.intercept_[0])

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(X)

plt.figure(figsize=(16, 8))
plt.scatter(
    data['Global_Sales'],
    data['NA_Sales'],
    c='black'
)
plt.plot(
    data['Global_Sales'],
    predictions,
    c='blue',
    linewidth=2
)
plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")
plt.show()

X = data['Global_Sales']
y = data['NA_Sales']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

'tbd' in data['User_Score'].values 
Xs = data.drop(['NA_Sales','Name','Platform','Genre','Publisher', 'Developer','Rating','User_Score'], axis=1)
y = data['NA_Sales']#.reshape(-1,1)

reg = LinearRegression()
reg.fit(Xs, y)
print(reg.coef_)
print(reg.intercept_)



X = np.column_stack((data['EU_Sales'], data['Global_Sales'], data['Critic_Score']))
y = data['NA_Sales']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
X_inter = interaction.fit_transform(X)
X2 = sm.add_constant(X_inter)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())