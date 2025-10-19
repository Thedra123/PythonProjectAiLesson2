import numpy
import pandas as pd

df=pd.DataFrame({
    'City':['Baku','Ganja','Sumgayit'],
    'Population':[100000,200000,300000],
})

# print(df)
#
# print(df.head(2))
# print(df.tail(2))
# print(df.sample())
# print(df.info())
# print(df.describe())



data={
    "Area_m2":[50,60,80,100,120,200],
    "Rooms":[1,2,2,3,3,5],
    "District":["Yasamal","Nizami","Nizami","Sebayil","Nerimanov","Sebayil"],
    "Price_AZN":[60000,75000,95000,120000,150000,500000]
}

houses=pd.DataFrame(data)
# print(houses)
# # print(houses[["Area_m2","Price_AZN"]])
# print(houses[houses['Rooms']>=3])
#
# print(houses.sort_values(by='Price_AZN',ascending=False))
#
# print(houses['District'].value_counts())

import numpy as np

prices = numpy.array([60000,75000,95000,120000,150000,500000])
# print("Mean : ", np.mean(prices))
# print("Median : ", np.median(prices))
from statistics import mode, variance
# print("Rooms", mode(houses['Rooms']))
# print("Variance", variance(houses['Price_AZN']))
q1 = houses['Price_AZN'].quantile(0.25)
q3 = houses['Price_AZN'].quantile(0.75)
print(q1)
print(q3)
iqr = q3 - q1
lover,upper = q1-1.5*iqr,q3+1.5*iqr
iqr_outliers = houses[(houses['Price_AZN'] < lover)] | (houses['Price_AZN'] > upper)
print(iqr_outliers)