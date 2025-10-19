import numpy as np
import pandas as pd

df = pd.DataFrame({
    'City': ['Baku', 'Ganja', 'Sumgayit'],
    'Population': [2000, 6000, 122000],
})

df["Population"] = pd.to_numeric(df["Population"], errors="coerce").astype("int64")

print(df)

print(df.head(2))

print(df.tail(1))

print(df.sample())

print(df.info())

print(df.describe())

print(df)

df["Density_guess"]=df["Population"]/100
print(df)


data={
    "Area_m2":[50,60,80,100,120,200],
    "Rooms":[1,2,2,3,3,5],
    "District":["Yasamal", "Nizami", "Nizami","Sebrayil","Nerimanov","Sebrayil"],
    "Price_Azn":[60000,75000,95000,120000,150000,500000]
}

houses = pd.DataFrame(data)
print(houses)

print(houses[['Area_m2', 'Price_Azn']])
print(houses['Rooms']>=3)
print(houses.sort_values(by='Price_Azn', ascending=False))
print(houses["District"].value_counts())



prices = np.array([60000,75000,95000,120000,150000,500000])

print("Mean : ", np.mean(prices))
print("Median : ", np.median(prices))


from statistics import mode
print("Mode : ", mode(houses["Rooms"]))
print("Variance : ", houses["Price_Azn"].var())

a=np.random.randint(1,10,size=[3,4])
print(a)
print(a.shape)
print(a.T)
print(a[0,1])
print(a[:,2])
print(a[1:3,1:3])

b = np.array([1,2,3,4,5])
print(b+5)
print(b*5)
print(b**2)


normal=np.random.normal(0,5,20)
uniform=np.random.uniform(0,5,20)

print(normal)
print(uniform)


import pandas as pd

s=pd.Series([5,10,15,20], index=['A', 'B', 'C', 'D'])
print(s)
print(s.mean(), s.median())

houses=pd.read_excel("houses_day1.xlsx")
print(houses.head(5))
print(houses.shape)
print(houses.columns)

houses["Price_per_m2"]=houses['Price_AZN'].astype(float)/houses["Area_m2"]
houses.to_excel("houses_day1.xlsx", index=False)

houses['Price_AZN'].fillna(houses['Price_AZN'].median(), inplace=True)
houses.to_excel("houses_day1.xlsx", index=False)

print("Mean : ", houses['Price_AZN'].mean())
print("Median : ", houses['Price_AZN'].median())
print("Mode : ", houses['Price_AZN'].mode())
print("======================================================")
print(houses[["Area_m2", "Price_AZN"]].cov())
print(houses[["Area_m2", "Price_AZN"]].corr())

import  matplotlib.pyplot as plt
plt.hist(houses["Price_AZN"], bins=20)
plt.title("Histogram of Price AZN")
plt.xlabel("Price AZN")
plt.ylabel("Count")
plt.show()

import seaborn as sns

# sns.heatmap(houses.corr(numeric_only=True), annot=True, cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.show()
#
#
# by_district = houses.groupby(["District", "Price_AZN"]).mean().sort_values(ascending=False)

# import seaborn as sns
# import matplotlib.pyplot as plt
sns.lmplot(data=houses,x="Area_m2", y="Price_AZN", line_kws={"color": "black"})
plt.title("Mean Price AZN")
plt.show()