EDA Cheat Sheet — Personal Reference (Mansi Ingle)

Master EDA commands for Data Science, ML, Analytics, BI — to analyze, clean, and explore any dataset.

---

Import Libraries

import numpy as np  
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt  
import scipy.stats as stats  
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

---

Load Data

df = pd.read_csv('your_data.csv')  
df.head()  
df.tail()  
df.sample(5)

---

Basic Info

df.shape  
df.columns  
df.dtypes  
df.info()  
df.describe()  
df.isnull().sum()  
df.duplicated().sum()

---

Missing Values

df.isnull().sum()/len(df)*100  
df.fillna(value)  
df.dropna()

---

Data Cleaning

df['column'].str.strip()  
df['column'] = df['column'].replace('-', np.nan)  
df.drop_duplicates(inplace=True)  
df.rename(columns={'old': 'new'}, inplace=True)

---

Univariate Analysis

df['column'].value_counts()  
df['column'].value_counts(normalize=True)  
df['column'].unique()  
df['column'].nunique()

---

Visual Univariate

sns.histplot(df['column'])  
sns.boxplot(x=df['column'])  
sns.countplot(x=df['categorical_column'])  
sns.violinplot(x=df['column'])  
df['column'].plot(kind='kde')

---

Bivariate / Multivariate

sns.scatterplot(x='feature1', y='feature2', data=df)  
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')  
sns.pairplot(df, diag_kind='kde')  
sns.barplot(x='category', y='numeric', data=df)

---

Outliers

sns.boxplot(x=df['column'])

Q1 = df['column'].quantile(0.25)  
Q3 = df['column'].quantile(0.75)  
IQR = Q3 - Q1  
lower = Q1 - 1.5 * IQR  
upper = Q3 + 1.5 * IQR  
df_outliers = df[(df['column'] < lower) | (df['column'] > upper)]

---

Skewness & Transformation

df['column'].skew()  
df['column'].kurt()  
df['new_column'] = np.log1p(df['column'])  
df['new_column'] = np.sqrt(df['column'])

---

Categorical Encoding

pd.get_dummies(df['category_column'], drop_first=True)  

le = LabelEncoder()  
df['encoded'] = le.fit_transform(df['category_column'])

---

Feature Scaling

scaler = StandardScaler()  
df_scaled = scaler.fit_transform(df[['col1', 'col2']])

scaler = MinMaxScaler()  
df_scaled = scaler.fit_transform(df[['col1', 'col2']])

---

Time Series EDA

df['date'] = pd.to_datetime(df['date'])  
df.set_index('date', inplace=True)  
df.resample('M').mean().plot()

---

Correlation

df.corr()  
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

---

Groupby & Aggregation

df.groupby('category_column')['numeric_column'].mean()  

df.pivot_table(values='numeric', index='category1', columns='category2', aggfunc='mean')

---

Save / Export

df.to_csv('cleaned_data.csv', index=False)

---

Other Useful Pandas Commands

df.sort_values(by='column', ascending=False)  
df['new_col'] = df['col1'] / df['col2']  
df['year'] = df['date'].dt.year  
df['month'] = df['date'].dt.month

---

Practice these regularly — this is your complete EDA core toolkit!
