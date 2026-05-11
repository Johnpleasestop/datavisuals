# Author: John Mitchell
# Week 4: Linear Regression
# Plots : Seaborn
# Fall 2025

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# Load the dataset
dat = pd.read_csv("penguins.csv")
dat.shape

dat.head()

# Prepare the data
dat.isna().sum()

dat = dat.dropna()

dat.shape

# Explore the contents of the dataset
# How many species are there?

dat.species.value_counts()
dat.species.unique()

dat.columns

# Relationship between 'bill_length_mm', 'bill_depth_mm'
sns.scatterplot(data=dat, x = 'bill_length_mm', y = 'bill_depth_mm')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.title('Bill Length vs Bill Depth')
plt.show()

# Correlation
stats.pearsonr(dat.bill_length_mm, dat.bill_depth_mm)

dat.loc[:, ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].corr()

# Regression model
X = dat.bill_length_mm
X = sm.add_constant(X) # Adds a constant term to the redictor
y = dat.bill_depth_mm

model_1 = sm.OLS(y, X).fit()
model_1.summary()

y_hat_1 = model_1.predict(X)

# Plot regression line
# Relationship between 'bill_length_mm' and 'bill_depth_mm'
sns.scatterplot(data=dat, x = 'bill_length_mm', y = 'bill_depth_mm')
plt.plot(dat.bill_length_mm, y_hat_1, color='red')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.title('Bill Length vs Bill Depth')
plt.show()

#================================================================================================================

# Relationship between 'bill_length_mm' and 'bill_depth_mm'
sns.scatterplot(data=dat, x = 'bill_length_mm', y = 'bill_depth_mm', hue = "species")
plt.plot(dat.bill_length_mm, y_hat_1, color='red')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.title('Bill Length vs Bill Depth')
plt.show()

# Create a linear model
X = dat[['bill_length_mm', 'species']]
X = pd.get_dummies(X, drop_first=True)
X.head()
X.species_Chinstrap = X.species_Chinstrap.astype(int)
X.species_Gentoo = X.species_Gentoo.astype(int)
X.head()
X = sm.add_constant(X) # Adds a constant term to the predictor
y = dat.bill_depth_mm

model_2 = sm.OLS(y, X).fit()
model_2.summary()

y_hat_2 = model_2.predict(X)

# Plot regression line
# Relationship between 'bill_length_mm' and 'bill_depth_mm'
sns.scatterplot(data=dat, x = 'bill_length_mm', y = 'bill_depth_mm', hue = "species")
plt.plot(dat.bill_length_mm, y_hat_1, color='red', alpha=0.5)
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.title('Bill Length vs Bill Depth')
plt.show()

#================================================================================================================

