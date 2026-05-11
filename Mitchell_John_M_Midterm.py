# Author: John Mitchell
# Exam 1: Babies Data Analysis
# Fall 2025

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

# 1. Read the dataset
dat = pd.read_csv("babies.csv")

# 2. Reveal the number of rows and columns
print(f"Rows: {dat.shape[0]}, Columns: {dat.shape[1]}")

# 3. Reveal the number of missing values per variable
print("\nMissing values per variable")
missing_values = dat.isna().sum()
print(missing_values)

# 4. Drop the missing values
dat = dat.dropna()
print(f"\nAfter dropping: Rows: {dat.shape[0]}, Columns: {dat.shape[1]}")

# 5. Convert smoke into an integer variable
dat['smoke'] = dat['smoke'].astype(int)

# 6. Create a correlation matrix
print("\nCorrelation Matrix")
corr_matrix = dat[['bwt', 'gestation', 'parity', 'age', 'height', 'weight', 'smoke']].corr()
print(corr_matrix.round(2))

# 7. Perform 80/20% random split with seed = 123
print("\nTrain/Test Split")
rng = np.random.default_rng(123)
train_index = rng.random(len(dat)) < 0.8
train = dat[train_index]
test = dat[~train_index]
print(f"Training set: {len(train)} rows")
print(f"Testing set: {len(test)} rows\n")

# 8. Create linear regression model
X_train = train[['gestation', 'parity', 'age', 'height', 'weight', 'smoke']]
X_train = sm.add_constant(X_train)
y_train = train['bwt']
model = sm.OLS(y_train, X_train).fit()

# 9. Compute residual standard error on testing set
X_test = test[['gestation', 'parity', 'age', 'height', 'weight', 'smoke']]
X_test = sm.add_constant(X_test)
y_test = test['bwt']
y_pred = model.predict(X_test)

residuals = y_test - y_pred
rss = (residuals**2).sum()
n_test = len(y_test)
p = X_test.shape[1] - 1
rse = (rss / (n_test - p - 1))**0.5
print(f"Residual Standard Error: {rse:.2f}")

# Question 1: Correlation between gestation and birth weight
corr_gestation = np.corrcoef(dat['bwt'], dat['gestation'])[0, 1]
print(f"\n1. Correlation coefficient between gestation and birth weight: {corr_gestation:.2f}")

# Questions 2 & 3: P-value for age and significance
model_summary = model.summary2().tables[1]
age_pvalue = model_summary.loc['age', 'P>|t|']
age_significant = 'y' if age_pvalue < 0.05 else 'n'
print(f"2. P-value for age: {age_pvalue:.2f}")
print(f"3. Is age a significant predictor? (y/n): {age_significant}")

# Question 4: Proportion of variability explained (R-squared)
rsquared = model.rsquared
print(f"4. Proportion of variability explained by the model: {rsquared:.2f}")

# Question 5: Slope for gestation
gestation_slope = model.params['gestation']
print(f"5. Slope for gestation: {gestation_slope:+.2f}")

# Question 6: Confidence interval bounds for smoke
conf_int = model.conf_int(alpha=0.05)
smoke_lower = conf_int.loc['smoke', 0]
smoke_upper = conf_int.loc['smoke', 1]
print(f"6. Bounds for the slope of smoke: [{smoke_lower:+.2f}, {smoke_upper:+.2f}]")

# Question 7: Residual Standard Error (already computed)
print(f"7. Residual Standard Error on test set: {rse:.2f}")
