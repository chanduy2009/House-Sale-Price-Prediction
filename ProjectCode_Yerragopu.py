#================================================================================
#Project Title: Predicting the Sales Price of a House in Ames, Iowa
#Course: ISQS 6347 Data and Text Mining
#By: Chandu Yerragopu
#Under the Guidance of : Dr.Jaeki Song, Ph.D

#====================================STEP-1=======================================
#Importing and Looking at the data
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#visualize entries of the data
train.head()

#Show columns
train.columns

#Select a column
train['SalePrice'].head()

#Select rows
train[0:3]

#==================================STEP-2========================================

#Visualizing the data
import seaborn as sns
sns.distplot(train['SalePrice']);
plt.show()

#I am interested in looking for correlation between data points, to identify variables that can be good predictors for the Sales Price.

#We can face two different kind of variables: categorical and numerical. For the case of numerical values, we can create scatter plots. In the following example, we will plot Salesprice vs other variables.

#GrLivArea: Above grade (ground) living area square feet

train['SalePrice'].describe()
plt.plot(train['GrLivArea'],train['SalePrice'],'ro');
plt.show();

plt.plot(train['TotalBsmtSF'],train['SalePrice'],'ro');
plt.show();

plt.plot(train['GrLivArea'], train['SalePrice'], 'r--', train['TotalBsmtSF'], train['SalePrice'], 'bs')
plt.show()

#We can observe that for this two variables, there appears to be a correlation.

#For observing categorical values,I use boxplots, using the function boxplot from seaborn. For example, I am plotting the variable Overall Quality against Price.

#OverallQual: Rates the overall material and finish of the house

var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.show();

var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
plt.show();

var = 'BedroomAbvGr'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.show();

#=================================STEP-3===========================================
#Cleaning the Data

#Removing the Outliers

train['SalePrice'].describe()
plt.plot(train['GrLivArea'],train['SalePrice'],'ro');
plt.show();

#Removing Outliers
outliers = train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)]
train = pd.concat([train, outliers, outliers]).drop_duplicates(keep=False)
plt.plot(train['GrLivArea'],train['SalePrice'],'ro');
plt.show();

train = train.append(test)

#Imputing Missing Values for Basement Condition
train['BsmtCond'].replace("NA","No Basement",inplace = True)
train['BsmtCond'].describe()

var = 'BsmtCond'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.show();

# Handle missing values for features where median/mean or most common value doesn't make sense (Taken from Kaggle)

# Alley : data description says NA means "no alley access"
train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")
train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")
train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)
#Snippet From Kaggle

categorical_features = train.select_dtypes(include = ["object"]).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
train_num = train[numerical_features]
train_cat = train[categorical_features]

# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
train_num = train_num.fillna(train_num.median())
print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

len(train_num.columns)

#====================================STEP-4==========================================

#Model Building


train.head()

train_cat = pd.get_dummies(train_cat)

train_cat.columns.values

# Join categorical and numerical features
train = pd.concat([train_num, train_cat], axis = 1)

len(train.columns)

#Linear Regression

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Create linear regression object
regr = linear_model.LinearRegression()
test = train[1458:]
train = train[:1458]

X_train = train.drop("SalePrice",axis=1)
y_train = train['SalePrice']

from sklearn.model_selection import train_test_split

X_train_1, X_validation, y_train_1, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train the model using the training sets
regr.fit(X_train_1, y_train_1)

y_pred = regr.predict(X_validation)

#Calculate RMSE, using the function mean_squared_error

from sklearn.metrics import mean_squared_error
from math import sqrt
from math import log
import numpy as np
mse = mean_squared_error(y_validation, y_pred) 
sqrt(mse)

test = test.drop("SalePrice",axis=1)
test.head()
y_pred = regr.predict(test)

y_pred_df = pd.DataFrame(columns=['Id','SalePrice'])
y_pred_df['Id'] = test['Id'] 
y_pred_df['SalePrice'] =  [0 if i < 0 else i for i in y_pred]

y_pred_df.to_csv('solution_1.csv',index=False)

#Random Forest

from sklearn.ensemble import RandomForestRegressor

max_depth = 30
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(X_train_1, y_train_1)

# Predict on new data
y_pred = regr_rf.predict(X_validation)

mse = mean_squared_error(y_validation.apply(log), np.log(y_pred)) 
sqrt(mse)

from sklearn.ensemble import RandomForestRegressor
max_depth = 30
regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_pred = regr_rf.predict(test)

y_pred_df = pd.DataFrame(columns=['Id','SalePrice'])
y_pred_df['Id'] = test['Id'] 
y_pred_df['SalePrice'] =  y_pred

y_pred_df.to_csv('solution_1.csv',index=False)
