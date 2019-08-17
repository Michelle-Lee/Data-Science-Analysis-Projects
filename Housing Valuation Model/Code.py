# House Prices: Advanced Regression Techniques

# data analysis and wrangling
import numpy as np
import pandas as pd
import random as rnd
from scipy.stats.stats import pearsonr
from scipy.stats.stats import linregress
from sklearn.metrics import mean_squared_error

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import statsmodels.formula.api as sm

train = pd.read_csv('/Users/michelle.mpl/Desktop/Kaggle_HousePrices/train.csv', engine= 'python')
test_df = pd.read_csv('/Users/michelle.mpl/Desktop/Kaggle_HousePrices/test.csv', engine = 'python')
#combine = [train_df,test_df]

train_df, cv_df = train_test_split(train, test_size=.40)

# Categorical, numerical or alphanumeric?
train_df.info()

# Sale Price Summary/Distribution
'''
SalePrice is skewed to the right, so the mean > median.
Mean: 180921
Median: 163000
'''
print(train_df['SalePrice'].describe())
plt.figure(1)
sns.distplot(train_df['SalePrice'])
plt.show()

plt.figure(0)
sns.distplot(np.log(train_df['SalePrice']))
plt.show()

#################### Missing/null Numeric Values ####################
'''
Missing Values:
    LotFrontage
    MasVnrArea
    GarageYrBuilt
Inspect if these variables are worth trying to impute, i.e. is there 
a strong relationship with SalePrice? If not, drop. 
GarageYrBlt and SalePrice has correlation, however, GarageYrBlt is 
multi-collinear with YearBlt (rvalue=0.825667), thus we will drop it anyway.
'''
filterLF = train_df[train_df['LotFrontage'].notnull()][['LotFrontage','SalePrice']]
plt.figure(2)
sns.regplot('LotFrontage','SalePrice',data = filterLF)
linregress(filterLF['LotFrontage'],filterLF['SalePrice'])
plt.show()
filterMV = train_df[train_df['MasVnrArea'].notnull()][['MasVnrArea','SalePrice']]
plt.figure(3)
sns.regplot('MasVnrArea','SalePrice',data = filterMV)
linregress(filterMV['MasVnrArea'],filterMV['SalePrice'])
plt.show()
filterGB = train_df[train_df['GarageYrBlt'].notnull()][['GarageYrBlt','SalePrice','YearBuilt']]
plt.figure(4)
sns.regplot('GarageYrBlt','SalePrice',data = filterGB)
linregress(filterGB['GarageYrBlt'],filterGB['SalePrice'])
plt.show()

train_df.drop(['LotFrontage','MasVnrArea','GarageYrBlt'],axis=1, inplace=True)

# Rename all variables that start with anumber
train_df.rename(columns = {'1stFlrSF':'FirstFlrSF',
                            '2ndFlrSF':'SecFlrSF',
                            '3SsnPorch':'ThreeSsnPorch'}, inplace=True)

#################### Missing/Null Categorical Variables ####################
# Cross-reference with readme file to check if they are intentional NAs
'''
All variables with NAs were intentional besides 'Electrical' which has 1 NA.
We can replace this NA value with the mode, 'SBrkr'.
Now that we know the rest of the NAs are intentional, we can replace all of
them with 0 AFTER we finish mapping categorical variables to new numerical variables.
'''
print(train_df.select_dtypes(include=['object']).isnull().sum())
print(train_df['Electrical'].value_counts())
train_df['Electrical'].fillna('SBrkr',inplace=True)

# Replace all the NAs with 0 (because these NAs are intentional)
train_df.fillna(0, inplace=True)

#################### Feature Engineer and Convert Categorical to Numerical ####################
# Binary
train_df['isStreetPave'] = (train_df['Street'] == 'Pave').astype(int)
train_df['isAlleyPave'] = (train_df['Alley'] == 'Pave').astype(int)
train_df['isCentralAC'] = (train_df['CentralAir'] == 'Y').astype(int)
train_df['isDrivePaved'] = (train_df['PavedDrive'] =='Y').astype(int)
train_df['isCondNormal'] = (train_df['Condition1'] == 'Norm').astype(int)
train_df['is1Fam_TwnhsE'] = ((train_df['BldgType'] == '1Fam') | (train_df['BldgType'] == 'TwnhsE')).astype(int)
train_df['isCDS_FR3'] = ((train_df['LotConfig'] == 'CulDSac') | (train_df['LotConfig'] == 'FR3')).astype(int)
train_df['isHip_Shed'] = ((train_df['RoofStyle'] == 'Hip') | (train_df['LotConfig'] == 'Shed')).astype(int)
train_df['isStdCompShg'] = (train_df['RoofMatl'] == 'CompShg').astype(int)
train_df['isBrk_Stone'] = ((train_df['MasVnrType'] == 'BrkFace') | (train_df['MasVnrType'] == 'Stone')).astype(int)
train_df['isFoundConcrete'] = (train_df['Foundation'] == 'PConc').astype(int)
train_df['isStdHeating'] = (train_df['Heating'] == 'GasA').astype(int)
train_df['isStdElectrical'] = (train_df['Electrical'] == 'SBrkr').astype(int)
train_df['isNew_RegContract'] = ((train_df['SaleType'] == 'New') | (train_df['SaleType'] == 'Con')).astype(int)

# Ordinal (assume scale from none/worst to best)
#   Note: Unintentional NAs have been dealth with above, the rest of features have intentional NA values 
#    to represent 'None' (e.g. no basement); these have been converted to 0 below to represent the 
#    lowest value of the scales mapped below.(.map() function does not map NAs anyway)
train_df['Zone'] = train_df['MSZoning'].map({"C (all)":1, "RH":2, "RM":3, "RL":4, "FV":5})
train_df['irregularLot_Scale'] = train_df['LotShape'].map({"Reg":0, "IR1":1, "IR2":2, "IR3":3})
train_df['landContour_Scale'] = train_df['LandContour'].map({"Lvl":0, "Low":1, "HILS":2, "BNK":3})
train_df['Util'] = train_df['Utilities'].map({"ELO":1, "NoSeWa":2, "NoSewr":3, "AllPub":4})
train_df['landSlope_Scale'] = train_df['LandSlope'].map({"Gtl": 1, "Mod":2, "Sev":3})
train_df['exterQual_Scale'] = train_df['ExterQual'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
train_df['exterCond_Scale'] = train_df['ExterCond'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
train_df['BsmtHeight_Scale'] = train_df['BsmtQual'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
train_df['BsmtCond_Scale'] = train_df['BsmtCond'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
train_df['BsmtExposure_Scale'] = train_df['BsmtExposure'].map({"No":1, "Mn":2, "Av":3, "Gd":4})
train_df['BsmtFinType1_Scale'] = train_df['BsmtFinType1'].map({"Unf":1, "LwQ":2, "Rec":3, "BLQ": 4, "ALQ":5, "GLQ":6})
train_df['BsmtFinType2_Scale'] = train_df['BsmtFinType2'].map({"Unf":1, "LwQ":2, "Rec":3, "BLQ": 4, "ALQ":5, "GLQ":6})
train_df['HeatingQC_Scale'] = train_df['HeatingQC'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})  
train_df['KitchenQual_Scale'] = train_df['KitchenQual'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}) 
train_df['FuncDamage_Scale'] = train_df['Functional'].map({"Typ":0, "Min1":1, "Min2":2, "Mod":3, "Maj1": 4, "Maj2":5, "Sev":6,"Sal":7})
train_df['FireplaceQu_Scale'] = train_df['FireplaceQu'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
train_df['GarageQual_Scale'] = train_df['GarageQual'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
train_df['GarageCond_Scale'] = train_df['GarageCond'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
train_df['GarageFinish_Scale'] = train_df['GarageFinish'].map({"Unf": 1, "RFn":2, "Fn":3})
train_df['PoolQC_Scale'] = train_df['PoolQC'].map({"Fa":1, "TA":2, "Gd":3, "Ex":4})
train_df['FenceQual_Scale'] = train_df['Fence'].map({"MnWw":1, "GdWo":2, "MnPrv":3, "GdPrv":4})


'''
In the scatterplot below, we see that most of the neighborhoods fall into a SalePrice clusters.
This is a good indicator that we can bucket each neighborhood into price scale from 1 to 4 based 
on the mean SalePrice of that neighborhood.

'''
# Neighborhood
plt.figure(5)
sns.stripplot(x='Neighborhood',y='SalePrice', data=train_df, jitter=True)
plt.xticks(rotation=45)
#plt.show()

nbhd_piv = pd.pivot_table(train_df,index=['Neighborhood'],values=["SalePrice"],aggfunc=np.mean).reset_index()      # Pivot Table for mean SalePrice of each Neighborhood
nbhd_piv.rename(columns = {'SalePrice':'Nbhd_Mean'}, inplace=True)
train_df = pd.merge(left=train_df, right=nbhd_piv,how='left', on='Neighborhood')                                   # Left Merge the mean SalePrice with the respective Neighborhoods

print(train_df['SalePrice'].describe())
bins = [0, 130000, 163000, 214000, 1000000]
binLabels = [1, 2, 3, 4]
train_df['Nbhd_Value'] = pd.cut(train_df['Nbhd_Mean'], bins, labels=binLabels).astype(int)

# GarageType
gt_piv = pd.pivot_table(train_df,index=['GarageType'],values=["SalePrice"],aggfunc=np.mean).reset_index()
gt_piv.rename(columns = {'SalePrice':'GT_Mean'}, inplace=True)
train_df = pd.merge(left=train_df, right=gt_piv,how='left', on='GarageType')
train_df['GarageType_Value'] = pd.cut(train_df['GT_Mean'], bins, labels=binLabels).astype(int)

# HouseStyle
hs_piv = pd.pivot_table(train_df,index=['HouseStyle'],values=["SalePrice"],aggfunc=np.mean).reset_index()
hs_piv.rename(columns = {'SalePrice':'HS_Mean'}, inplace=True)
train_df = pd.merge(left=train_df, right=hs_piv,how='left', on='HouseStyle')
train_df['HouseStyle_Value'] = pd.cut(train_df['HS_Mean'], bins, labels=binLabels).astype(int)

# Exterior1st
ex1_piv = pd.pivot_table(train_df,index=['Exterior1st'],values=["SalePrice"],aggfunc=np.mean).reset_index()
ex1_piv.rename(columns = {'SalePrice':'Ex1_Mean'}, inplace=True)
train_df = pd.merge(left=train_df, right=ex1_piv,how='left', on='Exterior1st')
train_df['Exterior1st_Value'] = pd.cut(train_df['Ex1_Mean'], bins, labels=binLabels).astype(int)

# Exterior2nd
ex2_piv = pd.pivot_table(train_df,index=['Exterior2nd'],values=["SalePrice"],aggfunc=np.mean).reset_index()
ex2_piv.rename(columns = {'SalePrice':'Ex2_Mean'}, inplace=True)
train_df = pd.merge(left=train_df, right=ex2_piv,how='left', on='Exterior2nd')
train_df['Exterior2nd_Value'] = pd.cut(train_df['Ex2_Mean'], bins, labels=binLabels).astype(int)

# SaleCondition
sc_piv = pd.pivot_table(train_df,index=['SaleCondition'],values=["SalePrice"],aggfunc=np.mean).reset_index()
sc_piv.rename(columns = {'SalePrice':'SC_Mean'}, inplace=True)
train_df = pd.merge(left=train_df, right=sc_piv,how='left', on='SaleCondition')
train_df['SaleCondition_Value'] = pd.cut(train_df['SC_Mean'], bins, labels=binLabels).astype(int)

# train_df.drop(['Nbhd_Mean','GT_Mean','HS_Mean','Ex1_Mean','Ex2_Mean','SC_Mean'],axis=1,inplace=True)

train_df.fillna(0, inplace=True)


#################### Unnecessary Variables & Multi-Collinearity ####################
'''
Potentially Drop:
    - Id: Names are unnecessary
    - BsmtFinSF1, BsmtFinSF2, BsmtUnfSF: (Redundant with TotalBsmtSF)
    - MoSold: (No seasonality trends with SalePrice, low r-value)
    - PoolQC_Scale & PoolArea: Only 7 Houses of 1460 examples have pools, not big enough of a sample size to show statistical significance.
    - MiscFeature: Rare; those with MiscFeatures show no trend
'''
plt.figure(6)
sns.regplot('MoSold','SalePrice',data = train_df)

train_df.drop(['Id','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','MoSold', 'MiscFeature'],axis=1, inplace=True)

'''
# Correlation Matrix (numerical vars)
sns.set(font_scale=0.5)
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12,59))
plt.figure(2)
sns.heatmap(corrmat, vmax=0.8, square=True,  xticklabels=True, yticklabels=True)
#plt.show()
plt.savefig('HeatMap.png')

Use Correlation Heatmap:
    Multi-Collinearity:
        - TotRmsAbvGrd, GrLivArea (rvalue=0.82549)
        - 1stFlrSF, TotBsmtSF (rvalue=0.8195)
        - GarageArea, GarageCars (rvalue=0.8825)
        - FireplaceQu_Scale, Fireplaces (rvalue=0.8632)
    Pick weaker correlation with SalePrice to drop
'''
train_df.drop(['TotRmsAbvGrd','TotalBsmtSF','GarageCars','Fireplaces'],axis=1,inplace=True)

# Correlation Matrix (numerical vars)
sns.set(font_scale=0.7)
corrmat = train_df.corr()
f, ax = plt.subplots(figsize=(12,9))
plt.figure(7)
sns.heatmap(corrmat, vmax=0.8, square=True,  xticklabels=True, yticklabels=True)
#plt.show()
plt.savefig('HeatMap.png')

# Separate categorical and numerical features
#objHouse = train_df.select_dtypes(include=['object']).copy()
#numHouse = train_df.select_dtypes(include=['int64','float64']).copy()


######################################################## APPLY ALL CHANGES TO CROSS-VALIDATION SET ##################################################################

#################### Missing/null Numeric Values ####################

cv_df.drop(['LotFrontage','MasVnrArea','GarageYrBlt'],axis=1, inplace=True)

# Rename all variables that start with anumber
cv_df.rename(columns = {'1stFlrSF':'FirstFlrSF',
                            '2ndFlrSF':'SecFlrSF',
                            '3SsnPorch':'ThreeSsnPorch'}, inplace=True)

#################### Missing/Null Categorical Variables ####################
# Cross-reference with readme file to check if they are intentional NAs

cv_df['Electrical'].fillna('SBrkr',inplace=True)

# Replace all the NAs with 0 (because these NAs are intentional)
cv_df.fillna(0, inplace=True)

#################### Feature Engineer and Convert Categorical to Numerical ####################
# Binary
cv_df['isStreetPave'] = (cv_df['Street'] == 'Pave').astype(int)
cv_df['isAlleyPave'] = (cv_df['Alley'] == 'Pave').astype(int)
cv_df['isCentralAC'] = (cv_df['CentralAir'] == 'Y').astype(int)
cv_df['isDrivePaved'] = (cv_df['PavedDrive'] =='Y').astype(int)
cv_df['isCondNormal'] = (cv_df['Condition1'] == 'Norm').astype(int)
cv_df['is1Fam_TwnhsE'] = ((cv_df['BldgType'] == '1Fam') | (cv_df['BldgType'] == 'TwnhsE')).astype(int)
cv_df['isCDS_FR3'] = ((cv_df['LotConfig'] == 'CulDSac') | (cv_df['LotConfig'] == 'FR3')).astype(int)
cv_df['isHip_Shed'] = ((cv_df['RoofStyle'] == 'Hip') | (cv_df['LotConfig'] == 'Shed')).astype(int)
cv_df['isStdCompShg'] = (cv_df['RoofMatl'] == 'CompShg').astype(int)
cv_df['isBrk_Stone'] = ((cv_df['MasVnrType'] == 'BrkFace') | (cv_df['MasVnrType'] == 'Stone')).astype(int)
cv_df['isFoundConcrete'] = (cv_df['Foundation'] == 'PConc').astype(int)
cv_df['isStdHeating'] = (cv_df['Heating'] == 'GasA').astype(int)
cv_df['isStdElectrical'] = (cv_df['Electrical'] == 'SBrkr').astype(int)
cv_df['isNew_RegContract'] = ((cv_df['SaleType'] == 'New') | (cv_df['SaleType'] == 'Con')).astype(int)

# Ordinal (assume scale from none/worst to best)
#   Note: Unintentional NAs have been dealth with above, the rest of features have intentional NA values 
#    to represent 'None' (e.g. no basement); these have been converted to 0 below to represent the 
#    lowest value of the scales mapped below.(.map() function does not map NAs anyway)
cv_df['Zone'] = cv_df['MSZoning'].map({"C (all)":1, "RH":2, "RM":3, "RL":4, "FV":5})
cv_df['irregularLot_Scale'] = cv_df['LotShape'].map({"Reg":0, "IR1":1, "IR2":2, "IR3":3})
cv_df['landContour_Scale'] = cv_df['LandContour'].map({"Lvl":0, "Low":1, "HILS":2, "BNK":3})
cv_df['Util'] = cv_df['Utilities'].map({"ELO":1, "NoSeWa":2, "NoSewr":3, "AllPub":4})
cv_df['landSlope_Scale'] = cv_df['LandSlope'].map({"Gtl": 1, "Mod":2, "Sev":3})
cv_df['exterQual_Scale'] = cv_df['ExterQual'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
cv_df['exterCond_Scale'] = cv_df['ExterCond'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
cv_df['BsmtHeight_Scale'] = cv_df['BsmtQual'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
cv_df['BsmtCond_Scale'] = cv_df['BsmtCond'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
cv_df['BsmtExposure_Scale'] = cv_df['BsmtExposure'].map({"No":1, "Mn":2, "Av":3, "Gd":4})
cv_df['BsmtFinType1_Scale'] = cv_df['BsmtFinType1'].map({"Unf":1, "LwQ":2, "Rec":3, "BLQ": 4, "ALQ":5, "GLQ":6})
cv_df['BsmtFinType2_Scale'] = cv_df['BsmtFinType2'].map({"Unf":1, "LwQ":2, "Rec":3, "BLQ": 4, "ALQ":5, "GLQ":6})
cv_df['HeatingQC_Scale'] = cv_df['HeatingQC'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})  
cv_df['KitchenQual_Scale'] = cv_df['KitchenQual'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}) 
cv_df['FuncDamage_Scale'] = cv_df['Functional'].map({"Typ":0, "Min1":1, "Min2":2, "Mod":3, "Maj1": 4, "Maj2":5, "Sev":6,"Sal":7})
cv_df['FireplaceQu_Scale'] = cv_df['FireplaceQu'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
cv_df['GarageQual_Scale'] = cv_df['GarageQual'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
cv_df['GarageCond_Scale'] = cv_df['GarageCond'].map({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})
cv_df['GarageFinish_Scale'] = cv_df['GarageFinish'].map({"Unf": 1, "RFn":2, "Fn":3})
cv_df['PoolQC_Scale'] = cv_df['PoolQC'].map({"Fa":1, "TA":2, "Gd":3, "Ex":4})
cv_df['FenceQual_Scale'] = cv_df['Fence'].map({"MnWw":1, "GdWo":2, "MnPrv":3, "GdPrv":4})

# We want to use the training data pivoted means to calculate to bucket the cv variables
# Neighborhood
cv_df = pd.merge(left=cv_df, right=nbhd_piv,how='left', on='Neighborhood')                                   # Left Merge the mean SalePrice with the respective Neighborhoods
cv_df['Nbhd_Value'] = pd.cut(cv_df['Nbhd_Mean'], bins, labels=binLabels).astype(int)

# GarageType
cv_df = pd.merge(left=cv_df, right=gt_piv,how='left', on='GarageType')
cv_df['GarageType_Value'] = pd.cut(cv_df['GT_Mean'], bins, labels=binLabels).astype(int)

# HouseStyle
cv_df = pd.merge(left=cv_df, right=hs_piv,how='left', on='HouseStyle')
cv_df['HouseStyle_Value'] = pd.cut(cv_df['HS_Mean'], bins, labels=binLabels).astype(int)

# Exterior1st
cv_df = pd.merge(left=cv_df, right=ex1_piv,how='left', on='Exterior1st')
cv_df['Exterior1st_Value'] = pd.cut(cv_df['Ex1_Mean'], bins, labels=binLabels).astype(int)

# Exterior2nd
cv_df = pd.merge(left=cv_df, right=ex2_piv,how='left', on='Exterior2nd')
cv_df['Exterior2nd_Value'] = pd.cut(cv_df['Ex2_Mean'], bins, labels=binLabels).astype(int)

# SaleCondition
cv_df = pd.merge(left=cv_df, right=sc_piv,how='left', on='SaleCondition')
cv_df['SaleCondition_Value'] = pd.cut(cv_df['SC_Mean'], bins, labels=binLabels).astype(int)

cv_df.drop(['Nbhd_Mean','GT_Mean','HS_Mean','Ex1_Mean','Ex2_Mean','SC_Mean'],axis=1,inplace=True)

# Delete categorical variables above since they will be replaced
# Make SalePrice to be at the end                        
cv_df.drop(['MSZoning','Street','Alley','CentralAir','PavedDrive','LotShape','LandContour',
            'Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
            'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType', 'ExterQual',
            'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'Heating','HeatingQC','Electrical','KitchenQual','Functional','FireplaceQu','GarageType',
            'GarageQual','GarageCond','GarageFinish','PoolQC','Fence','SaleCondition','SaleType'],axis=1, inplace=True)

cv_df.fillna(0, inplace=True)

#################### Unnecessary Variables & Multi-Collinearity ####################

cv_df.drop(['Id','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','MoSold', 'MiscFeature'],axis=1, inplace=True)

cv_df.drop(['TotRmsAbvGrd','TotalBsmtSF','GarageCars','Fireplaces'],axis=1,inplace=True)

################################################################ END OF Cross-Validation SET ##################################################################################

train_df.drop(['Nbhd_Mean','GT_Mean','HS_Mean','Ex1_Mean','Ex2_Mean','SC_Mean'],axis=1,inplace=True)

# Delete categorical variables above since they will be replaced
# Make SalePrice to be at the end  
sale = train_df['SalePrice']                          
train_df.drop(['MSZoning','Street','Alley','CentralAir','PavedDrive','LotShape','LandContour',
            'Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
            'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType', 'ExterQual',
            'ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
            'Heating','HeatingQC','Electrical','KitchenQual','Functional','FireplaceQu','GarageType',
            'GarageQual','GarageCond','GarageFinish','PoolQC','Fence','SaleCondition','SaleType','SalePrice'],axis=1, inplace=True)
train_df['SalePrice'] = sale

# Log Transform before modeling
train_df['ylog'] = np.log(train_df['SalePrice'])
#cv_df['ylog'] = np.log(cv_df['SalePrice'])
#test_df['ylog'] = np.log(test_df['SalePrice'])


#################### Modeling ####################

'''
#x_num = train_df.loc[:, train_df.columns != 'SalePrice'].select_dtypes(include=['int64','float64']).columns
# x_obj = pd.DataFrame(train_df.loc[:, train_df.columns != 'SalePrice'].select_dtypes(include=['object'])).columns
#trainDF = pd.DataFrame({'xn':[x_num], 'xo':[x_obj],'y':'SalePrice'})
x_cols = "+".join(train_df.columns.difference(["SalePrice"]))
lm_formula = "SalePrice~" + x_cols
linModel = sm.ols(lm_formula, data = train_df).fit()
print(linModel.params)
print(linModel.summary())
'''
list2 = ['BsmtExposure_Scale', 'BsmtFinType1_Scale', 'BsmtFullBath', 
'BsmtHeight_Scale', 'FuncDamage_Scale', 'GarageArea', #'GarageFinish_Scale', , 'exterQual_Scale','BsmtCond_Scale',  'isHip_Shed',
'GrLivArea', 'KitchenQual_Scale', 'MSSubClass', 'Nbhd_Value', 'OverallCond', 
'OverallQual', 'ScreenPorch', 'WoodDeckSF', 'isCDS_FR3',
'isCondNormal', 'isNew_RegContract']
x_cols2 = "+".join(list2)
#train_df['ylog'] = np.log(train_df['SalePrice'])

# Linear Model
lm_formula2 = "ylog~" + x_cols2
linModel2 = sm.ols(lm_formula2, data = train_df).fit()
print(linModel2.params)
print(linModel2.summary())

y_pred = np.exp(linModel2.predict(train_df))

lm_rmse = np.sqrt(mean_squared_error(np.log(train_df['SalePrice']), np.log(y_pred)))
print('Training RMSE: ',lm_rmse)

cv_y_pred = np.exp(linModel2.predict(cv_df))
cv_lm_rmse = np.sqrt(mean_squared_error(np.log(cv_df['SalePrice']), np.log(cv_y_pred)))
print('Cross-Validation RMSE: ',cv_lm_rmse)

#lm_rmse = np.sqrt(mean_squared_error(np.log(test_df['SalePrice']), y_pred))
#print('RMSE: ' + lm_rmse)

# Random Forest
x_train = train_df.drop(["SalePrice","ylog"],axis=1)
y_train = train_df['SalePrice']
rfModel = RandomForestClassifier(n_estimators=100)
rfModel.fit(x_train,y_train)
x_cv = cv_df.drop(["SalePrice"],axis=1)
rf_pred = rfModel.predict(x_cv)
rf_rmse = np.sqrt(mean_squared_error(np.log(cv_df['SalePrice']), np.log(rf_pred)))
print('Random Forest Cross-Validation RMSE: ',rf_rmse)

