import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import chardet#for encoding
import warnings# to avoid the warnings

warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns',0)

#Let's see which encoding we have to apply.
with open("new.csv","rb") as f:
    result=chardet.detect(f.read(100000))
print(result)

#so,we have to apply GB2312 encoding.
data=pd.read_csv("new.csv",encoding="GB2312")

df0 = data.copy()
print("Data Heading:", data.head)
print("Data shape:", data.shape)
print("Missing values per column:")
print(data.isnull().sum())

#Let's Visualize the missing value
sns.heatmap(data.isnull(),yticklabels=False,cbar=False)

#Drop 'DOM' Columns (irrelevant)
data.drop(columns=['DOM'],axis=1,inplace=True)

# Fill missing values with mode or median value
data['buildingType'].fillna(data['buildingType'].mode()[0], inplace=True)
data['elevator'].fillna(data['elevator'].mode()[0], inplace=True)
data['fiveYearsProperty'].fillna(data['fiveYearsProperty'].mode()[0], inplace=True)
data.subway.fillna(data.subway.median(),inplace=True)
data.communityAverage.fillna(data.communityAverage.median(),inplace=True)

print(data.livingRoom.unique())
print(data.floor.unique())

#so,floor have a chinese character...
print(data.bathRoom.unique())
print(data.drawingRoom.unique())

def Trade_Time(x):
    return x[0:4]
data['tradeTime']=data['tradeTime'].apply(Trade_Time)
print(data.head)

#clean & convert tradetime & etc, into int numeric
data['tradeTime'] = pd.to_numeric(data['tradeTime'])
data['livingRoom'] = data['livingRoom'].apply(pd.to_numeric, errors='coerce')
data['drawingRoom'] = data['drawingRoom'].apply(pd.to_numeric, errors='coerce')
data['bathRoom'] = data['bathRoom'].apply(pd.to_numeric, errors='coerce')
data['constructionTime'] = data['constructionTime'].apply(pd.to_numeric, errors='coerce')

for col in ['livingRoom', 'drawingRoom', 'bathRoom', 'constructionTime']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

#now if we check livingRoom Column it is clean data.
print(data.livingRoom.unique())

#Now,Split the column into a Floor_Type and Floor_Height
def Floor_Type(x):
    return x.split(' ')[0]

def Floor_Height(y):
    try:
        return int(y.split(' ')[1])
    except:
        return np.nan

data['floor_type'] = data['floor'].str.split(' ').str[0]
data['floor_height'] = data['floor'].str.extract(r'(\d+)').astype(float)

print(data.columns)

#DROPING UNNECESSARY COLUMNS
# Remove bad entries and drop unnecessary columns
data = data[data['buildingType'] >= 1]
data.drop(columns=['floor', 'url', 'id', 'Cid', 'price'], inplace=True)

print(data.head)

#Let's Perform one hot encoding
print(data.buildingType.unique())
print(data.renovationCondition.unique())
print(data.buildingStructure.unique())
#so,for buildingType we have a data like 0.5   0.333 0.125 0.25  0.429 0.048 0.375 0.667
# Which is unnecessary so,we have to remove them

# Drop missing
data.dropna(axis=0, inplace=True)

#so,for buildingType we have a data like 0.5   0.333 0.125 0.25  0.429 0.048 0.375 0.667
# Which is unnecessary so,we have to remove them

#Removing unnecessary data which is present in buildingType
data=data[data['buildingType']>=1]
print(data.shape)

#let's take a copy of our data for future use
df=data.copy()

col_for_dummies=['renovationCondition','buildingStructure','buildingType',
                 'district','elevator','floor_type']
data=pd.get_dummies(data=data,columns=col_for_dummies,drop_first=True)
data.head()

print(data.shape)
print(df0.shape)

# Drop missing
data.dropna(axis=0, inplace=True)

# Final data shape
print("Cleaned data shape:", data.shape)

#To see which columns are still available
print(data.columns)

#EDA SECTION:
df1=data[['Lng','Lat','tradeTime','totalPrice','followers','livingRoom','drawingRoom','kitchen',
    'bathRoom','square','communityAverage','ladderRatio']]

# Correlation Heatmap
plt.figure(figsize=(16, 16))
sns.heatmap(df1.corr(), annot=True, cmap="RdYlGn")
plt.title("Correlation Heatmap")
plt.show()

# KDE Plot for Price
sns.kdeplot(data=data['totalPrice'], fill=True)  # use fill=True instead of shade (newer syntax)
plt.title("KDE Plot of Total Price")
plt.xlabel("Total Price")
plt.show()

# Boxplot (use sample for performance)
df_sample = df.sample(n=1000, random_state=42)

sns.boxplot(x='renovationCondition', y='followers', data=df_sample)
plt.title("Boxplot: Followers vs Renovation Condition")
plt.show()

# Scatterplot: Followers vs Community Average
sns.scatterplot(x='followers', y='communityAverage', hue='elevator', data=df_sample)
plt.title("Scatterplot: Followers vs Community Avg (by Elevator)")
plt.show()

# Scatterplot: Total Price vs Community Average
sns.scatterplot(x=df['totalPrice'], y=df['communityAverage'], hue=df['renovationCondition'])
plt.title("Scatterplot: Price vs Community Avg (by Renovation)")
plt.show()

# Line plot of Community Average
sns.lineplot(data=df['communityAverage'])
plt.title("Line plot of Community Average")
plt.show()

print(data.head)
print(data.shape)

data.to_csv("After_EDA.csv", index=False)
print("EDA complete. Output saved as 'After_EDA.csv' :D")
#-------Original Github "EDA" python file (ends here)---------


# ---------- After combining Feature Selection ---------
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler

data=pd.read_csv("After_EDA.csv")
data.head()

sns.scatterplot(data=data['totalPrice'])
plt.show()

data['totalPrice'].describe()
data.totalPrice.quantile([0.25,0.50,0.75])

Q1=205
Q3=428
IQR=Q3-Q1
print("So,IQR of our totalPrice Features is",IQR)

High=Q3+1.5*IQR
Low=Q1-1.5*IQR
print("High value of our totalPrice Features is",High)
print("Low value of our totalPrice Features is",Low)

df=data.copy()

# Remove outliers
df=df[df['totalPrice']<=High]
df=df[df['totalPrice']>=Low]

sns.scatterplot(data=df['totalPrice'])
plt.show()

print(data.shape)
print(df.shape)
print("Total outlier Removed =",data.shape[0]-df.shape[0])

var = df.columns
X=df.drop(['totalPrice'],axis=1)
y=df['totalPrice']

model=ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)

plt.figure(figsize=(12,10))
feature_importance=pd.Series(model.feature_importances_,index=X.columns)
feature_importance.nlargest(30).plot(kind='barh')
plt.title("Top 30 Feature Importance's")
plt.show()

scaling=MinMaxScaler()
col_for_normalization=['Lng', 'Lat','followers','square','livingRoom', 'drawingRoom',
                       'kitchen', 'bathRoom','ladderRatio', 'fiveYearsProperty',
                       'subway', 'communityAverage','floor_height']
df[col_for_normalization]=scaling.fit_transform(df[col_for_normalization])
df.head()

df.to_csv("Data_For_Model.csv")
print("Feature selection & scaling complete. Saved as 'Data_For_Model.csv'")
#--------Feature Selection Ends Here --------------
