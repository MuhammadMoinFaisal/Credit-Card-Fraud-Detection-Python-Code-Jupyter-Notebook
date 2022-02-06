# To Perform Exploratory Data Analysis on the Data and to Predict the Output
# I will follow the following various steps given as 
# 1- Collecting Data        
# 2- Analyzing Data
# 3- Data Wrangling
# 4- Train & Test
# 5- Predict the output

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

#Load the CSV File
data = pd.read_csv(r"C:\UpworkProjects\DwayneSalo\covid.csv")

print(data)

# Analyzing the Data
print(data.info())

print(data.isna().sum().sum())

print(data.describe().T)

sns.countplot(x = 'sex', data = data)

sns.countplot(x = 'diabetes',  data = data)

sns.countplot(x = 'patient_type',  data = data)

sns.violinplot(x='sex', y='intubed', data=data, split = True)


sns.heatmap(data.corr())

data1 = data.replace(99, np.NaN)
data2 = data1.replace(98, np.NaN)
data3 = data2.replace(97, np.NaN)
data4 = data3.replace(40, np.NaN)
data5 = data4.replace(54, np.NaN)
data6 = data5.replace(119, np.NaN)

# Finding out the Mean, Standard Deviation of all the columns
print(data6.describe().T)


sns.heatmap(data6.isnull(), yticklabels = False, cmap = "viridis")

data6['contact_other_covid'].fillna(method = 'ffill', inplace  = True)

data6.drop(['icu', 'id', 'intubed', 'pregnancy'], axis = 1, inplace = True)

# Creating a Heat Map
sns.heatmap(data6.isnull(), yticklabels = False, cmap = "viridis")

# Check if there are any null values in any column
print(data6.isnull().sum())
data6.dropna(inplace = True)
print(data6.isnull().sum())
print(data6)

data6.loc[data6['date_died'] != '9999-99-99' , 'date_died'] = 'yes die'

data6['date_died'] = data6['date_died'].replace('9999-99-99', 'no die')

print(data6['date_died'])

# Applying Label Encoder
from sklearn.preprocessing import LabelEncoder

new = {}
for k in (['entry_date', 'date_symptoms', 'date_died']) :
    new[k] = LabelEncoder()
    data6[k] = new[k].fit_transform(data6[k]) 

print(data6['date_died'])
print(data6)

data6['Died or not'] = data6['date_died']
data6.drop('date_died', axis= 1, inplace = True)

# Applying Machine Learning Algorithm
#datas = data6[0:2000]
X  = data6.drop("Died or not", axis = 'columns')
print(X)
y = data6["Died or not"]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train.shape)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(X_train, y_train)
pred = lg.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score

print(classification_report(y_test, pred))

print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))

#Deep Learning Algorithm

import tensorflow as tf
from tensorflow import keras
#Now let us create a neural network
model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (18,), activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid'),
    
    
    ])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train, epochs = 20)
model.evaluate(X_test, y_test)
yp = model.predict(X_test)
print(yp[:5])
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_pred[:10]
print(y_pred)
print(accuracy_score(y_test, y_pred))

#Hybrid Algorithm

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

# Defining the Number of Estimators, learning rate and maximum features
ABC = AdaBoostClassifier(n_estimators = 10 , learning_rate = 0.4)
RFC = RandomForestClassifier(n_estimators = 10, max_depth = 6, max_features = 10)
XGC = XGBClassifier(objective = 'binary:logistic', n_estimators = 10, random_state = 42, learning_rate = 0.5, max_depth = 5)

vot = VotingClassifier(estimators = [('ABC', ABC), ('XGC',XGC), ('RFC', RFC)], voting = 'hard')

classifiers = [RandomForestClassifier(n_estimators = 10, max_depth = 6, max_features = 4), 
               XGBClassifier(objective = 'binary:logistic', n_estimators = 10, random_state = 42, learning_rate = 0.5, max_depth = 5),
               AdaBoostClassifier(n_estimators = 50 , learning_rate = 0.4)
        ]
# Importing Metrics from sklearn
import sklearn.metrics as metrics
for i in classifiers :
    classifier = i
    classifier.fit(X_train, y_train)
    res = classifier.predict(X_test)

# Checking out the Accuracy
    print("Accuracy score for " + str(i) + " is : ", metrics.accuracy_score(y_test, res))
    print(str(i) + "score is : ", classifier.score(X_train, y_train))
    print(str(i) + "score is : ", classifier.score(X_test, y_test))
    plt.show()