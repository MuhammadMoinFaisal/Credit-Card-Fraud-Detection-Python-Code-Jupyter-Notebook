import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


# # Load the Dataset

# In[282]:


data = pd.read_csv('student-mat.csv', sep = ';')


# In[283]:


data


# In[284]:


data1 = pd.read_csv('student-por.csv', sep = ';')


# In[285]:


data1


# In[286]:


data.columns


# In[287]:


data1.columns


# In[288]:


data_con = data.append(data1)


# In[289]:


data_con


# # Cleaning Data

# In[290]:


data_con.isnull()


# In[291]:


data_con.isnull().sum()


# In[292]:


plt.figure(figsize=(10,6))
sns.heatmap(data_con.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})


# # Data Exploration

# In[550]:


sns.pairplot(data[['absences', 'failures', 'school', 'Medu', 'Fjob', 'Walc', 'Dalc', 'G1','G2', 'G3']])


# In[557]:


sns.countplot(x = 'G1', data = data)


# In[558]:


sns.countplot(x = 'G2', data = data)


# In[559]:


sns.countplot(x = 'G3', data = data)


# In[560]:


sns.heatmap(data.corr())


# In[562]:


sns.violinplot(x='Fjob', y='G1', data=data, split = True)


# In[565]:


sns.violinplot(x='Fjob', y='G2', data=data, split = True)


# In[566]:


sns.violinplot(x='Fjob', y='G3', data=data, split = True)


# In[564]:


sns.boxplot(x ='Fjob', y= 'G1', data=data)


# In[567]:


sns.boxplot(x ='Fjob', y= 'G2', data=data)


# In[568]:


sns.boxplot(x ='Fjob', y= 'G3', data=data)


# # Check for Duplicates

# In[293]:


data_con.drop_duplicates(inplace = True)


# # Remove Ir-relevant Features (Columns)

# In[294]:


data_con.drop(['address', 'famsize', 'romantic', 'goout', 'nursery'], axis=1 , inplace=True)


# In[295]:


data_con


# In[296]:


data_con.head(1)


# In[297]:


data_con.columns


# In[298]:


data_con.dtypes


# # Create Dummies

# In[299]:


lst = ['school', 'sex','Pstatus',
       'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'higher', 'internet']


# In[300]:


data_cond = pd.get_dummies(data_con[lst], drop_first = True)


# In[301]:


data_cond


# In[302]:


data_final = pd.concat([data_con, data_cond], axis = 1)


# In[303]:


data_final


# In[304]:


data_final.drop(['school', 'sex','Pstatus',
       'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
       'higher', 'internet'], axis=1 , inplace=True)


# In[305]:


data_final


# In[306]:


data_final.columns


# # B. Investigate Perofrmance on Full Set of Features

# In[312]:


from sklearn.model_selection import train_test_split


# In[418]:


X = data_final[['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel',
       'freetime', 'Dalc', 'Walc', 'health', 'absences',
       'school_MS', 'sex_M', 'Pstatus_T', 'Mjob_health', 'Mjob_other',
       'Mjob_services', 'Mjob_teacher', 'Fjob_health', 'Fjob_other',
       'Fjob_services', 'Fjob_teacher', 'reason_home', 'reason_other',
       'reason_reputation', 'guardian_mother', 'guardian_other',
       'schoolsup_yes', 'famsup_yes', 'paid_yes', 'activities_yes',
       'higher_yes', 'internet_yes', 'G1', 'G2']]


# In[419]:


X


# In[420]:


y = data_final[['G3']]
y


# # Train & Test Split

# In[437]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# # Doing Optimized Training on Random Forest and SVM

# In[450]:


from sklearn.ensemble import RandomForestRegressor


# In[453]:


rf_random = RandomForestRegressor


# In[455]:


n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]


# In[456]:


max_features = ['auto', 'sqrt']


# In[458]:


max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]


# In[459]:


min_samples_split = [2, 5, 10, 15, 100]


# In[460]:


min_samples_leaf = [1,2,5,10]


# In[461]:


from sklearn.model_selection import RandomizedSearchCV


# In[462]:


random_grid = {'n_estimators': n_estimators, 'max_features':max_features, 'max_depth':max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}


# In[463]:


print(random_grid)


# In[464]:


rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter =10, cv = 5, verbose = 2, random_state = 42,n_jobs = 1)


# In[465]:


rf_random.fit(X_train, y_train)


# In[466]:


predict = rf_random.predict(X_test)


# In[467]:


plt.scatter(y_test, predict)


# In[469]:


r2_score=metrics.r2_score(y_test,predict)


# In[470]:


r2_score


# In[473]:


rf_random.cv_results_


# In[474]:


df = pd.DataFrame(rf_random.cv_results_)


# In[475]:


df


# In[476]:


dir(rf_random)


# In[480]:


rf_random.best_params_


# In[483]:


modelRF = RandomForestRegressor(n_estimators = 700,
 min_samples_split = 5,
 min_samples_leaf = 10,
 max_features = 'auto',
 max_depth = 20)


# In[484]:


modelRF.fit(X_train, y_train)


# In[485]:


predict = modelRF.predict(X_test)


# In[487]:


r2_score=metrics.r2_score(y_test,predict)
r2_score


# In[440]:


svr = SVR(kernel='rbf',C=100, epsilon=0.00001).fit(X_train,y_train)
print("{:.2f} is the accuracy of the SV Regressor".format(svr.score(X_train,y_train)))


# In[472]:


y_pred = svr.predict(X_test)
r2_score=metrics.r2_score(y_test,y_pred)
r2_score


# In[489]:


from sklearn.model_selection import GridSearchCV
GS_CV = GridSearchCV(SVR(epsilon=0.00001), {'C':[1, 10, 20], 'kernel': ['rbf', 'linear']}, return_train_score = False)


# In[490]:


GS_CV.fit(X_train, y_train)


# In[491]:


GS_CV.cv_results_


# In[492]:


datas = pd.DataFrame(GS_CV.cv_results_)


# In[493]:


datas


# In[494]:


dir(GS_CV)


# In[495]:


GS_CV.best_score_


# In[498]:


GS_CV.best_params_


# In[501]:


GS_CV = GridSearchCV(SVR(epsilon=0.00001), {'C': [20], 'kernel': ['rbf']}, return_train_score = False)


# In[502]:


GS_CV.fit(X_train, y_train)


# In[504]:


y_pred = GS_CV.predict(X_test)


# In[505]:


r2_score=metrics.r2_score(y_test,y_pred)
r2_score


# # Feature Selection Using Univariate Selection

# In[506]:


from sklearn.feature_selection import SelectKBest


# In[507]:


from sklearn.feature_selection import chi2


# I have so many feautres, to select the best feature among them i will apply SelectKBest class to select the top 10 best features

# In[508]:


best_features = SelectKBest(score_func = chi2, k = 10)


# In[509]:


fit = best_features.fit(X, y)


# In[510]:


dfscores = pd.DataFrame(fit.scores_)


# In[511]:


dfcolumns = pd.DataFrame(X.columns)


# In[512]:


featurescores = pd.concat([dfcolumns, dfscores], axis = 1)


# In[513]:


featurescores.columns = ['Specs', 'Score']


# In[514]:


featurescores


# In[515]:


print(featurescores.nlargest(10, 'Score'))


# In[516]:


X_f = data_final[['absences', 'failures', 'schoolsup_yes', 'Medu', 'school_MS', 'Fjob_teacher', 'Walc', 'Dalc', 'G1','G2']]


# In[517]:


y_f = data_final[['G3']]


# # Feature Selection using Extra Tree Regressor

# In[524]:


from sklearn.ensemble import ExtraTreesRegressor


# In[525]:


model = ExtraTreesRegressor()


# In[527]:


model_fit = model.fit(X,y)


# In[529]:


print(model.feature_importances_)


# In[531]:


importantfeatures = pd.Series(model.feature_importances_, index = X.columns)


# In[532]:


importantfeatures.nlargest(10).plot(kind ='barh')


# In[533]:


X_f = data_final[['G1','G2', 'absences','failures','studytime','age','traveltime','reason_home','freetime','famrel']]


# In[534]:


y_f = data_final[['G3']]


# In[535]:


X_train, X_test, y_train, y_test = train_test_split(X_f, y_f, test_size=0.20, random_state=42)


# # After Feature Selection Doing Optimized Training on Couple of Models

# In[536]:


svr = SVR(kernel='rbf',C=100, epsilon=0.00001).fit(X_train,y_train)
print("{:.2f} is the accuracy of the SV Regressor".format(svr.score(X_train,y_train)))


# In[538]:


y_pred = svr.predict(X_test)
r2_score=metrics.r2_score(y_test,y_pred)
r2_score


# In[577]:


GS_CV = SVR(epsilon=0.00001, C = 20, kernel = 'rbf')


# In[578]:


GS_CV_F.fit(X_train, y_train)


# In[579]:


y_pred = GS_CV_F.predict(X_test)


# In[580]:


r2_score=metrics.r2_score(y_test,y_pred)
r2_score


# In[581]:


modelRF = RandomForestRegressor(n_estimators = 700,
 min_samples_split = 5,
 min_samples_leaf = 10,
 max_features = 'auto',
 max_depth = 20)


# In[582]:


modelRF.fit(X_train, y_train)


# In[583]:


predict = modelRF.predict(X_test)


# In[585]:


r2_score=metrics.r2_score(y_test,predict)
r2_score


# In[ ]:




