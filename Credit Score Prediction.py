#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import itertools
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[4]:


df_application=pd.read_csv(r"C:\Users\ngaih\OneDrive\Desktop\Xccelarate Course Material\Assignment\Machine Learning Project\Data source\HenryChan application_record.csv", encoding = 'utf-8')
df_credit=pd.read_csv(r"C:\Users\ngaih\OneDrive\Desktop\Xccelarate Course Material\Assignment\Machine Learning Project\Data source\Henru Chan credit_record.csv", encoding = 'utf-8')


# In[ ]:





# In[5]:


#Feature Engineering, grading each client's credit 


# In[6]:


#When did the client open the account (The negative number means how many months before today(20230412)

AC_Open_Month=pd.DataFrame(df_credit.groupby(['ID'])['MONTHS_BALANCE'].agg(min))
AC_Open_Month=AC_Open_Month.rename(columns={'MONTHS_BALANCE':'AC_Open_Months_Ago'})

#Merging AC_Open_Months_Ago into df_application

new_df_application=df_application.merge(AC_Open_Month, how = 'left', on ='ID')
print(new_df_application)


# In[7]:


#Defining good credit record and bad credit record which are makred as 1 and 0 respectively.
# 0=good=overdue less than or equal 60 days
# 1=bad=overdue more than 60 days


# In[8]:


df_credit['Grade'] = None
df_credit['Grade'][df_credit['STATUS']=='2']='Bad'
df_credit['Grade'][df_credit['STATUS']=='3']='Bad'
df_credit['Grade'][df_credit['STATUS']=='4']='Bad'
df_credit['Grade'][df_credit['STATUS']=='5']='Bad'


# In[9]:


grading=df_credit.groupby('ID').count()
grading['Grade'][grading['Grade']>0]='Bad'
grading['Grade'][grading['Grade']==0]='Good'
grading=grading[['Grade']]
new_df_application=pd.merge(new_df_application, grading, how='inner', on='ID')
new_df_application['target']=new_df_application['Grade']
new_df_application.loc[new_df_application['target']=='Bad', 'target']=1
new_df_application.loc[new_df_application['target']=='Good', 'target']=0


# In[10]:


print(grading['Grade'].value_counts())
grading['Grade'].value_counts(normalize=True)


# In[ ]:





# In[11]:


#Renaming the column name of each feature


# In[12]:


new_df_application.rename(columns={'CODE_GENDER':'Gender','FLAG_OWN_CAR':'Car_Possession','FLAG_OWN_REALTY':'Property posession',
                         'CNT_CHILDREN':'How many children','AMT_INCOME_TOTAL':'Annual Income',
                         'NAME_EDUCATION_TYPE':'Education Level','NAME_FAMILY_STATUS':'Marital Status',
                        'NAME_HOUSING_TYPE':'Housing Type','FLAG_EMAIL':'Email',
                         'NAME_INCOME_TYPE':'Types of Income','FLAG_WORK_PHONE':'Work Phone Posession',
                         'FLAG_PHONE':'Personal Mobile Phone Posession','CNT_FAM_MEMBERS':'Family Size',
                        'OCCUPATION_TYPE':'Occupation'
                        },inplace=True)


# In[13]:


new_df_application.dropna()
new_df_application = new_df_application.mask(new_df_application == 'NULL').dropna()


# In[14]:


feature_names_df=pd.DataFrame(new_df_application.columns, columns=['Features'])
feature_names_df['IV']=None
feature_to_drop=['FLAG_MOBIL', 'AC_Open_Months_Ago', 'Grade', 'target', 'ID' ]

for i in feature_to_drop:
    feature_names_df.drop(feature_names_df[feature_names_df['Features']==i].index, inplace=True)


# In[ ]:





# In[15]:


#Defining functions for calculating Weight of evidence (WOE) and information value (IV)


# In[16]:


def calc_iv (df, feature, target, pr=False):
    lst=[]
    df[feature]=df[feature].fillna('NULL')
    
    for i in range(df[feature].nunique()):
        val=list(df[feature].unique())[i]
        lst.append([feature, val, 
                    df[df[feature]==val].count()[feature],
                    df[(df[feature]==val) & (df[target]==0)].count()[feature],
                    df[(df[feature]==val) & (df[target]==1)].count()[feature]])
        
    data=pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])
    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All']-data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WOE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])
    
    data= data.replace({'WOE':{np.inf:0, -np.inf: 0}})
    
    data['IV']=data['WOE'] * (data['Distribution Good'] - data['Distribution Bad'])
    
    data= data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range (len(data.index))
    
    if pr:
        print(data)
        print('IV = ' ,data['IV'].sum())
        
    iv = data['IV'].sum()
    print('This varible\'s IV is :' , iv)
    print(df[feature].value_counts())
    return iv, data


# In[17]:


def convert_dummy(df, feature, rank=0):
    pos= pd.get_dummies(df[feature], prefix=feature)
    mode= df[feature].value_counts().index[rank]
    biggest= feature + '_' + str(mode)
    pos.drop([biggest], axis=1, inplace=True)
    df.drop([feature], axis=1, inplace=True)
    df=df.join(pos)
    return df


# In[18]:


def get_category (df, col, binsnum, labels, qcut= False):
    
    if qcut:
        localdf = pd.qcut(df[col], q = binsnum, labels = labels)
    else:
        localdf = pd.cut(df[col], bins = binsnum, labels = labels)
    
    localdf = pd.DataFrame(localdf)
    name='Classified' + '_' + col
    localdf[name] = localdf [col]
    df=df.join(localdf[name])
    df[name]=df[name].astype(object)
    return df


# In[ ]:





# In[19]:


# WOE and IV of Gender


# In[20]:


new_df_application['Gender']=new_df_application['Gender'].replace(['F','M'],[0,1])
print(new_df_application['Gender'].value_counts())
iv, data = calc_iv(new_df_application, 'Gender', 'target')
feature_names_df.loc[feature_names_df['Features']=='Gender', 'IV'] = iv
data.head()


# In[21]:


# WOE and IV of Car Posession


# In[22]:


new_df_application['Car_Possession']=new_df_application['Car_Possession'].replace(['N','Y'],[0,1])
print(new_df_application['Car_Possession'].value_counts())
iv, data = calc_iv(new_df_application, 'Car_Possession', 'target')
feature_names_df.loc[feature_names_df['Features']=='Car_Possession', 'IV'] = iv
data.head()


# In[23]:


# WOE and IV of House Posession


# In[24]:


new_df_application['Property posession']=new_df_application['Property posession'].replace(['N','Y'],[0,1])
print(new_df_application['Property posession'].value_counts())
iv, data = calc_iv(new_df_application, 'Property posession', 'target')
feature_names_df.loc[feature_names_df['Features']=='Property posession', 'IV'] = iv
data.head()


# In[25]:


# WOE and IV of Personal Phone Posession


# In[26]:


new_df_application['Personal Mobile Phone Posession']=new_df_application['Personal Mobile Phone Posession'].astype(str)
print(new_df_application['Personal Mobile Phone Posession'].value_counts(normalize=True, sort=False))
new_df_application.drop(new_df_application[new_df_application['Personal Mobile Phone Posession']=='nan'].index, inplace=True)
iv, data =calc_iv(new_df_application, 'Personal Mobile Phone Posession', 'target')
feature_names_df.loc[feature_names_df['Features']=='Personal Mobile Phone Posession','IV'] =iv
data.head()


# In[27]:


# WOE and IV of having an email


# In[28]:


print(new_df_application['Email'].value_counts(normalize=True, sort=False))
new_df_application['Email']=new_df_application['Email'].astype(str)
iv, data = calc_iv(new_df_application, 'Email', 'target' )
feature_names_df.loc[feature_names_df['Features']=='Email','IV']=iv
data.head(5)


# In[29]:


# WOE and IV of Work Phone Posession


# In[30]:


new_df_application['Work Phone Posession']=new_df_application['Work Phone Posession'].astype(int)
iv, data = calc_iv(new_df_application, 'Work Phone Posession', 'target' )
new_df_application.drop(new_df_application[new_df_application['Work Phone Posession']=='nan'].index, inplace=True)
feature_names_df.loc[feature_names_df['Features']=='Work Phone Posession','IV']=iv
data.head(5)


# In[31]:


# Condinuous Variable, here I will first use pd.qcut to categorize the data, then use pd.get_dummy function as defined above 
# as there are more than 2 different types of values in the feature after categorization


# In[32]:


# WOE and IV of Number of Children


# In[33]:


new_df_application.loc[new_df_application['How many children'] >= 2,'How many children']='2More'
print(new_df_application['How many children'].value_counts(sort=False))


# In[34]:


iv, data=calc_iv(new_df_application,'How many children','target')
feature_names_df.loc[feature_names_df['Features']=='How many children','IV']=iv
data.head()


# In[35]:


new_df_application = convert_dummy(new_df_application,'How many children')


# In[36]:


# WOE and IV of Annual Income


# In[37]:


new_df_application['Annual Income']=new_df_application['Annual Income'].astype(int)
new_df_application['Annual Income'] = new_df_application['Annual Income']/10000 
print(new_df_application['Annual Income'].value_counts(bins=10,sort=False))
new_df_application['Annual Income'].plot(kind='hist',bins=50,density=True)


# In[38]:


new_df_application = get_category(new_df_application,'Annual Income', 3, ["low","medium", "high"], qcut=True)
iv, data = calc_iv(new_df_application,'Classified_Annual Income','target')
feature_names_df.loc[feature_names_df['Features']=='Annual Income','IV']=iv
data.head()


# In[39]:


new_df_application = convert_dummy(new_df_application,'Classified_Annual Income')


# In[40]:


# WOE and IV of Age


# In[41]:


new_df_application['Age']=-(new_df_application['DAYS_BIRTH'])//365
print(new_df_application['Age'].value_counts(bins=10,normalize=True,sort=False))
new_df_application['Age'].plot(kind='hist',bins=20,density=True)


# In[42]:


new_df_application = get_category(new_df_application,'Age',5, ["lowest","low","medium","high","highest"])
iv, data = calc_iv(new_df_application,'Classified_Age','target')
feature_names_df.loc[feature_names_df['Features']=='DAYS_BIRTH','IV'] = iv
data.head()


# In[43]:


new_df_application = convert_dummy(new_df_application,'Classified_Age')


# In[44]:


# WOE and IV of Working Years


# In[45]:


new_df_application['Work_Years']=-(new_df_application['DAYS_EMPLOYED'])//365
new_df_application[new_df_application['Work_Years']<0]=np.nan
new_df_application['Work_Years'].fillna(new_df_application['Work_Years'].mean(), inplace=True)
new_df_application['Work_Years'].plot(kind='hist', bins=20, density=True)


# In[46]:


new_df_application=get_category(new_df_application, 'Work_Years', 5, ['lowest', 'low','medium','high','highest'])
iv, data= calc_iv(new_df_application, 'Classified_Work_Years', 'target')
feature_names_df.loc[feature_names_df['Features']=='DAYS_EMPLOYED', 'IV']=iv
data.head()


# In[47]:


new_df_application = convert_dummy(new_df_application,'Classified_Work_Years')


# In[48]:


# WOE and IV of Family Size


# In[49]:


new_df_application['Family Size'].value_counts(sort=False)


# In[50]:


new_df_application['Family Size']=new_df_application['Family Size'].astype(int)
new_df_application['Classified_famsize']=new_df_application['Family Size']
new_df_application['Classified_famsize']=new_df_application['Classified_famsize'].astype(object)
new_df_application.loc[new_df_application['Classified_famsize']>=3, 'Classified_famsize']='3more'
iv, data = calc_iv(new_df_application, 'Classified_famsize', 'target')
feature_names_df.loc[feature_names_df['Features']=='Family Size', 'IV']=iv
data.head()


# In[51]:


new_df_application = convert_dummy(new_df_application,'Classified_famsize')


# In[52]:


# WOE and IV of Income Type


# In[53]:


print(new_df_application['Types of Income'].value_counts(sort=False))
print(new_df_application['Types of Income'].value_counts(normalize=True, sort=False))
new_df_application.loc[new_df_application['Types of Income']=='Pensioner', 'Types of Income']='State servant'
new_df_application.loc[new_df_application['Types of Income']=='Student','Types of Income']='State servant'
iv, data =calc_iv(new_df_application,'Types of Income', 'target')
feature_names_df.loc[feature_names_df['Features']=='Types of Income', 'IV']=iv
data.head()
new_df_application=convert_dummy(new_df_application,'Types of Income')


# In[54]:


# WOE and IV of Occupation Type


# In[55]:


new_df_application.loc[(new_df_application['Occupation']=='Cleaning staff') | (new_df_application['Occupation']=='Cooking staff') | (new_df_application['Occupation']=='Drivers') | (new_df_application['Occupation']=='Laborers') | (new_df_application['Occupation']=='Low-skill Laborers') | (new_df_application['Occupation']=='Security staff') | (new_df_application['Occupation']=='Waiters/barmen staff'),'Occupation']='Laborwk'
new_df_application.loc[(new_df_application['Occupation']=='Accountants') | (new_df_application['Occupation']=='Core staff') | (new_df_application['Occupation']=='HR staff') | (new_df_application['Occupation']=='Medicine staff') | (new_df_application['Occupation']=='Private service staff') | (new_df_application['Occupation']=='Realty agents') | (new_df_application['Occupation']=='Sales staff') | (new_df_application['Occupation']=='Secretaries'),'Occupation']='officewk'
new_df_application.loc[(new_df_application['Occupation']=='Managers') | (new_df_application['Occupation']=='High skill tech staff') | (new_df_application['Occupation']=='IT staff'),'Occupation']='hightecwk'
print(new_df_application['Occupation'].value_counts())
iv, data=calc_iv(new_df_application,'Occupation','target')
feature_names_df.loc[feature_names_df['Features']=='Occupation','IV']=iv
data.head()


# In[56]:


new_df_application = convert_dummy(new_df_application,'Occupation')


# In[57]:


# WOE and IV of House Type


# In[58]:


iv, data = calc_iv(new_df_application,'Housing Type', 'target')
feature_names_df.loc[feature_names_df['Features']=='Housing Type', 'IV']=iv
data.head()


# In[59]:


new_df_application=convert_dummy(new_df_application,'Housing Type')


# In[60]:


# WOE and IV of Education Level


# In[61]:


new_df_application.loc[new_df_application['Education Level']=='Academic degree','Education Level']='Higher education'
iv, data=calc_iv(new_df_application,'Education Level','target')
feature_names_df.loc[feature_names_df['Features']=='Education Level','IV']=iv
data.head()


# In[62]:


new_df_application=convert_dummy(new_df_application,'Education Level')


# In[63]:


# WOE and IV of Marriage Condition


# In[64]:


new_df_application['Marital Status'].value_counts(normalize=True,sort=False)
iv, data=calc_iv(new_df_application,'Marital Status','target')
feature_names_df.loc[feature_names_df['Features']=='Marital Status','IV']=iv
data.head()


# In[65]:


new_df_application = convert_dummy(new_df_application,'Marital Status')


# In[66]:


#Putting all the importance value of different features together


# In[67]:


feature_names_df=feature_names_df.sort_values(by='IV',ascending=False)
feature_names_df.loc[feature_names_df['Features']=='DAYS_BIRTH','Features']='Age'
feature_names_df.loc[feature_names_df['Features']=='DAYS_EMPLOYED','Features']='Work_Year'
feature_names_df.loc[feature_names_df['Features']=='Annual Income','Features']='Income'
feature_names_df


# In[68]:


#Data preparation for ML


# In[69]:


new_df_application.columns


# In[70]:


Y=new_df_application['target']
x=new_df_application[['Gender', 'Property posession','How many children_1',
       'How many children_2More', 'Work Phone Posession', 'Classified_Age_high',
       'Classified_Age_highest', 'Classified_Age_low', 'Classified_Age_lowest', 
         'Classified_Work_Years_high','Classified_Work_Years_highest', 'Classified_Work_Years_low',
       'Classified_Work_Years_medium',  'Occupation_hightecwk', 
        'Occupation_officewk','Classified_famsize_1',
       'Classified_famsize_3more', 'Housing Type_Co-op apartment', 
        'Housing Type_Municipal apartment','Housing Type_Office apartment', 'Housing Type_Rented apartment',
       'Housing Type_With parents', 'Education Level_Higher education',
       'Education Level_Incomplete higher', 'Education Level_Lower secondary',  
        'Marital Status_Civil marriage', 'Marital Status_Separated',
       'Marital Status_Single / not married', 'Marital Status_Widow']]


# In[71]:


print(x.head(3))


# In[72]:


print(Y.value_counts()) #dataset is bias


# In[73]:


#Using Synthetic Minority Over-Sampling Technique(SMOTE) to deal with imbalance dataset problem


# In[74]:


Y = Y.astype('int')
x_balance,Y_balance = SMOTE().fit_resample(x,Y)
x_balance = pd.DataFrame(x_balance, columns = x.columns)


# In[75]:


x_balance_train,x_balance_test,Y_balance_train,Y_balance_test=train_test_split(x_balance, Y_balance, test_size=0.3,
random_state=42, stratify=Y_balance) #Stratify makes sure the portion of Y_balance is the same in test and training data


# In[76]:


#Feature Scaling


# In[77]:


sc=StandardScaler()
x_balance_train=sc.fit_transform(x_balance_train)


# In[78]:


x_balance_test=sc.transform(x_balance_test)


# In[79]:


#Using Logistic Regression ML


# In[104]:


LR= LogisticRegression(random_state=0)
LR.fit(x_balance_train, Y_balance_train)
Y_predict=LR.predict(x_balance_test)


# In[105]:


print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print('F1 Score is {:.5}'.format(f1_score(Y_balance_test, Y_predict)))
print('Precision Score is {:.5}'.format(precision_score(Y_balance_test, Y_predict)))
print('Recall Score is {:.5}'.format(recall_score(Y_balance_test, Y_predict)))


# In[106]:


print(pd.DataFrame(confusion_matrix(Y_balance_test, Y_predict)))
sns.set_style('white')
class_names=['0','1']


# In[107]:


#Using Random Forest ML


# In[108]:


RF=RandomForestClassifier(random_state=0)
RF.fit(x_balance_train, Y_balance_train)
Y_predict=RF.predict(x_balance_test)

print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print(pd.DataFrame(confusion_matrix(Y_balance_test,Y_predict)))


# In[109]:


print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print('F1 Score is {:.5}'.format(f1_score(Y_balance_test, Y_predict)))
print('Precision Score is {:.5}'.format(precision_score(Y_balance_test, Y_predict)))
print('Recall Score is {:.5}'.format(recall_score(Y_balance_test, Y_predict)))


# In[110]:


#Using Support Vector Classification


# In[111]:


S_V_C=SVC(random_state=0)
S_V_C.fit(x_balance_train, Y_balance_train)
Y_predict=S_V_C.predict(x_balance_test)


# In[112]:


print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print('F1 Score is {:.5}'.format(f1_score(Y_balance_test, Y_predict)))
print('Precision Score is {:.5}'.format(precision_score(Y_balance_test, Y_predict)))
print('Recall Score is {:.5}'.format(recall_score(Y_balance_test, Y_predict)))


# In[89]:


print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print(pd.DataFrame(confusion_matrix(Y_balance_test,Y_predict)))


# In[90]:


#Using XGBoost Classification


# In[91]:


XGB = XGBClassifier(random_state=0)
XGB.fit(x_balance_train, Y_balance_train)
Y_predict=XGB.predict(x_balance_test)


# In[92]:


print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print('F1 Score is {:.5}'.format(f1_score(Y_balance_test, Y_predict)))
print('Precision Score is {:.5}'.format(precision_score(Y_balance_test, Y_predict)))
print('Recall Score is {:.5}'.format(recall_score(Y_balance_test, Y_predict)))


# In[93]:


print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print(pd.DataFrame(confusion_matrix(Y_balance_test,Y_predict)))


# In[94]:


#Hyperparameter tuning for XGBoost as it has the highest accuracy


# In[95]:


params={'learning_rate':[0.45,0.5,0.55],
       'max_depth':[8.5,9,9.5],
       'min_child_weight':[0.8,1,1.2],
       'gamma':[0.05,0.1,0.15],
       'colsample_bytree':[0.55, 0.6,0.5]}


# In[96]:


grid_search=GridSearchCV(estimator=XGB, param_grid=params, cv=2, n_jobs=-1, verbose=2)
grid_search.fit(x_balance_train, Y_balance_train)
grid_search.best_params_


# In[113]:


XGB = XGBClassifier(random_state=0,colsample_bytree=0.55,
        gamma=0.05,learning_rate=0.45,max_depth=9,min_child_weight=1)


# In[114]:


params={'reg_alpha': [0.8,0.9,1], 'reg_lambda': [0.9,1,1.1],'n_estimators':[100,500,1000] }


# In[115]:


grid_search=GridSearchCV(estimator=XGB, param_grid=params, cv=2, n_jobs=-1, verbose=2)
grid_search.fit(x_balance_train, Y_balance_train)
grid_search.best_params_


# In[100]:


#Final performance evaluation


# In[116]:


XGB = XGBClassifier(n_estimators=1000, random_state=0,colsample_bytree=0.55,
        gamma=0.05,learning_rate=0.45,max_depth=9,min_child_weight=1,reg_alpha= 0.8, reg_lambda= 1)
XGB.fit(x_balance_train, Y_balance_train)
Y_predict=XGB.predict(x_balance_test)


# In[117]:


print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print('F1 Score is {:.5}'.format(f1_score(Y_balance_test, Y_predict)))
print('Precision Score is {:.5}'.format(precision_score(Y_balance_test, Y_predict)))
print('Recall Score is {:.5}'.format(recall_score(Y_balance_test, Y_predict)))


# In[118]:


print('Accuracy Score is {:.5}'.format(accuracy_score(Y_balance_test, Y_predict)))
print(pd.DataFrame(confusion_matrix(Y_balance_test,Y_predict)))


# In[ ]:




