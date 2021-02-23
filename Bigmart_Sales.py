#!/usr/bin/env python
# coding: utf-8

# # Polynomial regression With Degree Two  is  the best performer and more generalised

# In[1]:


import seaborn as sns


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[3]:


df = pd.read_csv('C:\\Users\\sagar farkale\\Downloads\\bigmart_train.csv')


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


X = df.iloc[:,1:-1]
y = df['Item_Outlet_Sales']


# In[7]:


X.head()


# In[8]:


X_cat = X.iloc[:,[1,7,8,9]]
X_con = X.iloc[:,[0,2,4]]


# In[9]:


X_cat.isnull().sum()


# In[10]:


X_cat['Item_Fat_Content']=X_cat['Item_Fat_Content'].str.replace('low fat','Low')
X_cat['Item_Fat_Content']=X_cat['Item_Fat_Content'].str.replace('LF','Low')
X_cat['Item_Fat_Content']=X_cat['Item_Fat_Content'].str.replace('reg','Regular')
X_cat['Item_Fat_Content']=X_cat['Item_Fat_Content'].str.replace('Low Fat','Low')


# In[11]:


X_cat['Outlet_Size'].fillna(method = 'ffill',inplace = True)


# In[12]:


X_cat.isnull().sum()


# In[13]:


for i in X_cat.columns:
    print(X_cat[i].value_counts())
   


# In[14]:


from sklearn.preprocessing import LabelEncoder
X_label = LabelEncoder()
X_cat.iloc[:,0] = X_label.fit_transform(X_cat.iloc[:,0])
X_cat.iloc[:,1] = X_label.fit_transform(X_cat.iloc[:,1])
#X.iloc[:,4] = X_label.fit_transform(X.iloc[:,4])
X_cat.iloc[:,2] = X_label.fit_transform(X_cat.iloc[:,2])
X_cat.iloc[:,3] = X_label.fit_transform(X_cat.iloc[:,3])


# In[15]:


X_con.head()


# In[16]:


X_con['Item_Weight'].fillna(np.median(X_con['Item_Weight']),inplace = True)


# In[17]:


X_con['Item_Weight'] = (X_con['Item_Weight'] - np.mean(X_con['Item_Weight']))/X_con['Item_Weight'].std()
#X_con['Item_Visibility'] = (X_con['Item_Visibility'] - np.mean(X_con['Item_Visibility']))/X_con['Item_Visibility'].std()
X_con['Item_Visibility'] = np.sqrt(X_con['Item_Visibility'])
X_con['Item_MRP'] = (X_con['Item_MRP'] - np.mean(X_con['Item_MRP']))/X_con['Item_MRP'].std()


# In[18]:


X_con.head()


# In[19]:


core = pd.concat([X_con,(np.sqrt(df.iloc[:,-1]))/100],axis =1)
core.corr(method = 'pearson')


# In[20]:


from sklearn.feature_selection import f_oneway
from sklearn.feature_selection import SelectKBest


# In[21]:


model = SelectKBest(score_func=f_oneway,k=4)
m =model.fit(X_cat,np.sqrt(df.iloc[:,-1])/100)


# In[22]:


dfscores = pd.DataFrame(m.scores_,columns = ['Scores'])
dfcolumns = pd.DataFrame(X_cat.columns)
dfpvalues = pd.DataFrame(m.pvalues_)
frc = pd.concat([dfscores,dfcolumns,dfpvalues],axis=1)
frc.columns = ['Scores','Features' ,'Pvalues']


# In[23]:


frc


# In[61]:


X_main= pd.concat([X_cat['Outlet_Size'],X_cat['Outlet_Type'],X_con['Item_Visibility'],X_con['Item_MRP']],axis=1)


# In[62]:


X_main.head()


# In[63]:


X_main = pd.get_dummies(X_main,columns =['Outlet_Size','Outlet_Type'],drop_first = True)
X_main.head()


# In[64]:


X_main['Item_Visibility'] = X_main['Item_Visibility']**2


# In[65]:


X_main['Item_MRP'] = X['Item_MRP']


# In[66]:


X_main.head()


# In[67]:


type(X_main)


# In[68]:


y = df.iloc[:,[-1]].values
X_main = X_main.iloc[:,[0,1,4,5,6]].values


# In[69]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_main,y ,test_size = 0.25,random_state = 42)


# In[32]:


#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)


# In[70]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[71]:


y_pred = regressor.predict(X_test)


# In[72]:


sum((y_pred - y_test)**2)


# In[73]:


SS_Residual = sum((y_test-y_pred)**2)       
SS_Total = sum((y_test-np.mean(y_test))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[74]:


import statsmodels.api as sm
X1 = sm.add_constant(X_train)
result = sm.OLS(y_train, X1).fit()
print(result.summary())


# In[75]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly,y_train)


# In[76]:


linreg2 = LinearRegression()
linreg2.fit(X_poly,y_train)


# In[77]:


ypredP = linreg2.predict(poly_reg.fit_transform(X_test))


# In[78]:


sum((ypredP - y_test)**2)


# In[79]:


SS_Residual = sum((y_test-ypredP)**2)       
SS_Total = sum((y_test-np.mean(y_test))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[80]:


from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=100,random_state=0)
reg.fit(X_train,np.ravel(y_train,order='C'))
y_predR  = reg.predict(X_test)


# In[81]:


y_predR = (y_predR).reshape(-1,1)
sum(((y_predR) - y_test)**2)


# In[82]:


SS_Residual = sum((y_test-y_predR)**2)       
SS_Total = sum((y_test-np.mean(y_test))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[83]:


from xgboost import XGBRegressor
regc = XGBRegressor()
regc.fit(X_train,y_train)


# In[84]:


y_predX = regc.predict(X_test)


# In[85]:


y_predX = (y_predX).reshape(-1,1)
sum(((y_predX) - y_test)**2)


# In[86]:


SS_Residual = sum((y_test-y_predX)**2)       
SS_Total = sum((y_test-np.mean(y_test))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[87]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#y_train = sc_y.fit_transform(y_train)
X_test = sc_X.transform(X_test)
#y_test = sc_y.fit_transform(y_test)


# In[88]:


from sklearn.svm import SVR
regs = SVR(kernel = 'linear')
regs.fit(X_train,y_train.ravel())


# In[89]:


y_preds = regs.predict(X_test)


# In[90]:


y_preds = (y_preds).reshape(-1,1)
sum(((y_preds) - y_test)**2)


# In[91]:


SS_Residual = sum((y_test-y_preds)**2)       
SS_Total = sum((y_test-np.mean(y_test))**2)     
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_train.shape[1]-1)
print(r_squared, adjusted_r_squared)


# In[55]:


sample = pd.read_csv('C:\\Users\\sagar farkale\\Downloads\\bigmart_samplesub.csv')
sample.head()


# In[ ]:




