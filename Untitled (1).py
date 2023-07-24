#!/usr/bin/env python
# coding: utf-8

# # Netflix Stock Analysis

# # by Dalia Atef Abobakr

# In[2]:


#importing liberaries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import plot
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[3]:


#read dataset:
dataset= pd.read_csv(r"C:\Users\Mass\Downloads\NFLX.csv")


# In[4]:


dataset.head()


# In[23]:


dataset["Date"]=pd.to_datetime(dataset.Date)


# In[6]:


#get the number of rows and column:

dataset.shape


# In[7]:


#drop 'adj close' column:
dataset.drop("Adj Close",axis=1, inplace= True)


# In[27]:


dataset.head()


# In[29]:


dataset.isnull().sum()


# In[30]:


dataset.isna().any()


# In[31]:


dataset.info()


# In[32]:


dataset.describe()


# In[33]:


print(len(dataset))


# In[18]:


dataset['Year'] = pd.DatetimeIndex(dataset['Date']).year
dataset.head()


# In[20]:


px.line(dataset,x='Date',y='Open',title = 'Opening Price History',width=1000, height=600)


# In[21]:


px.line(dataset,x='Date',y='Close',title ='Close Price History',width=1000, height=600)


# In[22]:


dataset[["High","Low","Volume"]].plot(kind="box")


# In[23]:


fig = px.line(dataset, x='Date', y='Volume', title = 'Number of shares traded')
fig.show()


# In[25]:


fig = px.line(dataset, x='Year', y='Volume', title = 'Number of shares traded per year')
fig.show()


# In[26]:


layout = go.Layout(
    title="stock price of ntflix",
    xaxis=dict(
        title="Date",
        titlefont=dict(
            family="Courier New,monospace",
            size=18,
            color="#7f7f7f"
        )
    ),
    yaxis=dict(
        title="Price",
        titlefont=dict(
            family="Courier New,monospace",
            size=18,
            color="#7f7f7f"
        )
    )

)
ntflix_data=[{"x":dataset["Date"], "y":dataset["Close"]}]
plot = go.Figure(data=ntflix_data,layout=layout)


# In[27]:


iplot(plot)


# In[55]:


start_date = dataset.iloc[0][0]
end_date = dataset.iloc[-1][0]

print('Starting Date',start_date)
print('Ending Date',end_date)


# In[73]:


#build regression model
from sklearn.model_selection import train_test_split
#for preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#for model evaluation
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score



# In[88]:


#split the data into train and test sets
X=dataset[["Open","High","Low","Volume"]]
Y=dataset["Close"]
X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.3,random_state=101)


# In[89]:


X_train.shape


# In[90]:


X_test.shape


# In[60]:


#feature scaling
scaler = StandardScaler().fit(X_train)


# In[123]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# In[104]:


#creating linear model
regressor= LinearRegression()
regressor.fit(X_train,Y_train)


# In[105]:


print(regressor.coef_)


# In[108]:


print(regressor.intercept_)


# In[109]:


predicted=regressor.predict(X_test)


# In[110]:


print(X_test)


# In[111]:


predicted.shape


# In[112]:


d_frame=pd.DataFrame(Y_test,predicted)


# In[113]:


dfr=pd.DataFrame({"real_price":Y_test,"predicted_price":predicted})


# In[114]:


print(dfr)


# In[117]:


dfr.head(20)


# In[129]:


from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics


# In[130]:


regressor.score(X_test,Y_test)


# In[131]:


import math


# In[132]:


print("Mean Absolute Error:",metrics.mean_absolute_error(Y_test,predicted))


# In[133]:


print("Mean Squared Error:",metrics.mean_squared_error(Y_test,predicted))


# In[135]:


print("Root Mean Squared Error:",math.sqrt(metrics.mean_squared_error(Y_test,predicted)))


# In[136]:


graph=dfr.head(20)


# In[141]:


graph.plot(kind="bar")


# In[ ]:




