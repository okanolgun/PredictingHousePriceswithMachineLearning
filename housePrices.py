#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The first 3 files is in same format, so we need to read them together. 
import pandas as pd 

fed_files = ["MORTGAGE30US.csv", 
             "RRVRUSQ156N.csv", 
             "CPIAUCSL.csv"]

dfs = [pd.read_csv(f, parse_dates=True, 
                   index_col=0) for f in fed_files]

#telling panda to parse any dates that it finds in the 
# csv file into a pandas's date time format
# and read all 3 files using this list comprehension 


# In[2]:


dfs[0] 

#weekly data


# dfs.head() can't work because of dfs is a list object.

# In[3]:


dfs[1]

#quarterly data (3 months)


# In[4]:


dfs[2]

#monthly data


# All of our datasets are on slightly different timelines. So we're going to need merge them. 

# In[5]:


fed_data = pd.concat(dfs, axis=1)

#combined them with .concat() method from pandas


# In[6]:


fed_data


# As you see, we have got some gaps and NaN values. It's because of our datasets are on different timelines. 

# In[7]:


fed_data.tail(50)


# In the table there are some blanks and NaN values. Because of our datasets' timelines are different like we said. We gotta fix this. 
# 
# For example, 'CPIAUCSL' column's timeline is monthly and after releasing a data, it becomes NaN untill the next release. 
# We need to arrange that a data should stay same untill the next data release. 
# 
# 

# In[8]:


fed_data = fed_data.ffill()


# In[9]:


fed_data.tail(50)


# We filled all the NaN values with the previous not-NaN ones. 

# In[10]:


zillow_files = ["Metro_median_sale_price_uc_sfrcondo_week.csv", 
                "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"] 

dfs = [pd.read_csv(f) for f in zillow_files]


# In[11]:


dfs[0] 


# We can see the Median Sale Price data frame. It's kinda strange than Federal Reserve data. In Federal Reserve data, each row was a date and the columns were economic indicators. 
# 
# But in that data frame, each row is a region in the United States. First row is the whole country, second row is NewYork..etc. 
# 
# We will only use the first line to make a prediction about the whole country.
# 

# In[12]:


dfs = [pd.DataFrame(df.iloc[0,5:]) for df in dfs] 

#This statement will pick first 5 columns and cut of them 
# from our dataset. We deleted that columns 
# because we'll not work with them


# In[13]:


dfs[0]


# Now, you can see we don't have first 5 rows we deleted. Our data is consistent with our federal reserve data. 
# 
# In first data set, we have a sales price for avarage house sold in that week. 

# In[14]:


dfs[1]


# Our second data set is the house value. We have the house value each week for houses across the United States. 

# In[15]:


for df in dfs : 
    df.index = pd.to_datetime(df.index) 
    #first we need to convert our 'string' 
    # format to a 'date' format with pandas
    # exp : '1996-01-31' turned to a 'date' format 
    
    df["month"] = df.index.to_period("M")
    #it's gonna take date and cut of the day part
    # that will give us a column that 
    # we can join both data frames 
    
    


# In[16]:


dfs[0]


# In[17]:


price_data= dfs[0].merge(dfs[1], on='month')


# In[18]:


price_data.index = dfs[0].index 

#make sure about two data frames' indexes are same


# In[19]:


price_data


# As you can see, we combined our data frames in the right format like we want. 
# 
# 0_x is the avarage sales price and 0_y is the avarage value of a house. 

# In[20]:


del price_data["month"] 


# In[21]:


price_data.columns = ["price","value"]


# In[22]:


price_data


# We deleted "month" column because won't use it. After deleting we renamed the 0_x and 0_y columns for make them more accessible.  

# In[23]:


fed_data = fed_data.dropna()


# In[24]:


# fed_data
# when you run that, you can see we had some datas 
# from 1940s but we don't have now. Becasuse we dropped Na values. 
# Now we have rows where we have only all three indicators.  

fed_data.tail(20)


# If you look at 13th and 14th rows you can see that our data's timeline is not weekly. Although 2 days has passed, we see data again. It's because of our federal reserve and zillow data are not releasing at the same day. 
# 

# In[25]:


from datetime import timedelta

fed_data.index = fed_data.index + timedelta(days=2) 


#'timedelta' is used to represent the time difference 
# between two dates or times and to perform 
# mathematical operations on this difference.


# In[26]:


fed_data


# All of our data turned weekly format. Now we can merge it with the zillow data because dates are matching and they are consistent. 
# 

# In[27]:


price_data = fed_data.merge(price_data, 
                            left_index=True, 
                            right_index=True)

#telling pandas to combine these two data frames 
# it saying use the index which is the date in 
# both data frames to actually merge them


# In[28]:


price_data


# Now we have a single data frame with both our federal reserve data and out zillow data. Our dates are matching and we haven't got any Na or null value. 

# In[29]:


price_data.columns = ["interest", "vacancy", "cpi", "price", "value"]


# In[30]:


price_data


# Renamed our columns names for make them more accessible and clear. 

# In[31]:


price_data.plot.line(y="price", use_index=True)


# This is house prices in the United States over time. It's looking like pretty sharp after 2012. We can say that inflation can get conflated with house price changes. For example inflation makes things more expensive. Because everything gets more expensive over time. 
# 
# But we want to do is take out inflation and just figure out the increase of the houses prices. We don't want our model to have to predict inflation and house prices. We just wanna predict the primary house prices. 
# 
# 
# ***Under normal conditions, if inflation rises, demand for housing falls, and prices do not rise as fast as inflation due to low demand. 

# In[32]:


price_data["adj_price"] = price_data["price"] / price_data["cpi"] * 100 

#created a adjusted price column from 
# dividing price by inflation measure. 
# multiplied cpi wth 100 bcs it's in units of 100 


# In[33]:


price_data.plot.line(y="adj_price", use_index=True)


# So you can see, it is still rising sharply because we've corrected for inflation. The magnitude of the change is not so extreme now. The changes are not so huge because we took inflation out of it. Before, from 2012 to 2022, house prices were 160000 to +350000 thousand dolars. But now, they are 70000 to 130000 thousand dolars. It is still sharp but because of taking out of inflation, price differences are not so different. 

# In[34]:


price_data["adj_value"] = price_data["value"] / price_data["cpi"] * 100 

# we did same thing for adjusted value 
# to find primary value without inflation


# In[35]:


price_data["next_quarter"] = price_data["adj_price"].shift(-13) 

price_data 


# We created a column which contains a data adjusted price of a house of 13 months into the future. That's why we can not see any data in last rows. Because there is no data from 13 month future/13 rows ahead. 

# In[36]:


price_data.dropna(inplace=True) 

#this code line equals with 
# prica_data = price_data.dropna() 
# with inplace, we can use like that


# In[37]:


price_data


# In[38]:


price_data["change"] = (price_data["next_quarter"] > 
                       price_data["adj_price"]).astype(int) 

#this cell says that, if the price goes down in next months, 
# the "change" column is '0', otherwise it is '1' 


# In[39]:


price_data


# In[40]:


price_data["change"].value_counts()


# We looked at how many 0s and 1s are in our column. It will be better for our machine learning model if these numbers are more or less close numbers. Because it will have an equal data set to make more accurate predictions. Data sets with more than one side may be a wrong factor for future predictions.

# In[41]:


predictors = ["interest", "vacancy", "adj_price", "adj_value"] 
target = "change" 

#We'll use 'predictors', these 4 columns to predict 
# 'target', I mean "change" column. 


# In[42]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import numpy as np 


# Random Forest is a basic model that we can use at any ML task. It is robust to overfitting. 
# 
# The Accuracy Score is used for to judge if our model is good or not. It will tell us what percentage of the time is our model correct. 
# 
# Numpy is a basic library for data analysis. 

# In[66]:


def predict(train, test, predictors, target) : 
    rf = RandomForestClassifier(min_samples_split=10,
                               random_state=1)
    rf.fit(train[predictors], train[target]) 
    preds = rf.predict(test[predictors])
    return preds 


# We created a method for prediction. 
# 
# Created a 'rf' object from RandomClassifier.
# 
# 'min_samples_split=10' protects our model from overfitting. telling the model not going too deep. 
# 
# 'random_satate=1', every time you run the model, its going to generate the same sequence of random numbers and its going to give you same result.
# 
# We fitted our model using datas.
# 
# We generated our predictions using test set. 

# In[ ]:





# Actually, this part is kinda hard to understand. We created a backtesting engine. 
# 
# There are some techniques like 'cross validation' that in order to validate a model across an entire set of data. I explain the 'cross validation' in details in the top of my article. 
# 
# But this doesn't work for time series data because we don't want to use data from the future to predict the past. So in time series data like this data set, if you use 'cross validation' technique,  you'll use data from 2022 to predict what happened in 2019. 
# 
# Of course that can make your model look really really good when you are training your model. 
# 
# Back testing is going to let us generate predictions for most of our data set, but do it in a way that respects the order of the data set. 

# In[44]:


START = 260 # 5 years worth of data
STEP = 52 # 1 year is 52 weeks 

def backtest(data, predictors, target) : 
    all_preds = [] 
    for i in range(START, data.shape[0], STEP) :
        train = price_data.iloc[:i] #everything until i 
        test = price_data.iloc[i:(i+STEP)] #the year following i 
        all_preds.append(predict(train, test, predictors, target)) 
        
    preds = np.concatenate(all_preds) 
    return preds, accuracy_score(data.iloc[START:][target], preds) 


# First we initialized a list called 'all_preds'. We're going to append all of our prediction sets to this list. 
# 
# For loop is going to start with 5 years of data so our data starts in 2008 so it's going to take all of the data from 2008 to 2013. And it's going to make predictions for the next year, for 2014.  
# 
# Then, it will take data from 2008 to 2014 and make predictions for 2015. And it will keep going like that. Until we have predictions for every year from 2014 through 2022.
# 
# We splitted our data up into train sets and test sets. Maked predictions with .predict() method. Concatenated them into our single array, an empty list. Then returned our predictions and our accuracy. 

# In[ ]:





# In[45]:


preds, accuracy = backtest(price_data, predictors, target) 


# In[46]:


accuracy


# This means that, when our model made a prediction the prediction was correct 59% of the time. Actually, this is a good accuracy score but we can still improve our score for model. 

# In[47]:


# for improving our accuracy, we gotta add a few more variables into to model
# that info will be the recent trend in house prices 


yearly = price_data.rolling(52, min_periods=1).mean() 
#.rolling() method: we'll pass in the min periods 
# which means that, even if we have one period of data
# pandas wont return a Na, if pandas has less than 52 weeks 
# of data from the current row, backwards it'll return a Na
# but min periods avoids that 


# In[48]:


yearly


# This basically gives us our averages for the past year. So the values here for April 9th 2022 are actually the average interest rate, the average rental vacancy rate...etc. 
# 
# And we want to find the ratio between the current value and the value in the past year. 

# In[51]:


yearly_ratios = [p + "_year" for p in predictors] 
#defined our column names in one line 
# with generating new variables names

price_data[yearly_ratios] = price_data[predictors] / yearly[predictors]
#Example: we taking our interest rate and dividing 
# it by the average interest rate in last year
# same for vacancy rate same for price ... etc 


# In[52]:


price_data


# You can see that, these are all ratios. If you look at the last row, the week of April 9th 2022, we can see the ratio for interest rates is 1.46. So the interest rates are much much higher than they were over the last year. 
# 
# Our model can use these table to predict the price trend much better. Because we're giving it more variables and more datas about trend. 

# In[53]:


preds, accuracy = backtest(price_data, predictors + yearly_ratios, target)


# In[54]:


accuracy


# As you can see, our score is a little higher. In other words, we made our model more optimal than before. We did this by giving our model more data that was evaluated from different angles.

# In[ ]:





# We need to plot out where the algorithm is making mistakes in its predictions. 

# In[55]:


pred_match = (preds == price_data[target].iloc[START:]) 

#this is comparing our predicted values to the actual values. 


# In[57]:


pred_match[pred_match == True] = "green"
pred_match[pred_match == False] = "red" 

#pred_match will be true when they are aligned 
#pred_match will be false when they aren't aligned 


# In[59]:


import matplotlib.pyplot as plt 

plot_data = price_data.iloc[START:].copy()
#just copying our actual values 

plot_data.reset_index().plot.scatter(x="index", y="adj_price", color=pred_match)
#this plot shows us green is when we made 
# predictions and they were correct 
#in other hand red is when we made 
# predictions and they were false 

#.reset_index(): turning the index into a column in 
# our data frame called "index"   

#color=pred_match: whenever you plot a point, 
# color it based on pred_match 


# We can see from the plot, the model recently hasn't been doing as weel when the market was going up but shifts suddenly to going down. 
# 
# So this can tell us that maybe adding in some predictors or data sets that explain when the market is about the shift. Like maybe news, articles or other economic data sets..etc. They could actually help to improve this model. 
# 
# This type of a plot can give us an opinion for how to make a model better. 

# In[63]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import permutation_importance
#this can help you to figure out which columns you can drop 
# or if you need to find additional data 

rf = RandomForestClassifier(min_samples_split=10, random_state=1)
# Create a RandomForestClassifier instance

rf.fit(price_data[predictors], price_data[target])
# Fit the model with your data

result = permutation_importance(rf, price_data[predictors], price_data[target], n_repeats=10, random_state=1)
# Perform permutation importance analysis
# it is checking to see which variables are the most important 
# to the random forest model. 


# In[64]:


result["importances_mean"]


# In[65]:


predictors


# These output shows us how important each variable was to our model. From zero to one. So the higher value is the more important column to our model.  
# 
# For example, 'adj_value' column was more important than 'interest', 'vacancy' and 'adj_price' columns for our model. 

# In[ ]:





# In[ ]:





# In[ ]:




