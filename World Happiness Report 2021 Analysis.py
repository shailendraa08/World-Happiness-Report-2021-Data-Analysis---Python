#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


sns.set_style('darkgrid')
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = (10,7)
plt.rcParams['figure.facecolor'] = '#FFE5B4'


# In[5]:


data = pd.read_csv('world-happiness-report-2021.csv')


# In[7]:


data.head()


# In[12]:


data_columns = ['Country name', 'Regional indicator','Ladder score','Logged GDP per capita', 'Social support','Healthy life expectancy','Freedom to make life choices', 'Generosity', 'Perceptions of corruption']


# In[13]:


data = data[data_columns].copy()


# In[57]:


happy_df = data.rename(columns = {'Country name':'country_name','Regional indicator':'regional_indicator','Ladder score':'happiness_score','Logged GDP per capita':'logged_GDP_per_capita','Social support':'social_support','Healthy life expectancy':'healthy_life_expectancy','Freedom to make life choices':'freedom_to_make_life_choices','Generosity':'generosity','Perceptions of corruption':'perception_of_corruption'})


# In[58]:


happy_df.head()


# In[59]:


happy_df.isnull().sum()


# In[60]:


# Plot between happiness and GDP

plt.rcParams['figure.figsize'] = (15,7)
plt.title('Plot between Happiness Score and GDP')
sns.scatterplot(x = happy_df.happiness_score, y = happy_df.logged_GDP_per_capita, hue = happy_df.regional_indicator, s =200);

plt.legend(loc = 'upper left', fontsize = '10')
plt.xlabel('Happiness Score')
plt.ylabel('GDP per capita')


# In[61]:


gdp_region = happy_df.groupby('regional_indicator')['logged_GDP_per_capita'].sum()
gdp_region


# In[62]:


gdp_region.plot.pie(autopct = '%1.1f%%')
plt.title('GDP by Region')
plt.ylabel('')


# In[63]:


# Total Countries
total_country = happy_df.groupby('regional_indicator')[['country_name']].count()
print(total_country)


# In[68]:


# # Correlation Map

# cor = happy_df.corr(method = "pearson")
# f, ax = plt.subplots(figsize = (10, 5))
# sns.heatmap(cor, mask = np.zeros_like(cor, dtype = np.bool),
#            cmap="Blues", square=True, ax=ax)


# In[ ]:





# In[66]:


# Select only numeric columns
numeric_columns = happy_df.select_dtypes(include=[np.number])

# Calculate correlation matrix
cor = numeric_columns.corr(method="pearson")

# Plot correlation matrix
f, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(cor, cmap="Blues", square=True, ax=ax)


# In[70]:


# corruption in regions

corruption = happy_df.groupby('regional_indicator')[['perception_of_corruption']].mean()
corruption


# In[72]:


plt.rcParams['figure.figsize'] = (12,8)
plt.title('Perception of Corruption in various Regions')
plt.xlabel('Regions', fontsize = 15)
plt.ylabel('Corruption Index', fontsize = 15)
plt.xticks(rotation = 30, ha='right')
plt.bar(corruption.index, corruption.perception_of_corruption)


# In[76]:


top_10 = happy_df.head(10)
bottom_10 = happy_df.tail(10)


# In[78]:


fig, axes = plt.subplots(1,2, figsize = (16,6))
plt.tight_layout(pad= 2)
xlabels = top_10.country_name
axes[0].set_title('Top 10 happiest countries Life Expectancy')
axes[0].set_xticklabels(xlabels, rotation=45, ha='right')
sns.barplot(x= top_10.country_name, y= top_10.healthy_life_expectancy, ax=  axes[0])
axes[0].set_xlabel('Country Name')
axes[0].set_ylabel('Life Expectancy')

xlabels = bottom_10.country_name
axes[1].set_title('Bottom 10 happiest countries Life Expectancy')
axes[1].set_xticklabels(xlabels, rotation=45, ha='right')
sns.barplot(x= bottom_10.country_name, y= bottom_10.healthy_life_expectancy, ax=  axes[1])
axes[1].set_xlabel('Country Name')
axes[1].set_ylabel('Life Expectancy')


# In[82]:


country = happy_df.sort_values(by='perception_of_corruption').tail(10)
plt.rcParams['figure.figsize'] = (12,6)
plt.title('Countries with Most Percetion of Corruption')
plt.xlabel('Country', fontsize = 13)
plt.ylabel('Corruption Index', fontsize = 13)
plt.xticks(rotation = 30, ha = 'right')
plt.bar(country.country_name, country.perception_of_corruption)


# In[84]:


# corruption vs happiness

plt.rcParams['figure.figsize']=(15,7)
sns.scatterplot(x = happy_df.happiness_score, y=happy_df.perception_of_corruption, hue=happy_df.regional_indicator, s=200)
plt.legend(loc='lower left', fontsize = '14')
plt.xlabel('Happiness Score')
plt.ylabel('Corruption')


# In[ ]:




