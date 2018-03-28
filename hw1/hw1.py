
# coding: utf-8

# ## 103590450 四資四 馬茂源

# In[1]:


from pyspark import SparkConf, SparkContext, SQLContext
import pyspark
import os
import pandas as pd


# In[2]:


conf = (SparkConf()
        #.setMaster('spark://10.100.5.182:7077')
        #.setMaster("local")
        .setAppName("hw1"))


# In[3]:


sc = SparkContext(conf=conf)
sql_sc = SQLContext(sc)


# In[14]:


# rdd = (sc.textFile('./household_power_consumption.csv')
#        .map(lambda line: line.split(';')))
# header = rdd.first()
# data = sql_sc.createDataFrame(rdd.filter(lambda line: line != header), header)
# data.show()


# In[15]:


# data = pd.read_csv('./household_power_consumption.csv', 
#                    delimiter=';', 
#                    dtype={'Date':str,
#                           'Time':str,
#                           'Global_active_power':str,
#                           'Global_reactive_power':str,
#                           'Voltage':str,
#                           'Global_intensity':str,
#                           'Sub_metering_1':str,
#                           'Sub_metering_2':str,
#                           'sub_metering_3':str,})
# data = sql_sc.createDataFrame(data)


# In[16]:


data = sql_sc.read.csv('./household_power_consumption.csv', sep=';', header=True)


# In[17]:


data.dtypes


# In[18]:


data = (data.withColumn('Date', data.Date.cast('date'))
        .withColumn('Time', data.Time.cast('timestamp'))
        .withColumn('Global_active_power', data.Global_active_power.cast('float'))
        .withColumn('Global_reactive_power', data.Global_reactive_power.cast('float'))
        .withColumn('Voltage', data.Voltage.cast('float'))
        .withColumn('Global_intensity', data.Global_intensity.cast('float'))
        .withColumn('Sub_metering_1', data.Sub_metering_1.cast('float'))
        .withColumn('Sub_metering_2', data.Sub_metering_2.cast('float'))
        .withColumn('Sub_metering_3', data.Sub_metering_3.cast('float')))
data.dtypes


# In[19]:


data = data.drop('Date', 'Time', 'Sub_metering_1', 
        'Sub_metering_2', 
        'Sub_metering_3')


# 1. Output the **minimum**, **maximum**, and **count** of the columns:`Global_active_power`, `Global_reactive_power`, `Voltage`, and `Global_intensity`
# 2. Output the **mean** and **standard deviation** of these columns
# 

# In[20]:


features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']


# In[21]:


summary = data.select(features).describe()
summary.show()


# In[22]:


minmax = summary.toPandas().set_index('summary').loc[['min', 'max']].apply(pd.to_numeric)


# 3 . Perform **min-max normalization** on the columns to generate normalized output

# In[23]:


for feature in features:
    min = minmax[feature].loc['min']
    max = minmax[feature].loc['max'] 
    (data.withColumn('norm_' + feature, (data[feature]-min)/(max-min))
         .select(feature, 'norm_' + feature)
         .show())


# In[24]:


sc.stop()

