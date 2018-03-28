
# coding: utf-8

# ## 103590450 四資四 馬茂源

# In[44]:


from pyspark import SparkConf, SparkContext, SQLContext
#from pyspark.sql.functions import mean, max as max_, min as min_, stddev, count
import os, math


# In[45]:


conf = (SparkConf()
        #.setMaster('spark://10.100.5.182:7077')
        #.setMaster("local")
        .setAppName("hw1"))


# In[46]:


try:
    sc = SparkContext(conf=conf)
    sql_sc = SQLContext(sc)
except ValueError:
    pass


# In[47]:


data = sql_sc.read.csv('./household_power_consumption.csv', sep=';', header=True)


# In[48]:


data = data.drop('Date', 'Time', 'Sub_metering_1', 
        'Sub_metering_2', 
        'Sub_metering_3')


# In[49]:


data = (data.withColumn('Global_active_power', data.Global_active_power.cast('float'))
        .withColumn('Global_reactive_power', data.Global_reactive_power.cast('float'))
        .withColumn('Voltage', data.Voltage.cast('float'))
        .withColumn('Global_intensity', data.Global_intensity.cast('float')))


# In[50]:


data = data.dropna()


# 1. Output the **minimum**, **maximum**, and **count** of the columns:`Global_active_power`, `Global_reactive_power`, `Voltage`, and `Global_intensity`
# 2. Output the **mean** and **standard deviation** of these columns
# 

# In[51]:


features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']


# In[52]:


# def get_count(df, feature):
#     return df.select(feature).count()
# def get_minmax(df, feature):
#     return (data.select(feature)
#             .rdd
#             .sortBy(lambda x: x, ascending=False))


# In[53]:


count = data.count()


# In[54]:


summary = {k:{} for k in ['min', 'max', 'mean', 'stddev', 'count']}
summary['count'] = {f:count for f in features}


# In[55]:


max__ = (data.rdd.map(lambda x: (x.Global_active_power, 
                        x.Global_reactive_power,
                        x.Voltage,
                        x.Global_intensity))
.reduce(lambda x, y: tuple(max(p) for p in zip(x, y))))
min__ = (data.rdd.map(lambda x: (x.Global_active_power, 
                        x.Global_reactive_power,
                        x.Voltage,
                        x.Global_intensity))
.reduce(lambda x, y: tuple(min(p) for p in zip(x, y))))
for i, f in enumerate(features):
    summary['max'][f] = max__[i]
    summary['min'][f] = min__[i]


# In[56]:


sum_ = (data.rdd.map(lambda x: (x.Global_active_power, 
                        x.Global_reactive_power,
                        x.Voltage,
                        x.Global_intensity))
        .reduce(lambda x,y: tuple(x_n+y_n for x_n, y_n in zip(x, y))))
for i, f in enumerate(features):
    summary['mean'][f] = sum_[i]/summary['count'][f]


# In[57]:


temp = (data.rdd.map(lambda x: ((x.Global_active_power-summary['mean']['Global_active_power'])**2, 
                        (x.Global_reactive_power-summary['mean']['Global_reactive_power'])**2,
                        (x.Voltage-summary['mean']['Voltage'])**2,
                        (x.Global_intensity-summary['mean']['Global_intensity'])**2))
        .reduce(lambda x, y: tuple(x_n+y_n for x_n, y_n in zip(x, y))))

for i, f in enumerate(features):
    summary['stddev'][f] = math.sqrt(temp[i] / (summary['count'][f]-1)) 


# In[59]:


summary


# In[18]:


# summary = data.select(features).describe()
# summary.show()
# +-------+-------------------+---------------------+------------------+-----------------+
# |summary|Global_active_power|Global_reactive_power|           Voltage| Global_intensity|
# +-------+-------------------+---------------------+------------------+-----------------+
# |  count|            2049280|              2049280|           2049280|          2049280|
# |   mean| 1.0916150366540094|   0.1237144765251571|240.83985796672414|4.627759313004169|
# | stddev| 1.0572941611180013|  0.11272197958641254|  3.23998666120589|4.444396258981289|
# |    min|              0.076|                  0.0|             223.2|              0.2|
# |    max|             11.122|                 1.39|            254.15|             48.4|
# +-------+-------------------+---------------------+------------------+-----------------+


# 3 . Perform **min-max normalization** on the columns to generate normalized output

# In[20]:


for f in features:
    min_ = summary['min'][f]
    max_ = summary['max'][f] 
    data = (data.withColumn('norm_' + f, 
                     (data[f] - min_) / (max_ - min_)))


# In[21]:


data.select(['norm_'+f for f in features]).show()


# In[22]:


sc.stop()

