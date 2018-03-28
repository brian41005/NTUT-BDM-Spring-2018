
# coding: utf-8

# ## 103590450 四資四 馬茂源

# In[1]:


from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.utils import AnalysisException
import os, math, time


# In[2]:


conf = (SparkConf()
        #.setMaster('spark://10.100.5.182:7077')
        #.setMaster("local")
        .setAppName("hw1"))


# In[3]:


try:
    sc = SparkContext(conf=conf)
    sql_sc = SQLContext(sc)
except ValueError:
    pass


# In[4]:


try:
    data = sql_sc.read.csv('./household_power_consumption.txt', sep=';', header=True)
except AnalysisException:
    data = sql_sc.read.csv('hdfs:///bdm/hw1/household_power_consumption.txt', sep=';', header=True)


# In[5]:


data = data.drop('Date', 'Time', 'Sub_metering_1', 
        'Sub_metering_2', 
        'Sub_metering_3')


# In[6]:


data = (data.withColumn('Global_active_power', data.Global_active_power.cast('float'))
        .withColumn('Global_reactive_power', data.Global_reactive_power.cast('float'))
        .withColumn('Voltage', data.Voltage.cast('float'))
        .withColumn('Global_intensity', data.Global_intensity.cast('float')))


# In[7]:


data = data.dropna()


# 1. Output the **minimum**, **maximum**, and **count** of the columns:`Global_active_power`, `Global_reactive_power`, `Voltage`, and `Global_intensity`

# In[8]:


features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']


# In[9]:


summary = {k:{} for k in ['min', 'max', 'mean', 'stddev', 'count']}


# In[10]:


# count = data.count()
# summary['count'] = {f:count for f in features}


# In[11]:


# max__ = (data.rdd.map(lambda x: (x.Global_active_power, 
#                         x.Global_reactive_power,
#                         x.Voltage,
#                         x.Global_intensity))
# .reduce(lambda x, y: tuple(max(p) for p in zip(x, y))))
# min__ = (data.rdd.map(lambda x: (x.Global_active_power, 
#                         x.Global_reactive_power,
#                         x.Voltage,
#                         x.Global_intensity))
# .reduce(lambda x, y: tuple(min(p) for p in zip(x, y))))
# for i, f in enumerate(features):
#     summary['max'][f] = max__[i]
#     summary['min'][f] = min__[i]


# 2 . Output the **mean** and **standard deviation** of these columns
# 

# In[12]:


# sum_ = (data.rdd.map(lambda x: (x.Global_active_power, 
#                         x.Global_reactive_power,
#                         x.Voltage,
#                         x.Global_intensity))
#         .reduce(lambda x,y: tuple(x_n+y_n for x_n, y_n in zip(x, y))))
# for i, f in enumerate(features):
#     summary['mean'][f] = sum_[i]/summary['count'][f]


# In[13]:


# temp = (data.rdd.map(lambda x: ((x.Global_active_power-summary['mean']['Global_active_power'])**2, 
#                         (x.Global_reactive_power-summary['mean']['Global_reactive_power'])**2,
#                         (x.Voltage-summary['mean']['Voltage'])**2,
#                         (x.Global_intensity-summary['mean']['Global_intensity'])**2))
#         .reduce(lambda x, y: tuple(x_n+y_n for x_n, y_n in zip(x, y))))

# for i, f in enumerate(features):
#     summary['stddev'][f] = math.sqrt(temp[i] / (summary['count'][f]-1)) 


# In[14]:


t0 = time.time()


# In[15]:


descr = data.select(features).describe()
descr = (descr.withColumn('Global_active_power', descr['Global_active_power'].cast('float'))
            .withColumn('Global_reactive_power', descr['Global_reactive_power'].cast('float'))
            .withColumn('Voltage', descr['Voltage'].cast('float'))
            .withColumn('Global_intensity', descr['Global_intensity'].cast('float')))
for r in descr.collect():
    state_name = r.summary
    r = r.asDict()
    r.pop('summary')
    summary[state_name] = r


# In[16]:


for i in ['min', 'max', 'count', 'mean', 'stddev']:
    print('%6s: %s'%(i, summary[i]))
print("finding 'min', 'max', 'count', 'mean', 'stddev' cost:%.3fs"%(time.time()-t0))


# 3 . Perform **min-max normalization** on the columns to generate normalized output

# In[17]:


for f in features:
    min_ = summary['min'][f]
    max_ = summary['max'][f] 
    data = (data.withColumn('norm_' + f, 
                     (data[f] - min_) / (max_ - min_)))


# In[18]:


(data.select(['norm_'+f for f in features])
 .repartition(1)
 .write.mode("overwrite")
 .format("com.databricks.spark.csv")
 .option("header", "true")
 .save("norm.csv"))


# In[19]:


sc.stop()

