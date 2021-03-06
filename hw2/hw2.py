
# coding: utf-8

# # 103590450 四資四 馬茂源

# ![](1.png)
# ![](2.png)
# ![](3.png)
# ![](4.png)
# ![](5.png)

# In[1]:


from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.utils import AnalysisException
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql.functions import udf, mean
from pyspark.sql.types import StringType, StructType, StructField, FloatType
import pandas as pd
import numpy as np
import os, math, time
import itertools
import csv


# In[2]:


if not os.path.exists('result'):
    os.mkdir('result')
for i in range(1, 5):
    dir_ = 'result/task{}'.format(i)
    if not os.path.exists(dir_):
        os.mkdir(dir_)


# In[3]:


t0 = time.time()


# In[4]:


conf = (SparkConf()
        #.setMaster('spark://10.100.5.182:7077')
        #.setMaster("local")
        .setAppName("hw2"))


# In[5]:


try:
    sc = SparkContext(conf=conf)
    sql_sc = SQLContext(sc)
except ValueError:
    pass


# In[6]:


files = {'fb':['Facebook_Economy.csv', 
               'Facebook_Obama.csv', 
               'Facebook_Palestine.csv', 
               'Facebook_Microsoft.csv'],
        'google':['GooglePlus_Obama.csv', 
                  'GooglePlus_Palestine.csv', 
                  'GooglePlus_Economy.csv', 
                  'GooglePlus_Microsoft.csv'],
        'linkedin':['LinkedIn_Microsoft.csv', 
                    'LinkedIn_Palestine.csv',
                    'LinkedIn_Obama.csv', 
                    'LinkedIn_Economy.csv'],
        'news':'News_Final.csv'}


# ## Preprocessing

# * IDLink (numeric): Unique identifier of news items
# * Title (string): Title of the news item according to the official media sources
# * Headline (string): Headline of the news item according to the official media sources
# * Source (string): Original news outlet that published the news item
# * Topic (string): Query topic used to obtain the items in the official media sources
# * PublishDate (timestamp): Date and time of the news items' publication
# * SentimentTitle (numeric): Sentiment score of the text in the news items' title
# * SentimentHeadline (numeric): Sentiment score of the text in the news items' headline
# * Facebook (numeric): Final value of the news items' popularity according to the social media source Facebook
# * GooglePlus (numeric): Final value of the news items' popularity according to the social media source Google+
# * LinkedIn (numeric): Final value of the news items' popularity according to the social media source LinkedIn

# In[7]:


def read_csv(file_name):
    try:
        data = sql_sc.read.csv(file_name, 
                       sep=',', 
                       header=True, 
                       mode='DROPMALFORMED')
    except AnalysisException:
        data = sql_sc.read.csv('hdfs:///bdm/hw2/{}'.format(file_name), 
                       sep=',', 
                       header=True, 
                       mode='DROPMALFORMED')
    return data


# In[8]:


news = read_csv(files['news'])
# news = (sql_sc.read.load(files['news'], 
#                          format="csv", 
#                          schema=StructType([StructField("IDLink", StringType(), False),
#                                             StructField("Title", StringType(), False),
#                                             StructField("Headline", StringType(), False),
#                                             StructField("Source", StringType(), False),
#                                             StructField("Topic", StringType(), False),
#                                             StructField("PublishDate", StringType(), False),
#                                             StructField("SentimentTitle", StringType(), False),
#                                             StructField("SentimentHeadline", StringType(), False),
#                                             StructField("Facebook", StringType(), False),
#                                             StructField("GooglePlus", StringType(), False),
#                                             StructField("LinkedIn", StringType(), False)]),
#                          mode="DROPMALFORMED", 
#                          header="true")
#         .drop('IDLink')
#         .drop('Source')
#         .drop('SentimentTitle')
#         .drop('SentimentHeadline')
#         .drop('Facebook')
#         .drop('GooglePlus')
#         .drop('LinkedIn'))


# In[9]:


# news = news.sample(False, 0.01, 42)


# In[10]:


news.count()


# In[11]:


news = news.withColumn('SentimentScore', (news.SentimentTitle+news.SentimentHeadline))


# In[12]:


news_data = (news.dropna()
                .select('title', 
                        'headline', 
                        'topic', 
                        'publishDate',
                        'SentimentScore'))


# In[13]:


def wordTokenizer(data, columns):
    for c in columns:
        new_c = c + '_tokens'
        reTokenizer = RegexTokenizer(inputCol=c, 
                                     outputCol=new_c, 
                                     pattern='\\W', 
                                     minTokenLength=2)
        data = reTokenizer.transform(data)
    return data


# In[14]:


col =  ['title', 'headline']
news_data = wordTokenizer(news_data, col)
news_data = news_data.select('title_tokens', 
                             'headline_tokens', 
                             'topic',  
                             'publishDate',
                             'SentimentScore')


# In[15]:


news_data = news_data.withColumn('publishDate', 
                                 udf(lambda tmp: tmp[:10] , StringType())
                                 (news_data.publishDate))


# In[16]:


news_data.count()


# In[17]:


news.count()


# In[18]:


# news_data = news_data.dropna()
# news_data.show()


# In[19]:


news_data.count()


# ### In news data, count the words in two fields: ‘Title’ and ‘Headline’ respectively, and list the most frequent words according to the term frequency in descending order, in total, per day, and per topic, respectively

# In[20]:


def word_count_total(data, column, n=10):
    return (news_data.select(column)
            .rdd
            .flatMap(lambda tokens: tokens[column])
            .map(lambda word: (word, 1))
            .reduceByKey(lambda a, b: a + b)
            .sortBy(lambda w: w[1], ascending=False)
            .take(n))


# In[21]:


task1_file = open('result/task1/output.txt', 'w', encoding='utf-8', newline='\n')
task1_output = []


# #### In total

# In[22]:


task1_output.append('[title top-frequent words in total]')
for r in word_count_total(news_data, 'title_tokens', n=100):
    task1_output.append(r)


# In[23]:


task1_output.append('\n[headline top-frequent words in total]')
for r in word_count_total(news_data, 'headline_tokens', n=100):
    task1_output.append(r)


#  #### per day

# In[24]:


def sort(tokens):
    take = 100 if len(tokens) >= 100 else len(tokens)
    return sorted(tokens, key=lambda x: x[1], reverse=True)[:take]


# In[25]:


def word_count_per(data, column, per_col, take=-1):
#     rdd = (news_data.select(column, per_col)
#             .rdd
#             .flatMap(lambda row: [((row[per_col], w), 1) for w in row[column]])
#             .reduceByKey(lambda a, b: a + b)
#             .map(lambda pair: (pair[0][0], (pair[0][1], pair[1])))
#             .reduceByKey(lambda a, b: a if a[1] > b[1] else b)
#             .sortBy(lambda w: w[1][1], ascending=False)
#             )
    rdd = (news_data.select(column, per_col)
        .rdd
        .flatMap(lambda row: [((row[per_col], w), 1) for w in row[column]])
        .reduceByKey(lambda a, b: a + b)
        .map(lambda pair: (pair[0][0], (pair[0][1], pair[1])))
        .groupByKey()
        .map(lambda topic: (topic[0], sort(topic[1])))
        .sortBy(lambda w: w[1][1], ascending=False)
        )
    if take == -1:
        return rdd.collect()
    else:
        return rdd.take(take)


# In[26]:


task1_output.append('\n[title top-frequent words per day]')
for r in sorted(word_count_per(news_data, 'title_tokens', 'publishDate'), key=lambda x: x[0]):
    task1_output.append(r)


# In[27]:


task1_output.append('\n[headline top-frequent words per day]')
for r in sorted(word_count_per(news_data, 'headline_tokens', 'publishDate'), key=lambda x: x[0]):
    task1_output.append(r)


# #### per topic

# In[28]:


task1_output.append('\n[title top-frequent words per topic]')
for r in word_count_per(news_data, 'title_tokens', 'topic'):
    task1_output.append(r)


# In[29]:


task1_output.append('\n[headline top-frequent words per topic]')
for r in word_count_per(news_data, 'headline_tokens', 'topic'):
    task1_output.append(r)


# In[30]:


task1_file.writelines(['{}\n'.format(r) for r in task1_output])
task1_file.close()


# ### In social feedback data, calculate the average popularity of each news by hour, and by day, respectively (for each platform)

# In[31]:


def create_social_data(data, files):
    for f in files:
        df = read_csv(f)
        data = data.union(df) if data else df
    data = data.withColumn('IDLink', data['IDLink'].cast('int'))
    for i in range(1, 144+1):
        col_name = 'TS{}'.format(i)
        data = data.withColumn(col_name, data[col_name].cast('int'))
    return data


# In[32]:


fb_social_data = google_social_data = linkedin_social_data = None 


# In[33]:


fb_social_data = create_social_data(fb_social_data, files['fb'])
google_social_data = create_social_data(google_social_data, files['google'])
linkedin_social_data = create_social_data(linkedin_social_data, files['linkedin'])


# In[34]:


fb_social_data = fb_social_data.dropna()
google_social_data = google_social_data.dropna()
linkedin_social_data = linkedin_social_data.dropna()


# In[35]:


def get_avg(seq):
    sum_ = np.sum(seq)
    return sum_/48, sum_/2


# In[36]:


def avg_popu(data, by=3):
    return (data.rdd
            .map(lambda r: (r['IDLink'],  get_avg(r[1:])))
            .sortByKey()
            .collect())


# In[37]:


fb_avg_by_hour_and_day = avg_popu(fb_social_data)


# In[38]:


google_avg_by_hour_and_day = avg_popu(google_social_data)


# In[39]:


linkedin_avg_by_hour_and_day = avg_popu(linkedin_social_data, by=3)


# In[40]:


avg_popularity = {'fb':fb_avg_by_hour_and_day,
                 'google':google_avg_by_hour_and_day,
                 'linkedin':linkedin_avg_by_hour_and_day}


# In[41]:


def save_csv(file_name, data):
    with open(file_name, 'w', 
              encoding='utf-8', newline='\n') as csvfile:
        fieldnames = ['IDLink', 'avg_popularity']
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fieldnames)
        writer.writerows(data)


# In[42]:


for platform, data in avg_popularity.items():
    rows_by_hour = []
    rows_by_day = []
    
    for ID, (avg_by_hour, avg_by_day) in data:
        rows_by_hour.append((ID, avg_by_hour))
        rows_by_day.append( (ID, avg_by_day))
        
    save_csv('./result/task2/{}_avg_popularity_by_hour.csv'.format(platform), 
             rows_by_hour)
    save_csv('./result/task2/{}_avg_popularity_by_day.csv'.format(platform), 
             rows_by_day)


# ###  In news data, calculate the sum and average sentiment score of each topic, respectively

# In[43]:


task3_file = open('result/task3/output.txt', 'w', encoding='utf-8', newline='\n')
task3_output = []


# In[44]:


sum_of_score = (news.select('SentimentScore', 'topic')
                .rdd
                .map(lambda r: (r['topic'], (r['SentimentScore'], 1)))
                .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
                .map(lambda x: (x[0], x[1][0], (x[1][0]/x[1][1])))
#                 .groupByKey()
#                 .map(lambda topic: (topic[0], 
#                                     np.sum(list(topic[1])), 
#                                     np.mean(list(topic[1]))))
                .collect())


# In[45]:


task3_output.append('[sum sentiment score of each topic]')
for topic, sum_, avg in sum_of_score:
    print('{:10s}, {:.3f}, {:.6f}'.format(topic, sum_, avg))
    task3_output.append('{:10s}, {:.3f}'.format(topic, sum_))


# In[46]:


task3_output.append('[avg sentiment score of each topic]')
for topic, sum_, avg in sum_of_score:
    task3_output.append('{:>10s}, {:.6f}'.format(topic, avg))


# In[47]:


task3_file.writelines(['{}\n'.format(r) for r in task3_output])
task3_file.close()


# ### From subtask (1), for the top-100 frequent words per topic in titles and headlines, calculate their co-occurrence matrices (100x100), respectively. Each entry in the matrix will contain the co-occurrence frequency in all news titles and headlines, respectively

# In[48]:


fw_all = {'title_tokens':{topic:[w[0] for w in top] 
                          for topic, top in word_count_per(news_data, 'title_tokens', 'topic')}, 
         'headline_tokens':{topic:[w[0] for w in top] 
                          for topic, top in word_count_per(news_data, 'headline_tokens', 'topic')}}


# In[49]:


def counter(vocabulary, tokens):
    return  [int(tokens.count(v) > 0) for v in vocabulary]


# In[50]:


for col_name, v in fw_all.items():
    for topic, vocabulary in v.items():
        print('column name:{}, topic:{}'.format(col_name, topic))
        
        X = np.array(news_data.select(col_name, 'topic')
                     .rdd
                     .filter(lambda r: r['topic'] == topic)
                     .map(lambda r:counter(vocabulary, r[col_name]))
                     .collect(), dtype='int64')
        co_occ = X.T.dot(X)
        # np.fill_diagonal(co_occ, 0)
        df = pd.DataFrame(data=co_occ, columns=vocabulary, index=vocabulary)
        # display(df)
        df.to_csv('result/task4/{}_{}_matrix.csv'.format(col_name, topic))


# In[51]:


print('cost {:.2f} minutes'.format((time.time()-t0)/60))

