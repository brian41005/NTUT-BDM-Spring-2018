
# coding: utf-8

# # 103590450 四資四 馬茂源

# In[26]:


from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.utils import AnalysisException
from pyspark.ml.feature import RegexTokenizer
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, StructType, StructField
import os, math, time


# In[27]:


result = {}


# In[28]:


conf = (SparkConf()
        #.setMaster('spark://10.100.5.182:7077')
        #.setMaster("local")
        .setAppName("hw1"))


# In[29]:


try:
    sc = SparkContext(conf=conf)
    sql_sc = SQLContext(sc)
except ValueError:
    pass


# In[30]:


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

# In[31]:


#news = sql_sc.read.csv(files['news'], sep=',', header=True)
news = (sql_sc.read.load(files['news'], format="csv",
                         schema=StructType([
                        StructField("IDLink", StringType(), False),
                        StructField("Title", StringType(), False),
                        StructField("Headline", StringType(), False),
                        StructField("Source", StringType(), False),
                        StructField("Topic", StringType(), False),
                        StructField("PublishDate", StringType(), False),
                        StructField("SentimentTitle", StringType(), False),
                        StructField("SentimentHeadline", StringType(), False),
                        StructField("Facebook", StringType(), False),
                        StructField("GooglePlus", StringType(), False),
                        StructField("LinkedIn", StringType(), False)]),
                         mode="DROPMALFORMED", header="true")
        .drop('IDLink')
        .drop('Source')
        .drop('SentimentTitle')
        .drop('SentimentHeadline')
        .drop('Facebook')
        .drop('GooglePlus')
        .drop('LinkedIn'))
news.show()


# In[32]:


# news = news.sample(False, 0.1, None)


# In[33]:


news_data = news.select(['title', 'headline' , 'topic', 'publishDate'])


# In[34]:


def wordTokenizer(data, columns):
    for c in columns:
        new_c = c + '_tokens'
        reTokenizer = RegexTokenizer(inputCol=c, 
                                     outputCol=new_c, 
                                     pattern='\\W', 
                                     minTokenLength=2)
        data = reTokenizer.transform(data)
        # data = data.withColumn(new_c, data[new_c].cast())
    return data


# In[35]:


col =  ['title', 'headline']
news_data = wordTokenizer(news_data, col)
# news_data = news_data.drop(col)
# news_data = news_data.select('title_tokens', 'headline_tokens', 
#                              'topic',  'publishDate')


# In[36]:


# news_data = news_data.withColumn('publishDate', 
#                                  udf(lambda tmp: tmp[:10] , StringType())
#                                  (news_data.publishDate))


# In[51]:


# news_data.take(2500)[0]


# ### In news data, count the words in two fields: ‘Title’ and ‘Headline’ respectively, and list the most frequent words according to the term frequency in descending order, in total, per day, and per topic, respectively

# In[ ]:


def word_count(data, column, n=10):
    return (news_data
         .select(column).rdd
         .map(lambda tokens: tokens[column])
         .flatMap(lambda tokens: tokens)
         .map(lambda word: (word, 1))
         .reduceByKey(lambda a, b: a + b)
         .sortBy(lambda w: w[1], ascending=False)
        .take(n))


# #### In total

# In[ ]:


word_count(news_data, 'title')


# In[ ]:


word_count(news_data, 'headline_tokens')


# In[ ]:


news_data.collect()


# ### In social feedback data, calculate the average popularity of each news by hour, and by day, respectively (for each platform)

# ###  In news data, calculate the sum and average sentiment score of each topic, respectively

# ### From subtask (1), for the top-100 frequent words per topic in titles and headlines, calculate their co-occurrence matrices (100x100), respectively. Each entry in the matrix will contain the co-occurrence frequency in all news titles and headlines, respectively
