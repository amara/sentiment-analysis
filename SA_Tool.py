#!/usr/bin/env python
# coding: utf-8

# In[1]:


#libraries needed
import pandas as pd
import snscrape.modules.twitter as sntwitter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
# nltk.download('stopwords') #run once and comment it out to avoid it downloading multiple times
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re
import textblob
from textblob import TextBlob

from wordcloud import WordCloud, STOPWORDS
from emot.emo_unicode import UNICODE_EMOJI


porter = PorterStemmer()

lemmatizer = WordNetLemmatizer() 

from wordcloud import ImageColorGenerator
from PIL import Image

import warnings
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('dataset_twitter.csv',encoding='ISO-8859-1')
df.head()


# **Data** **Wrangling**

# **Data PreProcessing**

# In[5]:


import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
eng_stop_words = list(stopwords.words('english'))


# In[6]:


emoji = list(UNICODE_EMOJI.keys())


# In[7]:


# function for preprocessing tweet in preparation for sentiment analysis
def ProcessedTweets(text):
    #changing tweet text to small letters
    text = text.lower()
    # Removing @ and links 
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split())
    # removing repeating characters
    text = re.sub(r'\@\w+|\#\w+|\d+', '', text)
    # removing punctuation and numbers
    punct = str.maketrans('', '', string.punctuation+string.digits)
    text = text.translate(punct)
    # tokenizing words and removing stop words from the tweet text
    tokens = word_tokenize(text)  
    filtered_words = [w for w in tokens if w not in eng_stop_words]
    filtered_words = [w for w in filtered_words if w not in emoji]
    # lemmetizing words
    lemmatizer = WordNetLemmatizer() 
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    text = " ".join(lemma_words)
    return text


# In[8]:


# Generate a new column called 'Processed Tweets' by applying preprocessed tweets function to the 'Tweet' column.
#df['Processed_Tweets'] = df['tweet'].apply(ProcessedTweets)
df['Processed_Tweets'] = df['full_text'].apply(ProcessedTweets)


# In[9]:


df.head(5)


# **Sentiment Analysis**

# In[10]:


# Function for polarity score
def polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

def subjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

# Function to get sentiment type
#setting the conditions
def sentimenttextblob(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive" 


# In[11]:


# using the functions to get the polarity and sentiment
df['Polarity'] = df['Processed_Tweets'].apply(polarity)
df['Subjectivity'] = df['Processed_Tweets'].apply(subjectivity)

df['Sentiment'] = df['Polarity'].apply(sentimenttextblob)

sent = df['Sentiment'].value_counts()
sent


# In[12]:


df.head(2)


# In[20]:


X = df['Processed_Tweets'].values
y = df['Sentiment'].values

