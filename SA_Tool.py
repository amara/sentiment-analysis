#!/usr/bin/env python
# coding: utf-8

# In[356]:


#libraries needed
import streamlit as st
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


# In[357]:


df = pd.read_csv('dataset_twitter.csv',encoding='ISO-8859-1')
df.head()


# **Data** **Wrangling**

# **Data PreProcessing**

# In[358]:


import nltk,collections
from nltk.corpus import stopwords
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
eng_stop_words = list(stopwords.words('english'))


# In[359]:


emoji = list(UNICODE_EMOJI.keys())


# In[360]:


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
    
    clean_str = ''.join([c for c in text if ord(c) < 128])
    return clean_str


# In[361]:


# Generate a new column called 'Processed Tweets' by applying preprocessed tweets function to the 'Tweet' column.
df['Processed_Tweets'] = df['full_text'].apply(ProcessedTweets)


# **Sentiment Analysis**

# In[363]:


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


# In[364]:


# using the functions to get the polarity and sentiment

df['Polarity'] = df['Processed_Tweets'].apply(polarity)
df['Subjectivity'] = df['Processed_Tweets'].apply(subjectivity)
df['Sentiment'] = df['Polarity'].apply(sentimenttextblob)

sent = df['Sentiment'].value_counts()

#add colors
colors = ['#ff9999','#66b3ff','#99ff99']

#plot pie chart

fig = plt.figure(figsize=(6,6), dpi=100)
ax = plt.subplot(111)
sent.plot.pie(ax=ax, autopct='%1.1f%%', startangle=270, fontsize=12, label="", colors=colors)
st.pyplot(fig) 

X = df['Processed_Tweets'].values
y = df['Sentiment'].values
#break each tweet sentence into words
from nltk.stem.snowball import SnowballStemmer


sentences = []

for word in X:
    sentences.append(word)

lines = list()
for line in sentences:
    words = line.split()
    for w in words:
        lines.append(w)


#stemming all the words to their root word
stemmer = SnowballStemmer(language='english')
stem=[]
for word in lines:
    stem.append(stemmer.stem(word))
#removes stopwords (very common words in a sentence)
stem2 = []
for word in stem:
    if word not in eng_stop_words:
        stem2.append(word)
#creates a new dataframe for the stem and shows the count of the most used words
data = pd.DataFrame(stem2)
data=data[0].value_counts()

#plots the top 20 used words
data = data[:10]
with plt.style.context("rose-pine.mplstyle"):
    #data = data.nlargest(columns="Count", n = 10) 
    fig = plt.figure(figsize=(15,4))
    ax = sns.barplot(x=data.index,y=data.values, alpha=0.8)
    ax.set(xlabel = 'Word from Tweet')
    ax.set(ylabel = 'Count of words')
    plt.title('Top 10 Words Overall')
    st.pyplot(fig) 

#tokenization
from nltk.util import ngrams 
import functools

tokenized_tweet = df['Processed_Tweets'].apply(lambda x: list(ngrams(x.split(), 2)))

l = functools.reduce(lambda x, y: list(x)+list(y), zip(tokenized_tweet))

flatten = [item for sublist in l for item in sublist]
counts = collections.Counter(flatten).most_common()
df2 = pd.DataFrame.from_records(counts, columns=['Phrase', 'Count'])
df2['Phrase']= df2['Phrase'].apply(lambda x: ' '.join([w for w in x]))

with plt.style.context("rose-pine.mplstyle"):
    df2 = df2.nlargest(columns="Count", n = 10) 
    fig = plt.figure(figsize=(15,4))
    ax = sns.barplot(data=df2, x= "Phrase", y = "Count")
    ax.set(ylabel = 'Count')
    plt.title('Top 10 Occuring Bigrams')
    st.pyplot(fig) 


from PIL import Image
import plotly.express as px

cloud_image='mask/twitter_mask.png'

mask = np.array(Image.open(cloud_image))

#tweets_string = " ".join(cat.split()[1] for cat in df.text)
tweets_string = pd.Series(X).str.cat(sep=' ')
stopwords = set(STOPWORDS)

w_cloud = WordCloud(width = 7000, height = 5000,
                background_color ='white',
                stopwords = stopwords,
                mask = mask).generate(tweets_string)

# Display the generated Word Cloud
fig = px.imshow(w_cloud, interpolation='bilinear')
px.axis("off")
px.title('Word Cloud')
st.pyplot(fig)
