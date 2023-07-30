#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Data Cleaning

import string
from cmath import nan
from email.quoprimime import quote
from numpy import dtype
import pandas as pd
import re as re
from datetime import datetime
from re import search
import math
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

df_1 = pd.read_csv("Group_4.csv",usecols=[0], names=['created_at'])


def clean_date(dates):
    str(dates)
    dates = re.sub('{"created_at":', '', dates)
    dates = re.sub(r'\+0000', '', dates)
    dates = re.sub('Sun', '', dates)
    dates = re.sub('"', '', dates)
    return dates


df_1['created_at'] = df_1['created_at'].apply(clean_date)

df_1['created_at'] = pd.to_datetime(
    df_1['created_at'], format=' %b %d %H:%M:%S  %Y')


df_2 = pd.read_csv("Group_4.csv",usecols=[1], names=['id'])


def clean_id(id):
    id = re.sub('id:', '', id)
    return id


df_2['id'] = df_2['id'].apply(clean_id)
df_2['id'] = df_2['id'].astype('int64')


df = pd.read_csv("Group_4.csv",usecols=list(range(11, 233)), header=None)

user_id = []
name = []
protected = []
verified = []
foll_count = []
fren_count = []
fav_count = []
tot_tweets = []
created_at = []
location = []
quote_count = []
reply_count = []
retweet_count = []
tweet_fav_c = []

how = 0


for i, row in df.iterrows():
    j = 0
    while(row.values[j].split(":")[0] != "user"):
        j += 1
    user_id.append(row.values[j].split(":")[-1])

    j = 2
    while(row.values[j].split(":")[0] != "name"):
        j += 1
    name.append(row.values[j].split(":")[-1])

    j = 8

    while(str(row.values[j]).split(":")[0] != "protected"):
        j += 1
    protected.append(row.values[j].split(":")[-1])

    j = 9

    while(str(row.values[j]).split(":")[0] != "verified"):
        j += 1
    verified.append(row.values[j].split(":")[-1])

    j = 10
    while(str(row.values[j]).split(":")[0] != "followers_count"):
        j += 1
    foll_count.append(row.values[j].split(":")[-1])

    j = 11

    while(str(row.values[j]).split(":")[0] != "friends_count"):
        j += 1
    fren_count.append(row.values[j].split(":")[-1])

    j = 13

    while(str(row.values[j]).split(":")[0] != "favourites_count"):
        j += 1
    fav_count.append(row.values[j].split(":")[-1])

    j = 14

    while(str(row.values[j]).split(":")[0] != "statuses_count"):
        j += 1
    tot_tweets.append(row.values[j].split(":")[-1])

    j = 4

    while(str(row.values[j]).split(":")[0] != "location"):
        j += 1
    location.append(row.values[j].split(":")[-1])

    j = 15
    while(str(row.values[j]).split(":")[0] != "created_at"):
        j += 1
    date = str(row.values[j].split('"')[1])
    date = " ".join(date.split(" ")[1:])
    date = re.sub(r'\+0000', '', date)
    created_at.append(date)

    j = 10
    try:
        while(str(row.values[j]).split(":")[0] != "quote_count"):
            j += 1

        quote_count.append(row.values[j].split(":")[-1])
    except:
        quote_count.append("0")

    j = 10
    try:
        while(str(row.values[j]).split(":")[0] != "reply_count"):
            j += 1
        reply_count.append(row.values[j].split(":")[-1])
    except:
        reply_count.append("0")

    j = 10
    try:
        while(str(row.values[j]).split(":")[0] != "retweet_count"):
            j += 1
        retweet_count.append(row.values[j].split(":")[-1])
    except:
        retweet_count.append("0")

    j = 10
    try:
        while(str(row.values[j]).split(":")[0] != "favorite_count"):
            j += 1
        tweet_fav_c.append(row.values[j].split(":")[-1])
    except:
        tweet_fav_c.append("0")


df["User_id"] = user_id
df["Name"] = name
df["Protected?"] = protected
df["Verified"] = verified
df["Followers Count"] = foll_count
df["Friends Count"] = fren_count
df["Favourites Count"] = fav_count
df["Statuses Count"] = tot_tweets
df["Created At"] = created_at
df['Created At'] = pd.to_datetime(
    df['Created At'], format='%b %d %H:%M:%S  %Y')
df["Location"] = location
df["Quote_Count"] = quote_count
df["Reply_Count"] = reply_count
df["Retweet_Count"] = retweet_count
df["Tweet_Favourite_Count"] = tweet_fav_c


# Text Cleaning Code

df_text = pd.read_csv("Group_4.csv",usecols=[3], names=['tweet'])
# print(df_text)


def remove_lang(tweet):
    new_tweet = []
    for word in tweet.split():
        allowed = True
        for i in range(len(word)):
            if(word[i:i+2] == r'\u'):
                allowed = False
        if allowed:
            new_tweet.append(word)

    new_tweet = " ".join(new_tweet)

    return new_tweet


# print(df_text)


random = df_text['tweet'].apply(remove_lang)


def clean_tweet(tweet):
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    tweet = re.sub(r'#[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    tweet = re.sub(r'[^a-zA-Z0-9 ]', '', tweet)
    tweet = re.sub(r'http\S+', "", tweet)
    tweet = re.sub(r"www.\S+", "", tweet)
    tweet = re.sub("text", "", tweet)
    return tweet


def remove_punct(text):
    table = str.maketrans("", "", string.punctuation)
    return text.translate(table)


def remove_stopwords(text):
    s = ""
    str = text.split(" ")
    for word in str:
        if word.lower() not in stop:
            if s != "":
                s = s + " " + word
            else:
                s = word
    return s


df_text['tweet'] = df_text['tweet'].apply(remove_lang)
df_text['tweet'] = df_text['tweet'].apply(clean_tweet)

#!/usr/bin/env python
# coding: utf-8

#Analyzing Data

import pandas as pd
data = pd.read_csv("Final_Cleaned_Data.csv")
data.head()


# In[42]:


del data[data.columns[0]]
data.insert(8, "Tweet_Counter", 1)
data.max
data.head()


# In[43]:


#total tweets

len(data.index)


# In[44]:


data['Tweet_date'].sort_values()


# In[45]:


#number of retweets
data['Retweet_Count'].sum()


# In[46]:


#number of unique users
unique_users=data['User_id'].nunique()
unique_users


# In[47]:


#number of tweets with urls
int(data.Tweet.str.count('http').sum())


# In[48]:


#avg number of characters and words in a tweet
data['Characters'] = data['Tweet'].str.len()
data['Words'] = data['Tweet'].str.split().str.len()
data
mean_characters = data['Characters'].mean()
mean_words = data['Words'].mean()
print("Characters: ", round(mean_characters,2), "  Words: ", round(mean_words,2))


# In[49]:


#which twitter platform has been used to tweet
data['Source'].value_counts()


# In[50]:


#location for the most tweets
data['Location'].value_counts()


# In[51]:


#top 10 highest tweeters
data['User_id'].value_counts().head(10)


# In[52]:


data['User_id'].value_counts().tail(10)


# In[53]:


#names of top 10 tweeters
x=data['User_id'].groupby(data['Name']).value_counts().sort_values(ascending=False)
x
df = pd.DataFrame(x)
print(df)


# In[54]:


###Average number of tweets per user
tweets_sum=data['Tweet_Counter'].sum()
avg_tweets= round(tweets_sum/unique_users,2)
avg_tweets


# In[55]:


data['Protected'].value_counts()


# In[56]:


data['Verified'].value_counts()


# In[57]:


#locations of most fav-ed tweets
fav_loc=data[['Tweet_Favourite_Count', 'Location', 'Name','Tweet','Followers_Count']].sort_values('Tweet_Favourite_Count', ascending = False)
#top 05
fav_loc.head(10)


# In[58]:


fav_time=data[['Tweet_Favourite_Count', 'Tweet_date']].sort_values('Tweet_Favourite_Count', ascending = False)
fav_time.head(10)


# In[59]:


fav_foll=data[['Tweet_Favourite_Count', 'Followers_Count']].sort_values('Tweet_Favourite_Count', ascending = False)
fav_foll.head(10)


# In[60]:


fav_creat['Year'] = fav_creat['Created_At'].str.slice(4,10)
fav_creat.head(10)


# In[61]:


iphone = data.loc[data['Source'] == "iPhone"]
iphone['Location'].value_counts().sort_values(ascending=False)


# In[62]:


android = data.loc[data['Source'] == "Android"]
android['Location'].value_counts().sort_values(ascending=False)


# In[63]:


#Number of replies
x=data['Tweet'].str.count('@').sum()
print(x)


#Sentiment Analysis

def getSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

# Getting Polarity


def Polarity(tweet):
    return TextBlob(tweet).sentiment.polarity


# creating data frame for polarity and subjectivity
df_text['Subjectivity_Overall'] = df_text['tweet'].apply(getSubjectivity)
df_text['Polarity_Overall'] = df_text['tweet'].apply(Polarity)

allwords = ''.join([twts for twts in df_text['tweet']])
wordCloud = WordCloud(width=500, height=300, random_state=21,
                      max_font_size=119).generate(allwords)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
#plt.show()


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


df_text['Analysis'] = df_text['Polarity_Overall'].apply(getAnalysis)

#df_text.to_csv(r'C:\Users\Lenovo\Desktop\Cleaned_Text_Sentimenta_Analysis.csv')


#Wordcloud of positive words

df_positive = df_text[df_text['Analysis'] == 'Positive']
df_positive.rename(columns={'Polarity_Overall': 'Polarity_Positive',
                            'Subjectivity_Overall': 'Subjectivity_Positive'}, inplace=True)
# print(df_positive)

df_positive['Subjectivity_Positive'] = df_positive['tweet'].apply(
    getSubjectivity)
df_positive['Polarity_Positive'] = df_positive['tweet'].apply(Polarity)

allwords = ''.join([twts for twts in df_positive['tweet']])
wordCloud = WordCloud(width=500, height=300, random_state=21,
                      max_font_size=119).generate(allwords)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
#plt.show()

# Making Wordcloud of Negative Words

df_negative = df_text[df_text['Analysis'] == 'Negative']
df_negative.rename(columns={'Polarity_Overall': 'Polarity_Negative',
                            'Subjectivity_Overall': 'Subjectivity_Negative'}, inplace=True)
print(df_negative)

df_negative['Subjectivity_Negative'] = df_negative['tweet'].apply(
    getSubjectivity)
df_negative['Polarity_Negative'] = df_negative['tweet'].apply(Polarity)

allwords = ''.join([twts for twts in df_negative['tweet']])
wordCloud = WordCloud(width=500, height=300, random_state=21,
                      max_font_size=119).generate(allwords)

plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
#plt.show()


# print(df_text)

# df_text.to_csv(r'C:\Users\Lenovo\Desktop\File_Name_1.csv')


#Data Cleaning Cont.

source = []
df_tweet = pd.read_csv("Group_4.csv", usecols=list(range(4, 30)), header=None)

for i, row in df_tweet.iterrows():
    j = 0

    while True:

        try:
            while(row.values[j].split(":")[0] != "source" and row.values[j].split(":")[0] != ',source"'):
                j += 1
            break
        except:
            j += 1

    str = row.values[j].split(":")[-1]
    start = str.find("iPhone")
    lword = len("iPhone")
    if start == -1:
        start = str.find("Web App")
        lword = len("Web App")

    if start == -1:
        start = str.find("tweetdeck")
        lword = len("tweetdeck")
    if start == -1:
        start = str.find("ipad")
        lword = len("ipad")
    if start == -1:
        lo = "Android"
    else:
        lo = str[start: (start+lword)]
    source.append(lo)

df_tweet["Source"] = source


df2 = pd.DataFrame().assign(Tweet_id=df_2['id'], Tweet_date=df_1['created_at'], Source=df_tweet["Source"], Quote_Count=df["Quote_Count"], Reply_Count=df["Reply_Count"], Retweet_Count=df["Retweet_Count"], Tweet_Favourite_Count=df["Tweet_Favourite_Count"],Tweet=df_text['tweet'], User_id=df['User_id'], Name=df['Name'], Protected=df['Protected?'], Verified=df['Verified'], Followers_Count=df['Followers Count'],Friends_Count=df['Friends Count'], Favourites_Count=df['Favourites Count'], Statuses_Count=df['Statuses Count'], Created_At=df['Created At'],Location=df['Location'],Sentiment_Analysis=df_text['Analysis'])

# print(df2)


#df2.to_csv(r'C:\Users\Lenovo\Desktop\Final_Cleaned_Data_2.csv')




