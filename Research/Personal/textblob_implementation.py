
# coding: utf-8

# In[1]:

from textblob.classifiers import NaiveBayesClassifier
import pandas as pd
import nltk
import re
from sklearn import *
import time
from textblob import TextBlob
import tweepy


# In[2]:

def dataClean(tweets_raw):
    cleanTweets = []
    for tweet in tweets_raw:
        tweet = tweet.lower() #convert to lowercase
        tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet) #remove URL
        tweet = re.sub(r'(\s)@\w+', r'', tweet) #remove usernames
        tweet = re.sub(r'@\w+', r'', tweet) #remove usernames
        tweet = re.sub('<[^<]+?>', '', tweet) #remove HTML tags
        tweet = re.sub(r'[<>!#@$:.,%\?-]+', r'', tweet) #remove punctuation and special characters 
        lower_case = tweet.lower() #tokenization 
        words = lower_case.split()
        tweet = ' '.join([w for w in words if not w in nltk.corpus.stopwords.words("english")]) #remove stopwords
        ps = nltk.stem.PorterStemmer()
        stemmedTweet = [ps.stem(word) for word in tweet.split(" ")]
        stemmedTweet = " ".join(stemmedTweet)
        tweet = str(stemmedTweet)
        tweet = tweet.replace("'", "")
        tweet = tweet.replace("\"","")
        cleanTweets.append(tweet)
    return cleanTweets


# In[3]:

trainingFile = "train.xlsx"
df_obama = pd.read_excel(trainingFile,sheetname='Obama')
df_romney = pd.read_excel(trainingFile,sheetname='Romney')

#Removing the mixed class and the !!! class

df_obama = df_obama[(df_obama['Class'].isin((1,-1,0)))]
df_romney = df_romney[(df_romney['Class'].isin((1,-1,0)))]

#creating lists for raw tweets and classes

obama_tweets_raw = df_obama['Anootated tweet']
obama_class = df_obama['Class']
romney_tweets_raw = df_romney['Anootated tweet']
romney_class = df_romney['Class']

obama_tweets_raw = obama_tweets_raw.tolist()
romney_tweets_raw = romney_tweets_raw.tolist()
obama_class_train = obama_class.tolist()
romney_class_train = romney_class.tolist()

romney_tweets = dataClean(romney_tweets_raw) #romney tweets cleaning
obama_tweets = dataClean(obama_tweets_raw) #obama tweets cleaning

obama_merged = zip(obama_tweets, obama_class_train)
obama_merged = list(obama_merged)

romney_merged = zip(romney_tweets, romney_class_train)
romney_merged = list(romney_merged)


# In[4]:

c1 = NaiveBayesClassifier(obama_merged)
c2 = NaiveBayesClassifier(romney_merged)


# In[5]:

testingFile = "test.xlsx"
df_obama_test = pd.read_excel(testingFile,sheetname='Obama')
df_romney_test = pd.read_excel(testingFile,sheetname='Romney')

#Removing the mixed class and the !!! class

df_obama_test = df_obama_test[(df_obama_test['Class'].isin((1,-1,0)))]
df_romney_test = df_romney_test[(df_romney_test['Class'].isin((1,-1,0)))]

#creating lists for raw tweets and classes

obama_tweets_raw_test = df_obama_test['Anootated tweet']
obama_class_test = df_obama_test['Class']
romney_tweets_raw_test = df_romney_test['Anootated tweet']
romney_class_test = df_romney_test['Class']

obama_tweets_raw_test = obama_tweets_raw_test.tolist()
romney_tweets_raw_test = romney_tweets_raw_test.tolist()
obama_class_test = obama_class_test.tolist()
romney_class_test = romney_class_test.tolist()

romney_tweets_test = dataClean(romney_tweets_raw_test) #romney tweets cleaning
obama_tweets_test = dataClean(obama_tweets_raw_test) #obama tweets cleaning

obama_merged_test = zip(obama_tweets_test, obama_class_test)
obama_merged_test = list(obama_merged_test)

romney_merged_test = zip(romney_tweets_test, romney_class_test)
romney_merged_test = list(romney_merged_test)


# In[7]:

pred_obama = []
for each in obama_merged_test:
    temp = c1.classify(str(each))
    pred_obama.append(temp)

pred_romney = []
for each in romney_merged_test:
    temp = c2.classify(str(each))
    pred_romney.append(temp)


# In[8]:

#obama
accScore = metrics.accuracy_score(obama_class_test,pred_obama)
labels = [1,-1]
precision = metrics.precision_score(obama_class_test,pred_obama,average=None,labels=labels)
recall = metrics.recall_score(obama_class_test,pred_obama,average=None,labels=labels)
f1Score = metrics.f1_score(obama_class_test,pred_obama,average=None,labels=labels)
print("Obama: \nOverall Acurracy: ",accScore,"\n")
lbl = ['positive', 'negative']
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision[i]))
    print("Recall of %s class: %f" %(lbl[i],recall[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1Score[i]),"\n")
#romney
accScore = metrics.accuracy_score(romney_class_test,pred_romney)
precision = metrics.precision_score(romney_class_test,pred_romney,average=None,labels=labels)
recall = metrics.recall_score(romney_class_test,pred_romney,average=None,labels=labels)
f1Score = metrics.f1_score(romney_class_test,pred_romney,average=None,labels=labels)
print("Romney:\nOverall Acurracy: ",accScore,"\n")
for i in range(2):
    print("Precision of %s class: %f" %(lbl[i],precision[i]))
    print("Recall of %s class: %f" %(lbl[i],recall[i]))
    print("F1-Score of %s class: %f" %(lbl[i],f1Score[i]),"\n")
'''
Output:

Obama:
Overall Acurracy:  0.580727831881

Precision of positive class: 0.543027
Recall of positive class: 0.628866
F1-Score of positive class: 0.582803

Precision of negative class: 0.600846
Recall of negative class: 0.619186
F1-Score of negative class: 0.609878

Romney:
Overall Acurracy:  0.516842105263

Precision of positive class: 0.582524
Recall of positive class: 0.311688
F1-Score of positive class: 0.406091

Precision of negative class: 0.554225
Recall of negative class: 0.819792
F1-Score of negative class: 0.661345
'''