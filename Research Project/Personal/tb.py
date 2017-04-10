import pandas as pd
import nltk
import re
from sklearn import *
import time
from textblob import TextBlob
import tweepy

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

def getPrediction(tweets):
    preds = []
    for each in tweets:
        analysis = TextBlob(each)
        sentiment = analysis.sentiment.polarity
        if sentiment >0:
            label = 1
            preds.append(label)
        else:
            label = -1
            preds.append(label)
    return preds

def getScores(original_class,preds):
    accScore = metrics.accuracy_score(original_class,preds)
    labels = [1,-1]
    precision = metrics.precision_score(original_class,preds,average=None,labels=labels)
    recall = metrics.recall_score(original_class,preds,average=None,labels=labels)
    f1Score = metrics.f1_score(original_class,preds,average=None,labels=labels)
    print("\nOverall Acurracy: ",accScore,"\n")
    lbl = ['positive', 'negative']
    for i in range(2):
        print("Precision of %s class: %f" %(lbl[i],precision[i]))
        print("Recall of %s class: %f" %(lbl[i],recall[i]))
        print("F1-Score of %s class: %f" %(lbl[i],f1Score[i]),"\n")

#Loading the sheets into data frames

trainingFile = "train.xlsx"
df_obama = pd.read_excel(trainingFile,sheetname='Obama')
df_romney = pd.read_excel(trainingFile,sheetname='Romney')

#Removing the mixed class and the !!! class

df_obama = df_obama[(df_obama['Class'].isin((1,-1)))]
df_romney = df_romney[(df_romney['Class'].isin((1,-1)))]

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

preds_obama = getPrediction(obama_tweets)
preds_romney = getPrediction(romney_tweets)

print("Obama:");getScores(obama_class_train,preds_obama)
print("Romney:");getScores(romney_class_train,preds_romney)

'''
Output:
Obama:

Overall Acurracy:  0.585174825175

Precision of positive class: 0.576577
Recall of positive class: 0.387175
F1-Score of positive class: 0.463265

Precision of negative class: 0.589047
Recall of negative class: 0.755463
F1-Score of negative class: 0.661956

Romney:

Overall Acurracy:  0.613911290323

Precision of positive class: 0.325439
Recall of positive class: 0.396279
F1-Score of positive class: 0.357383

Precision of negative class: 0.755923
Recall of negative class: 0.694781
F1-Score of negative class: 0.724063
'''