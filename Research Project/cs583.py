import pandas as pd
import nltk
import re
from sklearn import *
import time

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

def vectorization(clean_train_tweets):
	vec = feature_extraction.text.TfidfVectorizer(min_df = 0.00125, max_df = 0.7, sublinear_tf=True, use_idf=True, stop_words=u'english', analyzer= 'word', ngram_range=(1,5),lowercase=True)
	train_vectors = vec.fit_transform(clean_train_tweets)
	return train_vectors

#Loading the sheets into data frames

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

obama_tweets_vectors = vectorization(obama_tweets)
romney_tweets_vectors = vectorization(romney_tweets)

models = [naive_bayes.BernoulliNB(),svm.SVC(kernel='rbf', gamma=0.58, C=0.81),tree.DecisionTreeClassifier(random_state=0),ensemble.RandomForestClassifier(criterion='entropy', n_jobs = 10),linear_model.LogisticRegression(),linear_model.SGDClassifier(),neural_network.MLPClassifier()]

for clf in models:
    start_time = time.clock()
    #obama
    preds = model_selection.cross_val_predict(clf, obama_tweets_vectors, obama_class_train, cv=10)
    accScore = metrics.accuracy_score(obama_class_train,preds)
    labels = [1,-1]
    precision = metrics.precision_score(obama_class_train,preds,average=None,labels=labels)
    recall = metrics.recall_score(obama_class_train,preds,average=None,labels=labels)
    f1Score = metrics.f1_score(obama_class_train,preds,average=None,labels=labels)
    print(clf);print("Obama: \nOverall Acurracy: ",accScore,"\n")
    lbl = ['positive', 'negative']
    for i in range(2):
        print("Precision of %s class: %f" %(lbl[i],precision[i]))
        print("Recall of %s class: %f" %(lbl[i],recall[i]))
        print("F1-Score of %s class: %f" %(lbl[i],f1Score[i]),"\n")
    #romney
    preds = model_selection.cross_val_predict(clf, romney_tweets_vectors, romney_class_train, cv=10)
    accScore = metrics.accuracy_score(romney_class_train,preds)
    precision = metrics.precision_score(romney_class_train,preds,average=None,labels=labels)
    recall = metrics.recall_score(romney_class_train,preds,average=None,labels=labels)
    f1Score = metrics.f1_score(romney_class_train,preds,average=None,labels=labels)
    print("Romney:\nOverall Acurracy: ",accScore,"\n")
    for i in range(2):
        print("Precision of %s class: %f" %(lbl[i],precision[i]))
        print("Recall of %s class: %f" %(lbl[i],recall[i]))
        print("F1-Score of %s class: %f" %(lbl[i],f1Score[i]),"\n")
    end_time = time.clock()
    print("Total time taken: %0.2f seconds \n\n"%(end_time-start_time))