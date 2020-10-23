import csv
from collections import namedtuple

from langdetect import detect

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

Tweet = namedtuple('Tweet', ['comp', 'sent', 'text'])

b_puncs = "():;,."
g_puncs = "?!"

compToDigit = {'apple': 0, 'google': 1, 'microsoft': 2, 'twitter': 3}
digitToComp = ['apple', 'google', 'microsoft', 'twitter']

sentToDigit = {'neutral': 0, 'positive': 1, 'negative': -1, 'irrelevant': -100}
digitToSent = {0: 'neutral', 1: 'positive', -1: 'negative', -100: 'irrelevant'}


# parsing of the 'Train.csv' file
def parse_csv(filename):
  with open(filename, 'r', encoding='utf8') as file:
    reader = csv.reader(file, delimiter=',')
    for i, line in enumerate(reader):
      if i == 0:
        continue
      yield Tweet(compToDigit[line[0]], sentToDigit[line[1]], cut_text(line[3]))


# preprocessing of tweet text
def cut_text(text):
  # all symbols to lowercase
  text = text.lower()
  # delete all links
  after_links = ''
  parts = text.split()
  for word in parts:
    if len(word) >= 7 and word[: 7] == 'http://':
      continue
    else:
      after_links += word + ' '
  # 
  after_punc = ''
  for sym in after_links:
    for p in b_puncs:
      if sym == p:
        after_punc += ' '
        break
    else:
      for p in g_puncs:
        if sym == p:
          after_punc += ' ' + p
          break
      else:
        after_punc += sym
  # delete all tegs except from companies ones
  parts = after_punc.split()
  new_text = ''
  for word in parts:
    if len(word) > 0:
      if word[0] == '@':
        for comp in digitToComp:
          if word[1 :] == comp:
            new_text += word + ' '
            break
      else:
        new_text += word + ' '
  # 
  return new_text

# get cut tweets for training
trains = list(parse_csv('Train.csv'))


# Company Classifier
class CompanyClassifier:
  
  def __init__(self):
    self.pipe = Pipeline([('vect', CountVectorizer()),
                          ('tfid', TfidfTransformer()),
                          ('clf', RandomForestClassifier())])

  def predict_all(self, tweet_texts):
    res = []
    x = []
    for tweet in tweet_texts:
      x.append(tweet)
    prediction = self.pipe.predict(x)
    for i, t in enumerate(tweet_texts):
      res.append(prediction[i])
    return res
  
  def train(self, tweets):                
    x = []
    y = []
    for tweet in tweets:                            
      x.append(tweet.text)
      y.append(tweet.comp)
    self.pipe.fit(x, y)


# Sentiment Classifier
class SentimentClassifier:
  
  def __init__(self):
    self.pipe = Pipeline([('vect', CountVectorizer()),
                          ('tfid', TfidfTransformer()),
                          ('clf', LinearSVC())])

  def predict_all(self, tweet_texts):
    res = []
    x = []
    for tweet in tweet_texts:
      x.append(tweet)
    prediction = self.pipe.predict(x)
    for i, tweet in enumerate(tweet_texts):
      # after cut_text the tweeet may be empty
      if len(tweet) == 0:      
        res.append(0)
      else:
        res.append(prediction[i])
    return res
  
  def train(self, tweets):                
    x = []
    y = []
    for tweet in tweets:                            
      x.append(tweet.text)
      y.append(tweet.sent)
    self.pipe.fit(x, y)

# classifiers training
companyClassifier = CompanyClassifier()
companyClassifier.train(trains)

sentimentClassifier = SentimentClassifier()
sentimentClassifier.train(trains)

# can determine the company and sentiment of one tweet in input
def one_tweet(text):
  cut = cut_text(text)
  comp = companyClassifier.predict_all([cut])[0]
  sent = sentimentClassifier.predict_all([cut])[0]
  print(digitToComp[comp], digitToSent[sent])

# parse file like 'Test.csv'
def from_file():
  tests = list(parse_csv('Test.csv'))
  texts = [t.text for t in tests]
  predComp = [digitToComp[t] for t in companyClassifier.predict_all(texts)]
  realComp = [digitToComp[t.comp] for t in tests]
  predSent = [digitToSent[t] for t in sentimentClassifier.predict_all(texts)]
  realSent = [digitToSent[t.sent] for t in tests]
  # statistics might be printed here
  print('for companies')
  print(classification_report(realComp, predComp))
  print('for sentiment')
  print(classification_report(realSent, predSent))

if __name__ == '__main__':
  while True:
    tweet_text = input()
    one_tweet(tweet_text)

  # from_file()
