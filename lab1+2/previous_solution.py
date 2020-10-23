import csv
from collections import namedtuple

from langdetect import detect
import re

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.model_selection import GridSearchCV

Tweet = namedtuple('Tweet', ['comp', 'sent', 'text'])

b_puncs = "():;,."
g_puncs = "?!"

toDigit = {'apple': 0, 'google': 1, 'microsoft': 2, 'twitter': 3}
companies = ['apple', 'google', 'microsoft', 'twitter']

def parse_csv(filename):
  with open(filename, 'r', encoding='utf8') as file:
    # headers = file.readline()
    reader = csv.reader(file, delimiter=',')
    for i, line in enumerate(reader):
      if i == 0:
        continue
      # if line[1] == 'irrelevant':
      #   continue
      sent = 0 if line[1] == 'neutral' else 1 if line[1] == 'positive' else -1 if line[1] == 'negative' else -100
      # if i % 10 == 0:
      #   print(line[1], sent)
      # text = line[3].encode('ascii',errors='ignore').decode('utf8',errors='ignore')
      # text = base64.b64decode(line[3]).decode('utf-8', errors='ignore')
      yield Tweet(toDigit[line[0]], sent, cut_text(line[3]))

def cut_text(text):
  # 
  text = text.lower()
  # 
  # after_links = ''
  # parts = text.split()
  # for word in parts:
  #   if len(word) >= 7 and word[: 7] == 'http://':
  #     continue
  #   else:
  #     after_links += word + ' '
  # 
  # after_punc = ''
  # for sym in after_links:
  #   for p in b_puncs:
  #     if sym == p:
  #       after_punc += ' '
  #       break
  #   else:
  #     for p in g_puncs:
  #       if sym == p:
  #         after_punc += ' ' + p
  #         break
  #     else:
  #       after_punc += sym
  # 
  parts = text.split() # after_punc.split()
  new_text = ''
  for word in parts:
    if len(word) > 0:
      if word[0] == '@':
        for comp in companies:
          if word[1 :] == comp:
            new_text += word + ' '
            break
      else:
        new_text += word + ' '
  # 
  # print(text)
  # print(after_links)
  # print(after_punc)
  # print(new_text)
  # 
  return new_text

trains = list(parse_csv('Train.csv'))
tests = list(parse_csv('Test.csv'))

print("train size =", len(trains))
print("test size =", len(tests))

# Company Classifier

class CompanyClassifier:
  
  def __init__(self):
    self.pipe = Pipeline([('vect', CountVectorizer()),
                          ('tfid', TfidfTransformer()),
                          ('clf', RandomForestClassifier())])

  def predict_all(self, tweets):
    res = []
    x = []
    for tweet in tweets:
      x.append(tweet.text)
    prediction = self.pipe.predict(x)
    for i, tweet in enumerate(tweets):
      res.append((prediction[i], tweet.comp, tweet))
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

  def predict_all(self, tweets):
    res = []
    x = []
    for tweet in tweets:
      x.append(tweet.text)
    prediction = self.pipe.predict(x)
    for i, tweet in enumerate(tweets):
      if len(tweet.text) == 0:      
        res.append((0, tweet.sent, tweet.text))
        continue
      res.append((prediction[i], tweet.sent, tweet.text))
      # 
      # try:
      #   lang = detect(tweet.text)
      #   if not lang == 'en':
      #     res.append((-100, tweet.sent, tweet.text))
      #   else:
      #     res.append((prediction[i], tweet.sent, tweet.text))
      # except:
      #   res.append((-100, tweet.sent, tweet.text))
      # 
      # if not detect(tweet.text) == 'en':
      #   # if not prediction[i] == -100 and tweet.sent == -100:
      #     # print(prediction[i], i, tweet.text)
      #   res.append((-100, tweet.sent, tweet.text))
      # else:
      #   res.append((prediction[i], tweet.sent, tweet.text))
    return res
  
  def train(self, tweets):                
    x = []
    y = []
    for tweet in tweets:                            
      x.append(tweet.text)
      y.append(tweet.sent)
    self.pipe.fit(x, y)


companyClassifier = CompanyClassifier()
companyClassifier.train(trains)

sentimentClassifier = SentimentClassifier()
sentimentClassifier.train(trains)

# with open('my_submission.csv', 'w') as fout:
#   writer = csv.writer(fout)
#   writer.writerow(['Prediction','Real'])
#   res = companyClassifier.predict_all(tests)
#   for item in res:
#     prediction = item[1]
#     writer.writerow([item[0], item[1]])

if __name__ == '__main__':
  # print(trains[0])
  # res = companyClassifier.predict_all(tests)
  res = sentimentClassifier.predict_all(tests)
  pred = [t[0] for t in res]
  real = [t[1] for t in res]
  # for item in res:
  #   # prediction = item[1]
  #   print(item[0], item[1])
    # if not item[0] == item[1]:
    #   print('     ', item[2]) 
  print(classification_report(real, pred))
  # for x in trains:
    # print(x)