# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 18:45:07 2020

@author: shiva dumnawar
"""

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

df= pd.read_csv('fake news.csv')

df.info()

df.isnull().sum()

df= df.dropna()

df.shape

df.reset_index(inplace= True)

ps= PorterStemmer()

message= df['text'][:2000]  
# This is a huge dataset so taking text feature with 2000 rows

corpus= []
for i in range(len(message)):
    review= re.sub('[^a-zA-Z]', ' ', message[i])
    review= review.lower()
    review= review.split()
    review= [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review= ' '.join(review)
    corpus.append(review)
  
# Bag of words    
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000)
X= cv.fit_transform(corpus).toarray()

y= df['label'][:2000].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=75)

from sklearn.naive_bayes import MultinomialNB
clf= MultinomialNB().fit(X_train, y_train)

y_pred= clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test, y_pred)
print(cm)

acc_score= accuracy_score(y_test, y_pred)
print(acc_score)






