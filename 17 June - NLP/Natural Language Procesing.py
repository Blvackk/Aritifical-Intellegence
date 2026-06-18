# Natural Language Processsing

#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing th dataset
dataset = pd.read_csv (r"E:\AI\17 June\Restaurant_Reviews.tsv", delimiter='\t',quoting=3)

#CLEANING THE TEXTS
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus =[]

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review= review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)
    corpus.append(review)


# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform (corpus).toarray()

y = dataset.iloc[:,1].values


#Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train , X_test, y_train ,y_test = train_test_split(X,y,test_size=0.20)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,y_pred)
print (ac)

bias  = classifier.score(X_train,y_train)
bias

variance = classifier.score(X_test,y_test)
variance 