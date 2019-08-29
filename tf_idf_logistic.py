from test_set_read import Read_from_TestSet,Read_from_test_set_unlabelled
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
import re
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
train_data = Read_from_TestSet()
test_data = Read_from_test_set_unlabelled()
#df = pd.DataFrame(train_data, columns=['author', 'text'])

#test_text = df_test['test_text']
train_data = np.array(train_data)
#author
df = pd.DataFrame(train_data, columns=['author', 'text'])
print(df.shape)
author = df['author']
wordnet_lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    return [wordnet_lemmatizer.lemmatize(w) for w in text]
def regular_expression(text):
    temp = []
    for words in text:
        words = re.sub(r'[:|,|)|(|\|/]',r'',words)
        words = re.sub(r'[\'|"|#]',r'',words)
        words = re.sub('!+','!',words)
        words = re.sub(r'\.+',r'.',words)
        words = re.sub(r'\$+',r'$',words)
        words = re.sub(r'\*+',r'*',words)
        temp.append(words)
    return temp
#df['text']=df.text.apply(regular_expression)
df['text']=df.text.apply(lemmatize_text)
text = df['text']

X_train, X_test, Y_train, Y_test = train_test_split(text, author,test_size=0.2)
print("Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_test.shape[0]))

def doc(text):
    return text
vectorizer = TfidfVectorizer(stop_words="english",
                             tokenizer=doc, lowercase=False,
                             ngram_range=(1, 2))

training_features = vectorizer.fit_transform(X_train)    
test_features = vectorizer.transform(X_test)
#real_test_features = vectorizer.transform(test_text)

classifier = LogisticRegression(n_jobs=1, C=1e5)
#mlb = MultiLabelBinarizer()
#Y_train1 = mlb.fit_transform(Y_train)
#classifier = OneVsRestClassifier(LogisticRegression(n_jobs=1, C=1e5))
print("start fitting")
classifier.fit(training_features, Y_train)
print("done fitting")


#Y_test1 = mlb.transform(Y_test)
#print(Y_test1)
Y_test_pred = classifier.predict(test_features)
#print(Y_test_pred)
print(accuracy_score(Y_test, Y_test_pred))

df= pd.DataFrame({'test_text':test_data})
test_real_features = vectorizer.transform(df['test_Data'])
Y_test_pred_output = classifier.predict(test_real_features)

df = pd.DataFrame(Y_test_pred_output,columns=['Predicted'])
df.index = df.index + 1
df.index.name = 'Id'
df.to_csv('predict_SML_Pro.csv')
