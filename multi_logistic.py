#from test_set_read import Read_from_TestSet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
#sns.set_style('darkgrid')
#%matplotlib inline
#train_data = Read_from_TestSet()
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import nltk
nltk.download('punkt')
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords
cores = multiprocessing.cpu_count()
df = pd.read_csv('train_tweets.txt', header = None,sep='\t')
#print(df)
df.columns = ['author','text']
#df = pd.DataFrame(train_data, columns=['author', 'text'])
author = df['author']
text = df['text']


train, test = train_test_split(df, test_size=0.1, random_state=90051)


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            word = re.sub(r'[:|,|)|(|\|/]','',word)
            word = re.sub(r'[\'|"|]','',word)
            word = re.sub('!+','!',word)
            word = re.sub(r'\.+',r'.',word)
            word = re.sub(r'\$+',r'$',word)
            word = re.sub(r'\*+',r'*',word)
            word = word.replace("http","")
            if not word.isupper():
                #print(word)
                word = word.lower()
            tokens.append(word)
    tokens = [wordnet_lemmatizer.lemmatize(w) for w in tokens]
    return tokens
train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.author]), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=[r.author]), axis=1)

print(train_tagged[0])
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=15, alpha=0.065, min_alpha=0.065,hs=0,sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha



def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    
    return targets, regressors
y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)
#print(X_train[:2])
#logreg = OneVsRestClassifier(LinearSVC())
logreg = LogisticRegression(n_jobs=1, C=1)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))


df = pd.read_csv('test_tweets_unlabeled.txt', header = None,sep='\t')
df.columns = ['text']
real_test_tagged = df['text'].apply(
    lambda r: TaggedDocument(words=tokenize_text(r), tags=[1]))

def vec_for_predict(model, tagged_docs):
    sents = tagged_docs.values
    regressors = [model.infer_vector(doc.words) for doc in sents]
    return regressors

X_test_real = vec_for_predict(model_dbow, real_test_tagged)

Y_test_pred_output = logreg.predict(X_test_real)
df = pd.DataFrame(Y_test_pred_output,columns=['Predicted'])
df.index = df.index + 1
df.index.name = 'Id'
df.to_csv('doc2vec.csv')

