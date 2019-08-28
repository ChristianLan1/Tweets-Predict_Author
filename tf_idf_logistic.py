from test_set_read import Read_from_TestSet,Read_from_test_set_unlabelled
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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

train_data = Read_from_TestSet()
test_data = Read_from_test_set_unlabelled()
df = pd.DataFrame(train_data, columns=['author', 'text'])
df_test= pd.DataFrame({'test_text':test_data})
test_text = df_test['test_text']
#train_data = np.array(train_data)
#author
#df = pd.DataFrame(train_data, columns=['author', 'text'])
print(df.shape)
author = df['author']
text = df['text']

X_train, X_test, Y_train, Y_test = train_test_split(text, author,test_size=0.2, random_state=42)
print("Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_test.shape[0]))


vectorizer = TfidfVectorizer(stop_words="english",
                             tokenizer=lambda doc: doc, lowercase=False,
                             ngram_range=(1, 2))

training_features = vectorizer.fit_transform(X_train)    
test_features = vectorizer.transform(X_test)
real_test_features = vectorizer.transform(test_text)


#mlb = MultiLabelBinarizer()
#Y_train1 = mlb.fit_transform(Y_train)
classifier = OneVsRestClassifier(LogisticRegression(n_jobs=1, C=1e5))
classifier.fit(training_features, Y_train)



#Y_test1 = mlb.transform(Y_test)
#print(Y_test1)
Y_test_pred = classifier.predict(test_features)
#print(Y_test_pred)
print(accuracy_score(Y_test, Y_test_pred))

Y_test_pred_output = classifier.predict(real_test_features)

df = pd.DataFrame(Y_test_pred_output,columns="Predicted")
df.index = df.index + 1
df.index.name = 'Id'
df.to_csv('predict_SML_Pro')
