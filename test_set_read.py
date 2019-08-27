import pdb
import zipfile
import datetime
from nltk.tokenize import word_tokenize, wordpunct_tokenize
#from collections import Counter,defaultdict
import numpy as np
ISOTIMEFORMAT = '%H:%M:%S'

def BM25_PREPROCESSING(Sentence):
    #Function to Tokecnize the sentence in to word, and also split punctuation
    #Remain the Capital words     
    New_line = []
    words = wordpunct_tokenize(Sentence)     
    return words

def Read_from_TestSet():
    # Function to read line from files
    # Put this file in the folder of "train_tweets.zip"
    # Return a tuple list [('8746',['I','am','@'...]),(),(),(),...] 
    
    #pdb.set_trace()
    my_docs = []
    #my_docs_key = defaultdict(list)
    print("begin to read data files,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    myzip = zipfile.ZipFile('train_tweets.zip')
    ziplist = myzip.namelist()
    ## loop each txt file
    fileobj = myzip.open(ziplist[0])
    j = 0
    for line in fileobj:
        # must use utf-8 to store different languages
        # remove "/n" at each line 
        myline = line.decode('utf-8').strip()
        # use first 1 TAB to cut the string
        j = j+1
        #pdb.set_trace()
        line_list = myline.split('\t',1)
        #print(line_list)
        value_txt = BM25_PREPROCESSING(line_list[1])
        my_docs.append((line_list[0],value_txt))
        
        # if line_list[0] in my_docs_key.keys():
            # pdb.set_trace()
            #my_docs_key[line_list[0]].append(value_txt)
        # else:
            # my_docs_key[line_list[0]] = []
            #my_docs_key[line_list[0]].append(value_txt)
    # break
    myzip.close()
    #pdb.set_trace()
    print("File Reading Finish,time:"+ datetime.datetime.now().strftime(ISOTIMEFORMAT))
    return my_docs

def Read_from_test_set_unlabelled():
    # Read unlabelled set to predict
    my_docs_unlabelled = []
    with open('test_tweets_unlabeled.txt', 'rb') as files:
        for line in files:
            myline = line.decode('utf-8').strip()
            value_txt = BM25_PREPROCESSING(myline)
            my_docs_unlabelled.append(value_txt)
    return my_docs_unlabelled

def create_set_test_train(my_docs_key):
    # random create test set and train set by numpy 
    # total: 328932
    # set the test set length to change the length of both sets 
    # present length is 20%
    test_set_length = 66000
    test_set = []
    train_set = []
    np.random.seed(12345)
    np.random.shuffle(my_docs_key)
    test_set = my_docs_key[0:test_set_length]
    train_set = my_docs_key[test_set_length:]
    return test_set,train_set 


#my_docs = Read_from_TestSet()
#test_set,train_set = create_set_test_train(my_docs)
#print('Read test and trainning set success! can do the following step now')
#pdb.set_trace()
#my_docs_unlabelled = Read_from_test_set_unlabelled()
#print('Read unlabelled file success! can do the predict now')