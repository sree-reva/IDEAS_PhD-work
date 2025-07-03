# Import necessary Libraries

import string
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize,word_tokenize

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from collections import Counter 
from string import punctuation
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS as stop_words
from sklearn.feature_extraction.text import CountVectorizer #countvectorizer is a class to 
                                                            #implement BOW model

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing

import pickle

#**************************************************************************************
# DATA PREPROCESSING
#*************************************************************************************


## REMOVE PUNCTUATION

import string
string.punctuation

def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


## TOKENIZATION

# Defining a function for tokenization

import re

def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens


# REMOVE STOP WORDS

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
stopwords[0:10]

def remove_stopwords(txt_tokenized):
    txt_rsw = [word for word in txt_tokenized if word not in stopwords]
    return txt_rsw



# STEMMING

import nltk
from nltk.stem import PorterStemmer
ps = PorterStemmer()
dir(ps)


## Defining  a function for stemming

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

## WORDNET LEMMATIZER

import nltk
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
dir(wn)

# Defining a Lemmatizer function

def lemmatization(token_text):
    text = [wn.lemmatize(word) for word in token_text]
    return text



def preprocessing(data):
    
    df = data
    
    # Create a new column in the dataframe 'RA_tokenized_sents' to store Reference answer sentence tokens
    
    df['RA_tokenized_sents'] = df.apply(lambda row: nltk.sent_tokenize(row['REFANS']), axis=1)
    
    # Create a  new column in the dataframe called 'REFERENCE ANSWER_NOPUNCT' to store reference answer without punctuations
    
    df['REFERENCE ANSWER_NOPUNCT'] = df['REFANS'].apply(lambda x: remove_punctuation(x))
    
    #Create a new column 'RA_TOKENIZED' to store tokens
    
    df['RA_TOKENIZED'] = df['REFERENCE ANSWER_NOPUNCT'].apply(lambda x: tokenize(x.lower()))
    
    # Create a new column 'RA_NO_SW' to store non stop words
    
    df['RA_NO_SW'] = df['RA_TOKENIZED'].apply(lambda x: remove_stopwords(x))

    # Create a new column 'RA_STEMMED' to store STEMMED words
    
    df['RA_STEMMED'] = df['RA_NO_SW'].apply(lambda x:stemming(x))
    
    # creating a new column 'RA_LEMMATIZED' to store lemmatized words
    
    df['RA_LEMMATIZED'] =  df['RA_NO_SW'].apply(lambda x: lemmatization(x))
    
    # Create a new column in the dataframe 'SA_tokenized_sents' to store Student answer sentence tokens
    
    df['SA_tokenized_sents'] = df.apply(lambda row: nltk.sent_tokenize(row['STUANS']), axis=1)
    
    # creating a new column ''SA_NOPUNCT'' to store student answer without punctuations
    
    df['SA_NOPUNCT'] = df['STUANS'].apply(lambda x: remove_punctuation(x))
    
    #Create a new column 'SA_TOKENIZED' to store tokens
    
    df['SA_TOKENIZED'] = df['SA_NOPUNCT'].apply(lambda x: tokenize(x.lower()))

    # Create a new column 'SA_NO_SW' to store non stop words

    df['SA_NO_SW'] = df['SA_TOKENIZED'].apply(lambda x: remove_stopwords(x))
    
    # Create a new column 'SA_STEMMED' to store STEMMED words

    df['SA_STEMMED'] = df['SA_NO_SW'].apply(lambda x:stemming(x))

    # creating a new column 'SA_LEMMATIZED' to store lemmatized words

    df['SA_LEMMATIZED'] =  df['SA_NO_SW'].apply(lambda x: lemmatization(x))
    
    return df      

