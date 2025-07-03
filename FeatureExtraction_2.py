# Import necessary Libraries
import string
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


#********************************************************************************************
#2. EXTRACTING SIMILARITIES
# *****************************************************************************************

#FINDING EUCLIDIAN DISTANCE

def euclid_dist(t1, t2):
    return np.sqrt(((t1-t2)**2).sum(axis = 0))


#REMOVE PUNCTUATION


def remove_punctuation(txt):
    txt_nopunct = "".join([c for c in txt if c not in string.punctuation])
    return txt_nopunct


#REQUIRED FOR LEMMA SIMILARITY

## WORDNET LEMMATIZER

import nltk
nltk.download('all')

wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()
dir(wn)

# Defining a Lemmatizer function

def lemmatization(token_text):
    text = [wn.lemmatize(word) for word in token_text]
    return text


#required for semantic similarity

import numpy as np
import torch
import numpy.linalg as LA

# Trained model from: 

GLOVE_EMBS = 'C:/Users/Venkat/glove.840B.300d/glove.840B.300d.txt'
INFERSENT_MODEL = r'C:/Users/Venkat/infersent1.pkl'

W2V_PATH = r'C:/Users/Venkat/fastText/crawl-300d-2M.vec/crawl-300d-2M.vec'
INFERSENT_MODEL = r'C:/Users/Venkat/infersent2.pkl'


# define a function which returns the cosine similarity between 2 vectors
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# 2) Load pre-trained model (in encoder/):

# First Copy the models.py file from inferent_master folder to the place of your current .pynb file location
 
from models import InferSent
V = 2 # version V= 2
MODEL_PATH = r'C:/Users/Venkat/encoder/infersent%s.pkl' % V
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,'pool_type': 'max', 
                'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))


# 3) Set word vector path for the model

W2V_PATH = r'C:\Users\Venkat\fastText\crawl-300d-2M.vec\crawl-300d-2M.vec'
infersent.set_w2v_path(W2V_PATH) 


## TOKENIZATION

# Defining a function for tokenization

import re

def tokenize(txt):
    tokens = re.split('\W+', txt)
    return tokens


#REQUIRED FOR SUMMARY SIMILARITY

# REMOVE STOP WORDS

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


def remove_stopwords(txt_tokenized):
    txt_rsw = [word for word in txt_tokenized if word not in stopwords]
    return txt_rsw



# defining the sentence tokenizer

def sent_tokenizer(s):
    sents = []
    for sent in s.split('.'):
        sents.append(sent.strip())
    return sents


#To count the occurrences of each word in the reference answer tokens.

def count_words(tokens):
    word_counts = {}
    for token in tokens:
        if token not in stop_words and token not in punctuation:
            if token not in word_counts.keys():
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    return word_counts



# Build a word frequency distribution:we divided the occurrence of each word by 
# the frequency of the most occurring word to get our distribution.

def word_freq_distribution(word_counts):
    freq_dist = {}
    max_freq = max(word_counts.values())
    for word in word_counts.keys():  
        freq_dist[word] = (word_counts[word]/max_freq)
    return freq_dist



# to score our sentences by using the frequency distribution we generated. 
#This is simply summing up the scores of each word in a sentence and hanging on to 
#the score. Our function takes a max_len argument which sets a maximum length to 
#sentences which are to be considered for use in the summarization.

def score_sentences(sents, freq_dist, max_len=40):
    sent_scores = {}  
    for sent in sents:
        words = sent.split(' ')
        for word in words:
            if word.lower() in freq_dist.keys():
                if len(words) < max_len:
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = freq_dist[word.lower()]
                    else:
                        sent_scores[sent] += freq_dist[word.lower()]
    return sent_scores


# select (i.e. extract, as in "extractive summarization") the top k sentences to 
# represent the summary of the article. 
#This function will take the sentence scores we generated above as well as a value for 
#the top k highest scoring sentences to sue for summarization. It will return a string 
# summary of the concatenated top sentences, as well as the sentence scores of the 
# sentences used in the summarization.

#REQUIRED FOR SUMMARY SIMILARITY

# REMOVE STOP WORDS

import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


def remove_stopwords(txt_tokenized):
    txt_rsw = [word for word in txt_tokenized if word not in stopwords]
    return txt_rsw



# defining the sentence tokenizer

def sent_tokenizer(s):
    sents = []
    for sent in s.split('.'):
        sents.append(sent.strip())
    return sents


#To count the occurrences of each word in the reference answer tokens.

def count_words(tokens):
    word_counts = {}
    for token in tokens:
        if token not in stop_words and token not in punctuation:
            if token not in word_counts.keys():
                word_counts[token] = 1
            else:
                word_counts[token] += 1
    return word_counts



# Build a word frequency distribution:we divided the occurrence of each word by 
# the frequency of the most occurring word to get our distribution.

def word_freq_distribution(word_counts):
    freq_dist = {}
    
    if not word_counts:  # Check if word_counts is empty
        return freq_dist

    max_freq = max(word_counts.values())
    
    for word in word_counts.keys():
        freq_dist[word] = (word_counts[word] / max_freq)
    
    return freq_dist

# to score our sentences by using the frequency distribution we generated. 
# This is simply summing up the scores of each word in a sentence and hanging on to 
# the score. Our function takes a max_len argument which sets a maximum length to 
# sentences which are to be considered for use in the summarization.

def score_sentences(sents, freq_dist, max_len=40):
    sent_scores = {}  
    for sent in sents:
        words = sent.split(' ')
        for word in words:
            if word.lower() in freq_dist.keys():
                if len(words) < max_len:
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = freq_dist[word.lower()]
                    else:
                        sent_scores[sent] += freq_dist[word.lower()]
    return sent_scores


# select (i.e. extract, as in "extractive summarization") the top k sentences to 
# represent the summary of the article. 
#This function will take the sentence scores we generated above as well as a value for 
#the top k highest scoring sentences to sue for summarization. It will return a string 
# summary of the concatenated top sentences, as well as the sentence scores of the 
# sentences used in the summarization.

def summarize(sent_scores, k):
    top_sents = Counter(sent_scores) 
    summary = ''
    scores = []
    
    top = top_sents.most_common(k)
    for t in top: 
        summary += t[0].strip()+'. '
        scores.append((t[1], t[0]))
     # return summary[:-1], scores      # returns both summary and scores
    return summary  # returns only summary


"#####################################################################################"

# Define Functions for each similarity
 
 #1. STATISTICAL SIMILARITY

def statistical_similarity(RA,SA):
    RA = RA
    SA = SA
    #Count the sentences in reference answer
    RA_sent_count = len(sent_tokenizer(RA))
    
    #Count the sentences in student answer
    SA_sent_count = len(sent_tokenizer(SA))
    
    #Count the words in reference answer
    RA_word_sount = len(word_tokenize(RA))
    
    #Count the words in Student answer
    SA_word_count = len(word_tokenize(SA))
    
    #Count the Unique words in reference answer 
    RA_Unique_word = remove_stopwords(word_tokenize(RA))
    RA_Unique_word_count = len(RA_Unique_word)
    
    #Count the unique words in student answer
    SA_Unique_word = remove_stopwords(word_tokenize(SA))
    SA_Unique_word_count = len(SA_Unique_word)
    
    #Store all reference answer statistical data in a numpy array
    RA_stats_data = np.array([RA_sent_count,RA_word_sount,RA_Unique_word_count])   
    
    #Store all student answer statistical data in a numpy array
    SA_stats_data = np.array([SA_sent_count,SA_word_count,SA_Unique_word_count])
    
    #FInding the euclidian distance between the statistical data of reference and student answer numpy arrays
    sta_sim = euclid_dist(RA_stats_data,SA_stats_data)

    # Return the similarity value with decimal places
    decimal_places = 3 # Specify the number of decimal places
    sta_sim = np.round( sta_sim, decimal_places)
    return sta_sim


#2. WORD-WORD SIMILARITY USING BAG OF WORDS

def BOW_similarity(RA,SA):
    train_set = [RA]
    test_set = [SA]
    rvect = CountVectorizer()#CREATE AN INSTANCE OF WORD-WORD SIMILARITY USING BAG OF WORDSCOUNT VECTORIZER
    rbow = rvect.fit_transform(train_set) #FIT THE COUNT VECTORIZER ON TRAIN SET IE REF ANSWER
    
    # TRASFORM THE COUNT VECTORIZER ON TEST SET IE STUDENT ANSWER
    sbow = rvect.transform(test_set)
    
    #CALCULATING THE COSINE SIMILARITY BETWEEN REFERENCE AND STUDENT ANSWER BAG OF WORDS
    W_W_SIMILARITY = cosine_similarity(rbow,sbow)

    # Return the similarity value with decimal places
    decimal_places = 3 # Specify the number of decimal places
    W_W_SIMILARITY = np.round( W_W_SIMILARITY[0][0], decimal_places)
    return W_W_SIMILARITY


#3. UNIQUE OR NO STOP WORDS SIMILARITY

def NO_STOP_WORDS_similarity(RA, SA):
    train_set = [RA]
    test_set =[SA]
    rvect = CountVectorizer(stop_words='english')#CREATE AN INSTANCE OF COUNT VECTORIZER by considering the stop words
    rbow = rvect.fit_transform(train_set) #FIT THE COUNT VECTORIZER ON TRAIN SET IE REF ANSWER
    
    # TRANSFORM THE COUNT VECTORIZER ON TEST SET IE STUDENT ANSWER
    sbow = rvect.transform(test_set)
    
    "CALCULATING THE COSINE SIMILARITY OF REFANS AND STUANS BAG OF WORDS"
    NO_SW_SIMILARITY = cosine_similarity(rbow,sbow)

    # Return the similarity value with decimal places
    decimal_places = 3 # Specify the number of decimal places
    NO_SW_SIMILARITY = np.round(NO_SW_SIMILARITY[0][0], decimal_places)
    return NO_SW_SIMILARITY
 
 

 # 4. LEMMA_SIMILARITY
def LEMMA_SIMILARITY(RA, SA):
    RA = [RA]
    SA = [SA]
    RA_LEMMATIZED = lemmatization(RA)
    SA_LEMMATIZED = lemmatization(SA)
    
    #instantiate Count Vectorizer(vectorizer)
    rans_lemma_vect = CountVectorizer()
    rans_lemma_bow = rans_lemma_vect.fit_transform(RA_LEMMATIZED)
    
    # mapping the student answer lemmatized words to the rans_lemma_vect
    sans_lemma_bow = rans_lemma_vect.transform(SA_LEMMATIZED)
    
    #CALCULATING THE COSINE SIMILARITY OF REFANS AND STUANS BAG OF WORDSLEMMATIZED Words

    LEMMA_SIMILARITY = cosine_similarity(rans_lemma_bow,sans_lemma_bow)
    
    # Return the similarity value with decimal places
    decimal_places = 3 # Specify the number of decimal places
    LEMMA_SIMILARITY = np.round(LEMMA_SIMILARITY[0][0] , decimal_places)
    return LEMMA_SIMILARITY
    
    

#5. TFIDF SIMILARITY

def TFIDF_SIMILARITY(RA,SA):
    train_set = [RA]
    test_set = [SA]
    
    vectorizer = TfidfVectorizer() #CREATING AN INSTANCE OF TFIDF VECTORIZER
    
    #FITTING THE TFIDF VECTORIZER ON TRAIN DATA   
    trainVectorizerArray = vectorizer.fit_transform(train_set)
    
    #TRANSFORMING THE TEST DATA USING TFIDF VECTORIZER

    testVectorizerArray = vectorizer.transform(test_set)
    
    #CALCULATING THE COSINE SIMILARITY BETWEEN REFERENCE AND STUDENT ANSWER 
    TFIDF_SIM = cosine_similarity(trainVectorizerArray,testVectorizerArray)
    
    # Return the similarity value with decimal places
    decimal_places = 3 # Specify the number of decimal places
    TFIDF_SIM = np.round(TFIDF_SIM[0][0], decimal_places)
   
    return TFIDF_SIM

# 6. LSA SIMILARITY

def LSA_SIMILARITY(RA,SA):
    
    # Define your model answer (RefA) and student answer (StuA)
    RefA = str(RA)
    StuA = str(SA)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([RefA, StuA])
    
    # Apply TruncatedSVD for dimensionality reduction
    n_components = 1 # Set the desired number of components (topics)
    lsa = TruncatedSVD(n_components)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)

    # Calculate cosine similarity between the LSA representations
    cosine_sim = cosine_similarity(lsa_matrix)

    # Retrieve the similarity value
    similarity = cosine_sim[0, 1]

    # Return the similarity value with decimal places
    decimal_places = 3 # Specify the number of decimal places
    similarity_decimal = np.round(similarity, decimal_places)
    return  similarity_decimal

"""
# 7.SEMANTIC SIMILARITY

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from models import InferSent

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from models import InferSent

def SEMANTIC_SIMILARITY(RA, SA):
    
    RefA = str(RA)
    StuA = str(SA)

    # Build vocabulary
    infersent.build_vocab([RefA, StuA], tokenize=True)

    # Encode paragraphs
    RefAE = infersent.encode([RefA], tokenize=True)
    StuAE = infersent.encode([StuA], tokenize=True)

    # Calculate cosine similarity
    Sem_simi = np.dot(RefAE[0], StuAE[0]) / (np.linalg.norm(RefAE[0]) * np.linalg.norm(StuAE[0]))
    return  Sem_simi
"""


# 7. Semantic similarity using Doc2vec

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

def doc2vec_similarity(RA,SA):
    # Example sentences
    sentence1 = RA
    sentence2 = SA
    # Tokenize the sentences
    tokenized_sentence1 = word_tokenize(sentence1.lower())
    tokenized_sentence2 = word_tokenize(sentence2.lower())

    # Create TaggedDocument objects
    tagged_data = [
        TaggedDocument(words=tokenized_sentence1, tags=["0"]),
        TaggedDocument(words=tokenized_sentence2, tags=["1"]),
        ]

    # Initialize and train Doc2Vec model
    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Get vector representations for the sentences
    vector_sentence1 = model.infer_vector(tokenized_sentence1)
    vector_sentence2 = model.infer_vector(tokenized_sentence2)

    # Calculate cosine similarity between the two sentence vectors
    similarity = cosine_similarity([vector_sentence1], [vector_sentence2])[0][0]
    return similarity







# # 7. SUMMARY SIMILARITY
def SUMMARY_SIMILARITY(RA,SA): 
        
    Refans = [RA]
    Stans = [SA]
    
    # REMOVE PUNCTUATION
    RA_NOPUNCT = remove_punctuation(RA)
    SA_NOPUNCT = remove_punctuation(SA)
    
    # TOKENIZATION
    RA_TOKENIZED = tokenize(RA_NOPUNCT.lower())
    SA_TOKENIZED = tokenize(SA_NOPUNCT.lower())
    
    # REMOVE STOP WORDS
    RA_NO_SW = remove_stopwords(RA_TOKENIZED)
    SA_NO_SW = remove_stopwords(SA_TOKENIZED)

    #SENTENCE TOKENIZATION
    R_SENTS = sent_tokenizer(RA)
    S_SENTS = sent_tokenizer(SA)
    
    #To count the occurrences of each word in the Reference and STUDENT answer tokens
    R_WORD_COUNTS = count_words(RA_NO_SW)
    S_WORD_COUNTS = count_words(SA_NO_SW)
    
    # Build a word frequency distribution:we divided the occurrence of each word by the frequency of the 
    #most occurring word to get our distribution.

    R_FREQ_DIST = word_freq_distribution(R_WORD_COUNTS)
    S_FREQ_DIST = word_freq_distribution(S_WORD_COUNTS)
    
    # SCORE THE SENTENCES
    R_SENT_SCORES = {}
    for sent in R_SENTS:
        words = tokenize(sent.lower())
        no_sw = remove_stopwords(words)
        score = sum(R_FREQ_DIST[word] for word in no_sw if word in R_FREQ_DIST)
        R_SENT_SCORES[sent] = score
    
    S_SENT_SCORES = {}
    for sent in S_SENTS:
        words = tokenize(sent.lower())
        no_sw = remove_stopwords(words)
        score = sum(S_FREQ_DIST[word] for word in no_sw if word in S_FREQ_DIST)
        S_SENT_SCORES[sent] = score
    
       
    # FINDING SUMMARY
    R_SUMMARY = [summarize(R_SENT_SCORES,3)]
    S_SUMMARY = [summarize(S_SENT_SCORES,3)]
    
    #FINDING THE SIMILARITY BETWEEN THE REFERENCE ANSWER Summary AND STUDENT ANSWER Summary 
    
    #instantiate Count Vectorizer for reference answer summary
    rsumvect = CountVectorizer()
    rsumbow = rsumvect.fit_transform(R_SUMMARY)
    
    #instantiate Count Vectorizer for Student answer summary and map the student answer summary 
    #to the reference answer vectors
    ssumbow = rsumvect.transform(S_SUMMARY)
    
    #CALCULATE COSINE SIMILARITY OF REFANS AND STUANS BAG OF WORDS
    
    SUMMARY_SIMILARITY = cosine_similarity(rsumbow,ssumbow)
    SUMMARY_SIMILARITY =  SUMMARY_SIMILARITY[0][0]
    
    return SUMMARY_SIMILARITY





def get_metrics_sta(RA,SA):
    RA = RA
    SA = SA
    ST_SIM = statistical_similarity(RA,SA)
    return ST_SIM

def get_metrics_bow(RA,SA):
    RA = RA
    SA = SA
    BOW_SIM = BOW_similarity(RA,SA)
    return BOW_SIM

def get_metrics_NSW(RA,SA):
    RA = RA
    SA = SA
    NO_SW_SIM = NO_STOP_WORDS_similarity(RA, SA)
    return  NO_SW_SIM 

def get_metrics_LEMMA(RA,SA):
    RA = RA
    SA = SA
    LEMMA_SIM = LEMMA_SIMILARITY(RA,SA)
    return LEMMA_SIM

def get_metrics_TFIDF(RA,SA):
    RA = RA
    SA = SA
    TFIDF_SIM = TFIDF_SIMILARITY(RA,SA)
    return TFIDF_SIM

def get_metrics_LSA(RA,SA):
    RA = RA
    SA = SA
    LSA_SIM = LSA_SIMILARITY(RA,SA)
    return LSA_SIM

def get_metrics_SEM(RA,SA):
    RA = RA
    SA = SA
    SEM_SIM = doc2vec_similarity(RA,SA)
    return SEM_SIM

def get_metrics_SUM(RA,SA):
    RA = RA
    SA = SA
    SUM_SIM = SUMMARY_SIMILARITY(RA,SA)
    return SUM_SIM


def Extractsimilarities(data):
    df = data
    STA_sim = []
    BOW_sim = []
    NO_SW_sim = []
    LEMMA_sim = []
    TFIDF_sim = []
    LSA_sim = []
    SEM_sim = []
    SUM_sim = []

    for (index,rows) in df.iterrows():
     # Access individual elements of each row
     RefA = rows['REFANS']
     StuA = rows['STUANS']
       
     #Calling various get_metrics functions for calculating similarities
     STA_SIM = get_metrics_sta(RefA,StuA)
     BOW_SIM = get_metrics_bow(RefA,StuA)
     NO_SW_SIM = get_metrics_NSW(RefA,StuA)
     LEMMA_SIM = get_metrics_LEMMA(RefA,StuA)
     TFIDF_SIM = get_metrics_TFIDF(RefA,StuA)
     LSA_SIM = get_metrics_LSA(RefA,StuA)
     SEM_SIM = get_metrics_SEM(RefA,StuA)
     SUM_SIM = get_metrics_SUM(RefA,StuA)

     #Append all similarity values to the list of similarities  
     STA_sim.append(STA_SIM)
     BOW_sim.append(BOW_SIM)
     NO_SW_sim.append(NO_SW_SIM)
     LEMMA_sim.append(LEMMA_SIM)
     TFIDF_sim.append(TFIDF_SIM)
     LSA_sim.append(LSA_SIM)
     SEM_sim.append(SEM_SIM)
     SUM_sim.append(SUM_SIM)

    #Store each similarity list as a column in a dataframe
    df['sta_sim'] = STA_sim
    df['BOW_sim'] = BOW_sim
    df['No_SW_sim'] = NO_SW_sim
    df['LEMMA_sim'] = LEMMA_sim
    df['TFIDF_sim'] = TFIDF_sim
    df['LSA_sim'] = LSA_sim
    df['SEM_sim'] = SEM_sim
    df['SUM_sim'] = SUM_sim

    return df





