#!/usr/bin/env python
# coding: utf-8

############################################## All packages
from ast import literal_eval
from collections import Counter
import glob, os
import numpy as np
import pandas as pd
from sklearn import preprocessing
# keras
import keras
from keras import backend as K
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
# NLTK
import nltk
from nltk.data import load
from nltk import word_tokenize
from nltk import StanfordTagger
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
tagdict = load('help/tagsets/upenn_tagset.pickle')
from statistics import mean

############################################## CONSTANT VALUES
MAXLEN = 100
BATCH_SIZE = 32
PAD_VALUE = 99
MAX_SENT_PAD = 50
MAX_POS_PAD = 2000

# column definition
label_col = 0
argfeat3_col = 6
argfeat6_col = 7
sentsum_col = 4
pos_col = 5

############################################## initialize POS label encoder
le = preprocessing.LabelEncoder()
le.fit(list(tagdict.keys()))

############################################## Customized keras metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

############################################## Helper functions

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def counter_pos(article):
    a =[]  
    for idx,sent_pos in enumerate(article):
        count_pos = Counter(sent_pos)
        a.append(dict(count_pos))
    return a
        
def pos_count_article(counter_result, pos_index):
    article_pos_count_array = np.zeros(shape=(MAXLEN,len(le.classes_)))
    for art_i,sent_pos_count in enumerate(counter_result):
        if art_i >= MAXLEN:        
            pass
        else:
            for pos_item in sent_pos_count:
                try:
                    item_idx = pos_index.index(pos_item)
                    article_pos_count_array[art_i,item_idx] = sent_pos_count.get(pos_item)
                except:
                    pass
    return article_pos_count_array

# POS padding
def pad_sent(sent):
    sent_le = le.transform(sent)
    if len(sent_le) > MAX_SENT_PAD:
        sent_le = sent_le[:MAX_SENT_PAD]
    sent_pos_padded = np.pad(np.array(sent_le), (0, MAX_SENT_PAD-len(sent_le)) , 'constant', constant_values=(PAD_VALUE))
    return sent_pos_padded

def pad_article(article):
    art_pos_pad = np.empty(shape=(MAXLEN, MAX_SENT_PAD))
    art_pos_pad.fill(PAD_VALUE)
    for i,sent in enumerate(article):
        if i < MAXLEN:
            try:
                art_pos_pad[i] = pad_sent(sent,MAX_SENT_PAD)   
            except:
                pass
    return art_pos_pad

# Helper function for padding
def padding_X(X):    
    return sequence.pad_sequences(X, maxlen=MAXLEN)

# Helper function to transform test data
def process_test_df(df,sent=False,pos_count=False,pos_pad=False,pos_seq=False,af3=False,af6=False):
    
    out = []

    # labels
    labels = df[label_col].values
    labels = pd.get_dummies(labels).to_numpy()

    # sent_sum
    if sent:
        x_sent = df[sentsum_col].apply(literal_eval)
        X_sent = padding_X(x_sent)
        out.append(X_sent)

    # pos count
    if pos_count:        
        x_pos = df[pos_col].apply(literal_eval)
        x_pos_list = [] 
        for x in x_pos: 
            art_pos = pos_count_article(counter_pos(x_pos[0]),list(le.classes_)).reshape(-1,1)
            x_pos_list.append(art_pos) 
        X_pos = np.stack(x_pos_list) 
        X_pos = X_pos.reshape(X_pos.shape[0],X_pos.shape[1]) 
        out.append(X_pos)
        
    # pos pad
    if pos_pad:        
        x_pos = df[pos_col].apply(literal_eval)
        x_pos_list = [] 
        for x in x_pos:
            art_pos = pad_article(x).reshape(-1,1)
            x_pos_list.append(art_pos)
        X_pos = np.stack(x_pos_list) 
        out.append(X_pos)
        
    # pos seq
    if pos_seq:        
        x_pos = df[pos_col].apply(literal_eval)
        x_pos_list = [] 
        for x in x_pos: 
            flatten = [item for sublist in x for item in sublist if item != '#']
            flatten = le.transform(flatten)
            if len(flatten) < MAX_POS_PAD:
                x = np.concatenate([flatten,np.zeros(MAX_POS_PAD - len(flatten))])
            else:
                x = flatten[:MAX_POS_PAD]
            x = np.array(x).reshape(-1,1)
            x_pos_list.append(x)
            X_pos = np.stack(x_pos_list) 
        out.append(X_pos)
        
    # argfeat
    if af3:
        x_argfeat3 = df[argfeat3_col].apply(literal_eval)
        X_argfeat3 = padding_X(x_argfeat3)
        out.append(X_argfeat3)
    
    if af6:
        x_argfeat6 = df[argfeat6_col].apply(literal_eval)
        X_argfeat6 = padding_X(x_argfeat6)
        out.append(X_argfeat6)
        
    return labels, out # out might need flatten when with one input

def select_files(path, startwith):
    list_of_files = []
    files = os.listdir(path)
    for file in files:
        if file.startswith(startwith):
            list_of_files.append(str(path)+str(file))
    return list_of_files

############################################## Import data
# use 1986 data as test data
list_of_files = select_files('/data/ProcessedNYT/','test')
print ('Import data...')
list_of_dfs = [pd.read_csv(file, sep='\t', header=None) for file in list_of_files]

############################################## Import models
print ('Import models...')

list_of_models = select_files('/project/ModelWeights/','')
list_of_models.remove('/project/ModelWeights/.ipynb_checkpoints')

############################################## model evaluation
print ('Start model evaluation...')
for model in list_of_models:
    
    print('\nEvaluating:', model.split('/')[-1])
    
    # reset bool values
    sent=False
    pos_count=False
    pos_pad=False
    pos_seq=False
    af3=False
    af6=False
    
    lm = keras.models.load_model(model, custom_objects={'f1_m':f1_m}, compile=False)
    lm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',f1_m])
    
    print('Loaded:',model)
    
    if 'af3' in model:
        af3=True
    if 'af6' in model:
        af6=True
    if 'sent' in model:
        sent=True
    if 'pos.' in model:
        pos_count=True
    if 'pos-count' in model:
        pos_count=True
    if 'pos-pad' in model:
        pos_pad=True
    if 'pos-seq' in model:
        pos_seq=True
     
    acclist=[]
    f1list=[]
    for idx,df in enumerate(list_of_dfs):

        print('Evaluating:',list_of_files[idx])
        y_test, out = process_test_df(df, sent=bool(sent), pos_count=bool(pos_count), pos_pad=bool(pos_pad), pos_seq=bool(pos_seq), af3=bool(af3), af6=bool(af6))
    
        _, acc, f1 = lm.evaluate(out, y_test, batch_size=BATCH_SIZE)
        acclist.append(acc)
        f1list.append(f1)
        print('\tEvaluating {}: \t- accuracy: {} \t- f1_m: {}'.format(list_of_files[idx].split('_')[1],acc,f1))
    acclist.pop(2)
    f1list.pop(2)
    print('Average cross-topic accuracy: {}\tf1 score: {}'.format(mean(acclist),mean(f1list)))
    del lm