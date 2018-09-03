import os
import numpy as np
import nltk
import string
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import TweetTokenizer
import re
import pickle as pkl
import pandas as pd
import gensim
from gensim.models import Word2Vec

print('[INFO] Starting to load Word2Vec Model')
BASE = '../data/'
model_2 = gensim.models.KeyedVectors.load_word2vec_format(BASE + 'GoogleNews-vectors-negative300.bin', binary=True)

PATH = BASE + 'Real-life_Deception_Detection_2016/Transcription'
target_vocabulary = set()

print('[INFO] Generating vocabulary')
for D in ['Truthful', 'Deceptive']:
    files = os.listdir(PATH+'/'+D)
    for file in files:
        with open(PATH+'/'+D+'/'+file, 'r') as myfile:
            data_ = myfile.read()
        tokens = TweetTokenizer().tokenize(data_.lower())
        for token in tokens:
            target_vocabulary.add(token)


target_vocabulary = ['<PAD>'] + sorted(list(target_vocabulary))
print('[INFO] transforming sentence to index token representation')
sentences = []
filename = []
label = []
for D in ['Truthful', 'Deceptive']:
    files = os.listdir(PATH+'/'+D)
    for file in files:
        with open(PATH+'/'+D+'/'+file, 'r') as myfile:
            data_ = myfile.read()
        tokens = TweetTokenizer().tokenize(data_.lower())
        vect_ = []
        for token in tokens:
            try:
                vect = target_vocabulary.index(token)
                vect_.append(vect)
            except KeyError as e:

                continue
        sentences.append(np.array(vect_))
        filename.append(file[:-4]+'.mp4')
        label.append(D)

text_data = pd.DataFrame([filename, sentences, label]).T
text_data.columns = ['File', 'Embedding', 'Label']

text_data.set_index('File').to_pickle(BASE+'Text_Dataset.pkl')
