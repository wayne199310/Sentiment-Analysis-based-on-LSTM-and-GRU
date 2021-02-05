#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:11:36 2019

@author: Wayne
"""

from PTTData import Load as PTT
import re
import jieba
import scipy.stats as stats
import pylab as pl
#from snownlp import SnowNLP
import numpy as np
#--------------------------------------
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
#--------------------------------------
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import pickle


r1 = '''[a-zA-Z0-9’"#$%&\'()*+-./:;<=>?@★、…【】　。◢▲◣▃●￣：\\n「」，～）\\\ （？《》“”‘’！[\\]^_`{|}~▲▼!▃▄●▍▎︶◥◤◢◣]+'''
def __pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()
    
def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


def main():
    
    
    #---------------------------------------------------------------------------------
    print('load data')
    happy = PTT.LoadData(table = 'happy',start = '2010-01-01',end = '2018-01-01',
                         select = 'article')
    
    print( 'happy article : {}'.format(len(happy)) )
     #23673
    
    Hate = PTT.LoadData(table = 'Hate',start = '2017-09-01',end = '2018-01-01',
                         select = 'article')
    print( 'Hate article : {}'.format(len(Hate)) )
    # 28287
    
    good_movie = PTT.LoadData(table = 'movie',start = '2016-01-01',end = '2018-01-01',
                         select = ['article'],article_type = '好雷')
    print( 'good_movie : {}'.format(len(good_movie)) )
    # 7306
    bad_movie = PTT.LoadData(table = 'movie',start = '2013-01-01',end = '2018-01-01',
                         select = ['article'],article_type = '負雷')
    print( 'bad_movie : {}'.format(len(bad_movie)) )
    # 3540
    
    happy_article = list( happy['article'] )
    Hate_article = list( Hate['article'] )
    good_movie_article = list( good_movie['article'] )
    bad_movie_article = list( bad_movie['article'] )
    #---------------------------------------------------------------------------------
    print(' bind data and label' )
    article = []
    ''' training data '''
    [ article.append((h,'positive')) for h in happy_article if h  ]
    [ article.append((h,'negative')) for h in Hate_article if h ]
    [ article.append((h,'positive')) for h in good_movie_article if h  ]
    [ article.append((h,'negative')) for h in bad_movie_article if h  ]
    
    print(' train_test_split' )
    train_article,test_article = train_test_split(
            article,test_size = 0.1,random_state = 100)
    
    #---------------------------------------------------------------------------------
    
    print( ' clean and cut article by jieba' )
    totalX = []
    totalY = [str(doc[1]).lower() for doc in train_article]
    for doc in train_article:# doc = train_article[0]
        tex = re.sub(r1, '', doc[0])
        seg_list = jieba.cut(tex, cut_all=False)
        seg_list = list(seg_list)
        totalX.append(seg_list)
    
    print('get max length of article')
    h = sorted([len(sentence) for sentence in totalX])
    maxLength = h[int(len(h) * 0.60)]
    print("Max length is: ",h[len(h)-1])
    print("60% cover length up to: ",maxLength)
    #h = h[:5000]
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  
    
    pl.plot(h,fit,'-o')
    pl.hist(h,normed=True)      #histogram
    pl.show()
    maxlen = maxLength
    #---------------------------------------------------------------------------------
    
    print('word embedding, build dictionary of target')
    input_tokenizer = Tokenizer(30000) #num_words:None或整数,处理的最大单词数量。
    input_tokenizer.fit_on_texts(totalX)
    vocab_size = len(input_tokenizer.word_index) + 1
    input_tokenizer.word_index
    print("input size:",vocab_size)
    totalX2 = np.array(pad_sequences(
            input_tokenizer.texts_to_sequences(totalX), 
            maxlen=maxlen))
    #totalX[0]
    #totalX2[0]
    print( 'save tokenizer' )
    __pickleStuff("input_tokenizer_chinese.p", input_tokenizer)
    #---------------------------------------------------------------------------------
    print('target embedding')
    target_tokenizer = Tokenizer(3)
    target_tokenizer.fit_on_texts(totalY)
    target_tokenizer.word_index
    print("output vocab_size:",len(target_tokenizer.word_index) + 1)
    totalY2 = np.array(target_tokenizer.texts_to_sequences(totalY)) -1
    totalY2 = totalY2.reshape(totalY2.shape[0])
    
    totalY2 = to_categorical(totalY2, num_classes=2)
    output_dimen = totalY2.shape[1]
    print('output_dimension : {}'.format(output_dimen))
    


    main()