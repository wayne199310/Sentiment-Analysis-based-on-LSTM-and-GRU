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
from sklearn.metrics import confusion_matrix
import numpy as np
from random import sample
import datetime

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from keras.layers import Dense, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.recurrent import GRU,LSTM
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
import pandas as pd
import pickle

r1 = '''[a-zA-Z0-9’"#$%&\'()*+-./:;<=>?@★、…【】　。◢▲◣▃●￣：\\n「」，～）\\\ （？《》“”‘’！[\\]^_`{|}~▲▼!▃▄●▍▎︶◥◤◢◣]+'''
output_dimen = 2

def __pickleStuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()
    
def __loadStuff(filename):
    saved_stuff = open(filename,"rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff
    
    
def jieba_cut_word(article_amount,data):
    article_list = []
    for i in range(article_amount):
        if i%100 == 0 : print('{}/{}'.format(i,article_amount))
        tem = data.loc[i,'article']
        
        # replace english
        tem = re.sub(r1, '', tem)
        if str( type(tem) ) != "<class 'NoneType'>" and tem != '' and tem != ' ':
            seg_list = jieba.cut(tem)  
            value = " ".join(seg_list)
            article_list.append(value)
    #article = " ".join(article_list).replace('\n','')
    article = " ".join(article_list).replace('\n','').replace('\\','').split(' ')
    print('corpus length:', len(article))
    return article

#---------------------------------------------------------------------------------  
def validation(model,val_article,sentiment_tag):

    val_X = []
    val_Y = [str(doc[1]).lower() for doc in val_article]
    
    input_tokenizer_load = __loadStuff("input_tokenizer_chinese.p")
    #val_data = np.array()
    for i in range(len(val_article)):
        if i%100 == 0: print('{}/{}'.format(i,len(val_article)))
        text = val_article[i]
        text = text[0].replace("\n", "")
        text = text.replace("\r", "") 
        text = re.sub(r1, '', text)
        seg_list = jieba.cut(text, cut_all=False)
        seg_list = list(seg_list)
        text = " ".join(seg_list)
        val_X.append( text )

    textArray = np.array(pad_sequences(
            input_tokenizer_load.texts_to_sequences(val_X), 
            maxlen = maxLength))
    
    predicted = model.predict(textArray,verbose=1,batch_size = 1024)
    pred_val = []
    for p in predicted:
        # p = predicted[0]
        p = p[0] # we have only one sentence to predict, so take index 0
        if p>0.5:
        #probab = p.max()
            pred_val.append(sentiment_tag[0])
        else:
            pred_val.append(sentiment_tag[1])
    
    val_Y2 = [ 1 if x == 'positive' else 0 for x in val_Y ]
    pred_val2 = [ 1 if x == 'positive' else 0 for x in pred_val ]
    
    plot_confusion_matrix(y_true = val_Y, 
                          y_pred = pred_val,
                          classes = sentiment_tag,
                          title='Confusion matrix')
    print('acc : {}'.format(metrics.accuracy_score(val_Y2, pred_val2)))
    print('recall : {}'.format(metrics.recall_score(val_Y2, pred_val2)))
    print('precision : {}'.format(metrics.precision_score(val_Y2, pred_val2)))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title = 'Confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    print(cm)

    fig, ax = plt.subplots(figsize = (10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    plt.xticks(fontsize = 32)
    plt.yticks(fontsize = 32)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),fontsize = 32,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix

def validation2(model1,totalX2,totalY,sentiment_tag):
        
    predicted = model1.predict(totalX2,verbose=1,batch_size = 1024)
    pred_val = []
    for p in predicted:
        # p = predicted[0]
        p = p[0] # we have only one sentence to predict, so take index 0
        #p = np.array(p)
        if p>0.5:
        #probab = p.max()
            pred_val.append(sentiment_tag[0])
        else:
            pred_val.append(sentiment_tag[1])

    plot_confusion_matrix(y_true = totalY, 
                          y_pred = pred_val,
                          classes = sentiment_tag,
                          title='Confusion matrix')
    
    print( confusion_matrix(totalY, pred_val) )
    
def build_model_two_layer_gru(vocab_size,totalX2, totalY2):
    embedding_dim = 256
    
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim,input_length = maxLength))
    # Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
    # All the intermediate outputs are collected and then passed on to the second GRU layer.
    model.add(GRU(256, dropout=0.9, return_sequences=True))
    model.add(GRU(256, dropout=0.9, return_sequences=True))
    # Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
    model.add(GRU(256, dropout=0.9))
    # The output is then sent to a fully connected layer that would give us our final output_dim classes
    model.add(Dense(2, activation='softmax'))
    model.summary()
    # We use the adam optimizer instead of standard SGD since it converges much faster
    tbCallBack = TensorBoard(log_dir='sentiment_chinese', histogram_freq=0,
                                write_graph=True, write_images=True)
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(
            totalX2, totalY2, validation_split=0.1, 
            batch_size=1024, epochs=10, verbose=1, 
            callbacks=[tbCallBack])
    model.save('build_model_two_layer_gru.HDF5')
    # loss: 0.0343 - acc: 0.9883 - val_loss: 0.2664 - val_acc: 0.9277
    print("Saved model!")
    #return history.history['val_acc'][9]

    return model

def build_model_two_layer_lstm(vocab_size,totalX2, totalY2):
    
    embedding_dim = 256
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim,input_length = maxLength))
    model.add(LSTM(256,dropout=0.9, return_sequences=True))
    model.add(LSTM(256,dropout=0.9, return_sequences=True))
    #model.add(GRU(256, dropout=0.9))
    model.add(LSTM(256,dropout=0.9))
    #model.add(Dense(64 ,  activation = 'relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()
    # We use the adam optimizer instead of standard SGD since it converges much faster
    tbCallBack = TensorBoard(log_dir='sentiment_chinese', histogram_freq=0,
                                write_graph=True, write_images=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(totalX2, totalY2, validation_split=0.3, 
                        batch_size=1024, 
                        epochs=10, verbose=1, 
                        callbacks=[tbCallBack])
    model.save('build_model_two_layer_lstm.HDF5')
    
    return model


#------------------------------------------------------------------------------------------
model = None
sentiment_tag = None
maxLength = None
def load_two_layer_gru():
    global model, sentiment_tag, maxLength
    metaData = __loadStuff("meta_sentiment_chinese.p")
    maxLength = metaData.get("maxLength")
    vocab_size = metaData.get("vocab_size")
    output_dimen = metaData.get("output_dimen")
    sentiment_tag = metaData.get("sentiment_tag")
    embedding_dim = 256
    if model is None:
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=maxLength))
        # Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
        # All the intermediate outputs are collected and then passed on to the second GRU layer.
        model.add(GRU(256, dropout=0.9, return_sequences=True))
        model.add(GRU(256, dropout=0.9, return_sequences=True))
        # Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
        model.add(GRU(256, dropout=0.9))
        # The output is then sent to a fully connected layer that would give us our final output_dim classes
        model.add(Dense(output_dimen, activation='softmax'))
        # We use the adam optimizer instead of standard SGD since it converges much faster
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights('build_model_two_layer_gru.HDF5')
        #model.summary()
    print("Model weights loaded!")
    return model

def load_two_layer_lstm():
    global model, sentiment_tag, maxLength
    metaData = __loadStuff("meta_sentiment_chinese.p")
    maxLength = metaData.get("maxLength")
    vocab_size = metaData.get("vocab_size")
    output_dimen = metaData.get("output_dimen")
    sentiment_tag = metaData.get("sentiment_tag")
    embedding_dim = 256
    if model is None:
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=maxLength))
        # Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
        # All the intermediate outputs are collected and then passed on to the second GRU layer.
        model.add(LSTM(256, dropout=0.9, return_sequences=True))
        model.add(LSTM(256, dropout=0.9, return_sequences=True))
        # Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
        model.add(LSTM(256, dropout=0.9))
        # The output is then sent to a fully connected layer that would give us our final output_dim classes
        model.add(Dense(output_dimen, activation='softmax'))
        # We use the adam optimizer instead of standard SGD since it converges much faster
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights('build_model_two_layer_lstm.HDF5')
        #model.summary()
    print("Model weights loaded!")
    return model

def findFeatures(text):
    #text = langconv.Converter('zh-hans').convert(text)
    text = text.replace("\n", "")
    text = text.replace("\r", "") 
    text = re.sub(r1, '', text)
    seg_list = jieba.cut(text, cut_all=False)
    seg_list = list(seg_list)
    text = " ".join(seg_list)
    textArray = [text]
    input_tokenizer_load = __loadStuff("input_tokenizer_chinese.p")
    textArray = np.array(pad_sequences(
            input_tokenizer_load.texts_to_sequences(textArray), 
            maxlen=maxLength))
    return textArray

def predictResult(model,text):
    if model is None:
        print("Please run \"loadModel\" first.")
        return None
    features = findFeatures(text)
    predicted = model.predict(features)[0] # we have only one sentence to predict, so take index 0
    predicted = np.array(predicted)
    probab = predicted.max()
    predition = sentiment_tag[predicted.argmax()]
    return predition, probab

'''

'''
#---------------------------------------------------------------------------------
def movie_score(
        movie2,model,sentiment_tag,keyword = ['復仇者','復'],article_type = ''):
    # movie2 = movie
    if isinstance(keyword,str):
        keyword = [keyword]
    if article_type:
        _bool = []
        for at in list( movie2['article_type'] ):
            # 拿 title
            if article_type in at:
                _bool.append(True)
            else:
                _bool.append(False)
        movie2 = movie2[_bool]
    _bool = []
    for title in list( movie2['title'] ):
        # 拿 title
        amount = [ 1 if k in title else 0 for k in keyword  ]
        amount = np.sum(amount)
        if amount>0:
            _bool.append(True)
        else:
            _bool.append(False)
    movie2 = movie2[_bool]
    movie2.index = range(len(movie2))
    
    article = list( movie2['article'] )
    val_X = []
    input_tokenizer_load = __loadStuff("input_tokenizer_chinese.p")
    #val_data = np.array()
    # 清除標點符號, jieba 斷詞
    for i in range(len(article)):
        if i%100 == 0: print('{}/{}'.format(i,len(article)))
        text = article[i]
        text = text.replace("\n", "")
        text = text.replace("\r", "") 
        text = re.sub(r1, '', text)
        seg_list = jieba.cut(text, cut_all=False)
        seg_list = list(seg_list)
        text = " ".join(seg_list)
        val_X.append( text )
    # 向量化
    textArray = np.array(pad_sequences(
            input_tokenizer_load.texts_to_sequences(val_X), 
            maxlen = maxLength))
        
    predicted = model.predict(textArray,verbose=1,batch_size = 1024)
    
    if sentiment_tag[0] == 'positive':
        index = 0
    else:
        index = 1
    score = 0
    for p in predicted:
        score = score + p[index]
    score = score/len(predicted)

    pred_val = []
    for p in predicted:
        # p = predicted[0]
        p = p[0] # we have only one sentence to predict, so take index 0
        if p>0.5:
            pred_val.append(sentiment_tag[0])
        else:
            pred_val.append(sentiment_tag[1])
    
    happy_score = []
    hate_score = []
    #none_score = []
    for x in pred_val:
        #    none_score.append(1)
        if x == 'negative':
            hate_score.append(1)
        elif x == 'positive':
            happy_score.append(1)
    
    print('hate amount : {}'.format( np.sum(hate_score) ) )
    print('happy amount : {}'.format( np.sum(happy_score) ) )
    #print('none amount : {}'.format( np.sum(none_score) ) )
    print( 'score : {}'.format(score) )
    return predicted
    #return np.sum(happy_score)/len(predicted)

def demo(model,test_article,i):
    
    val_X = []
    val_Y = [str(doc[1]).lower() for doc in test_article]
    
    input_tokenizer_load = __loadStuff("input_tokenizer_chinese.p")
    #val_data = np.array()
    text = test_article[i]
    print('origin article : \n{}'.format(text[0]))
    text = text[0].replace("\n", "")
    text = text.replace("\r", "") 
    text = re.sub(r1, '', text)
    
    seg_list = jieba.cut(text, cut_all=False)
    seg_list = list(seg_list)
    text = " ".join(seg_list)
    print(text)
    val_X.append( text )

    textArray = np.array(pad_sequences(
            input_tokenizer_load.texts_to_sequences(val_X), 
            maxlen = maxLength))
    print(textArray)
    predicted = model.predict(textArray,verbose=1,batch_size = 1024)
    pred_val = []
    for p in predicted:
        # p = predicted[0]
        p = p[0] # we have only one sentence to predict, so take index 0
        if p>0.5:
        #probab = p.max()
            pred_val.append(sentiment_tag[0])
        else:
            pred_val.append(sentiment_tag[1])
    
    #val_Y2 = [ 1 if x == 'positive' else 0 for x in val_Y ]
    #pred_val2 = [ 1 if x == 'positive' else 0 for x in pred_val ]
    print('predict : {}'.format( pred_val[0] ) )
    print('label : {}'.format( val_Y[i] ) )

    

def main():
    
    
    #---------------------------------------------------------------------------------
    print('load data')
    happy = PTT.LoadData(table = 'happy',start = '2010-01-01',end = '2018-01-01',
                         select = 'article')
    
    print( 'happy article : {}'.format(len(happy)) )
    # 23673
    
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
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
    
    pl.plot(h,fit,'-o')
    pl.hist(h,normed=True)      #use this to draw histogram of your data
    pl.show()
    maxlen = maxLength
    #---------------------------------------------------------------------------------
    
    print('word embedding, build dictionary of target')
    input_tokenizer = Tokenizer(30000) 
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
    #---------------------------------------------------------------------------------
    
    print( 'save sentiment' )
    target_reverse_word_index = {v: k for k, v in list(target_tokenizer.word_index.items())}
    sentiment_tag = [target_reverse_word_index[1],target_reverse_word_index[2]] 
    metaData = {"maxLength":maxLength,"vocab_size":vocab_size,"output_dimen":output_dimen,"sentiment_tag":sentiment_tag}
    __pickleStuff("meta_sentiment_chinese.p", metaData)
    
    #--------------    ----------------------------
    print('====================================================================')
    print('model1')
    s = datetime.datetime.now()
    model1 = build_model_two_layer_gru(vocab_size,totalX2, totalY2)
    t1 = datetime.datetime.now() - s
    print(t1)
    
    #print('train:')
    #validation2(model1,totalX2,totalY,sentiment_tag = sentiment_tag)
    
    print('test:')
    validation(model1,val_article = test_article,sentiment_tag = sentiment_tag)
    '''
    [[2943  219]
     [ 316 2803]]
    acc : 0.9148224804967362
    recall : 0.8986854761141392
    precision : 0.9275314361350099
    '''
    
    demo(model1,test_article,i = 15)
    
    #--------------    ----------------------------
    print('====================================================================')
    print('model4')
    s = datetime.datetime.now()
    model4 = build_model_two_layer_lstm(vocab_size,totalX2, totalY2)
    t4 = datetime.datetime.now() - s
    print(t4)
    
    #print('train:')
    #validation2(model4,totalX2,totalY,sentiment_tag = sentiment_tag)
    
    print('test:')
    validation(model4,test_article,sentiment_tag = sentiment_tag)
    '''
    [[2828  334]
     [ 339 2780]]
    acc : 0.892851456774399
    recall : 0.8913113177300417
    precision : 0.892742453436095
    '''

    
    gru_model = load_two_layer_gru()
    lstm_model = load_two_layer_lstm()
    
    #---------------------------------------------------------------------------------
    
    val_good_movie = PTT.LoadData(table = 'movie',
                              start = '2018-01-01',end = '2019-01-01',
                         select = ['article'],article_type = '好雷')
    print( 'good_movie : {}'.format(len(val_good_movie)) )
    
    val_bad_movie = PTT.LoadData(table = 'movie',
                             start = '2018-01-01',end = '2019-01-01',
                         select = ['article'],article_type = '負雷')
    print( 'bad_movie : {}'.format(len(val_bad_movie)) )
        
    val_good_movie_article = list( val_good_movie['article'] )
    val_bad_movie_article = list( val_bad_movie['article'] )
    
    print(' bind data and label' )
    val_article = []
    [ val_article.append((h,'positive')) for h in val_good_movie_article if h  ]
    [ val_article.append((h,'negative')) for h in val_bad_movie_article if h ]
    #sentiment_tag2 = ['好雷','負雷']
    
    validation(gru_model,val_article,sentiment_tag = sentiment_tag)
    validation(lstm_model,val_article,sentiment_tag = sentiment_tag)

    #--------------------------------------------------
    movie = PTT.LoadData(table = 'movie',
                         start = '2018-01-01',
                         end = '2020-01-01',
                         select = ['article','title','date','article_type'])
    
    predicted = movie_score(movie2 = movie,
                         sentiment_tag = sentiment_tag,
                         model = gru_model,
                         keyword = ['復仇者'],
                         article_type = '好雷')# imdb 8.8/10
    
    predicted = movie_score(
            movie2 = movie,
            model = gru_model,
            sentiment_tag = sentiment_tag,
            keyword = ['驚奇隊長']
            ,article_type = '雷'
            )# imdb 7.1/10
    
    predicted = movie_score(movie,gru_model,sentiment_tag = sentiment_tag,keyword = '皮卡丘')
    
    predicted = movie_score(movie,gru_model,sentiment_tag = sentiment_tag,keyword = '比悲傷')# imdb 5.5
    
    
main()