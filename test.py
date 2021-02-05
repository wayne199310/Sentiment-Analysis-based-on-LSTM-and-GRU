# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 20:20:48 2019

@author: Wayne
"""

from PTTData import Load as PTT


def main():

    print('load data')
    
    happy = PTT.LoadData(table = 'happy',start = '2010-01-01',end = '2018-01-01',
                         select = 'article')
    
    print( 'happy article : {}'.format(len(happy)) )
    # 23673
    
    Hate = PTT.LoadData(table = 'Hate',start = '2017-09-01',end = '2018-01-01',
                         select = 'article')
    print( 'Hate article : {}'.format(len(Hate)) )
    # 28287
    
    good_movie = PTT.LoadData(table = 'movie',start = '2018-01-01',end = '2018-12-31',
                         select = ['article'],keyword = ['黑豹'],article_type = '好雷')
    print( 'good_movie : {}'.format(len(good_movie)) )
    # 2016/01/01-2018/01/01
    # 7306
    bad_movie = PTT.LoadData(table = 'movie',start = '2018-01-01',end = '2018-12-31',
                         select = ['article'],article_type = '負雷')
    print( 'bad_movie : {}'.format(len(bad_movie)) )
    # 2013/01/01-2018/01/01
    # 3540
    
   