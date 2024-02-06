# -*- coding: utf-8 -*-
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np


class TokenShake():

    def initialize(self) -> None:
        self.path = '.\\lib\\data\\alllines.txt'
        #loader dataen ind i et pandas dataframe hvor dataen bliver delt mellem hver sætning
        self.data = pd.read_csv(self.path,sep = " ",header = None)
        #laver det om til numpy array
        self.sentences = self.data[0].to_numpy()
        #tager de første 10000 sætninger
        self.training_sens = self.sentences[:10000]
        self.tokenizer = Tokenizer()
        #fitter tokenizeren, dette gør tokenizeren danner en dictionary hvor alle ord er forbundet med et tal
        self.tokenizer.fit_on_texts(self.training_sens)
        self.word_idx = self.tokenizer.word_index
        #laver sætningerne om til arrays med tal så det kan komme igennem modellen
        self.word_num = self.tokenizer.texts_to_sequences(self.training_sens)
        self.df = self.create_sets(self.word_num)
        #finder den længste sætning i datasættet
        self.max_len = max([len(x) for x in self.df])
        #padder hvilket tilføjer nuler så alle sequences er lige lange
        df_padded = np.array(pad_sequences(self.df,maxlen = self.max_len))
        self.X , y = self.df_to_X_y(df_padded)
        #gør så y er categorical
        self.y = to_categorical(y, num_classes=len(self.word_idx)+1)
        _, self.input_size = self.X.shape
        self.total_words = len(self.word_idx)+1    
        
    def getInputSize(self):
        return self.input_size

    def getwordlist(self):
        return self.word_idx
    
    def getTotalWords(self):
        return self.total_words
    
    def getDf(self):
        return self.df

    def getTokenizer(self):
        return self.tokenizer
    
    def getX(self):
        return self.X
    
    def getY(self):
        return self.y

    def getMaxlen(self):
        return self.max_len

    def create_sets(self,nums):
        '''
        vi vil få netværket til at forudsige det næste ord derfor skal vi have en liste med de forige ord plus det ord den skal forudsige
        '''

        lst = []
        for sentence in nums:
            for i in range(1,len(sentence)):
                lst.append(sentence[:i+1])
        
        return lst

    def df_to_X_y(self,df):
        '''
        laver vores dataen om så vi kan lave supervised learning.
        '''

        X = []
        y = []
        for i in range(len(df)):
            X.append(np.array(df[i][:-1]))
            y.append(df[i][-1])
        return np.array(X), np.array(y)