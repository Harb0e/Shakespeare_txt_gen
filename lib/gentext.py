# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import lib.init as init
import lib.model as model
import lib.train as train


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences





"""# Datasæt er fået fra https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays?select=alllines.txt"""

def trainmodel():
  token = init.TokenShake()
  token.initialize()
  nmodel = model.createModel(token.getInputSize(),token.getDf(),token.getTotalWords())
  
  # Train model
  tm = train.ShTrainer().trainMe(nmodel,token.getX(),token.getY())
  





def get(start_string,number_of_words):
  '''
  henter den trænet model og bruger den til at predicted det næste ord
  '''

  gen_string = start_string
  #loader model
  genmodel = keras.models.load_model("lib\model")
  initobj = init.TokenShake()
  initobj.initialize()
  #henter den tokenizer der blev brugt til at lave sætningerne om til tal
  tokenizer = initobj.getTokenizer()
  #henter dictionariet 
  word_idx = tokenizer.word_index
  key_list = list(word_idx.keys())
  val_list = list(word_idx.values())

  for words in range(number_of_words):
    #laver start string om til tal
    token_string = tokenizer.texts_to_sequences([gen_string])[0]
    #padder arrayet så dimentionen passer med inputlayeret i modellen
    pad_string = pad_sequences([token_string],maxlen=initobj.getMaxlen()-1)
    #sender arrayet igennem modellen og tager det ord der har højst chance for at forekomme
    pre = np.argmax(genmodel.predict(pad_string),-1)
    #laver tallet om til ord
    position = val_list.index(pre[0])
    gen_string = gen_string + ' ' + key_list[position]
  return gen_string

