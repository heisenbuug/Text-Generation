#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:11:26 2020

@author: heisenbug
"""



import tensorflow as tf
import tensorflow_datasets as tfdb
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

tokenizer = Tokenizer()

data = open('/home/heisenbug/Workspace/TensorFlow/Lesson7/adele.txt').read()
          
corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequence = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequence.append(n_gram_sequence)
        
max_sequence_len = max([len(x) for x in input_sequence])
input_sequence = np.array(pad_sequences(input_sequence,
                                        maxlen = max_sequence_len,
                                        padding = 'pre'))

xs = input_sequence[:, :-1]
labels = input_sequence[:, -1]

ys = tf.keras.utils.to_categorical(labels, num_classes = total_words)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(total_words, 100, input_length = max_sequence_len - 1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
model.add(tf.keras.layers.Dense(total_words, activation = 'softmax'))

adam = Adam(lr = 0.01)

model.compile(loss = 'categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'])

model.fit(xs, 
          ys,
          epochs = 100,
          verbose = 1)

seed_text = 'You are my life'
next_words = 100

for _ in range(next_words):
    foo = tokenizer.texts_to_sequences([seed_text])[0]
    foo_padded = pad_sequences([foo],
                                maxlen = max_sequence_len - 1,
                                padding = 'pre')
    
    predicted = model.predict_classes(foo_padded, verbose = 0)
    output_word = ' '
    
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += ' ' + output_word
    
print(seed_text)