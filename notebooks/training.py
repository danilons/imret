
# coding: utf-8

# In[1]:

import csv
import numpy as np


# In[2]:

def read_file(fname):
    labels = {'exclusive': 0, 'item': 1, 'sku': 2}
    content = []
    with open(fname, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';')
        for nn, row in enumerate(spamreader):
            if len(row) == 3 and nn > 0:
                content.append(({"preoffer": row[0], "item": row[1]}, labels[row[2]]))
    return content


# In[3]:

trainset = read_file("/Users/danilonunes/workspace/dataset/trainset.csv")
testset = read_file("/Users/danilonunes/workspace/dataset/testset.csv")
validset = read_file("/Users/danilonunes/workspace/dataset/validationset.csv")


# In[4]:

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence 


# In[5]:

tokenizer = Tokenizer(nb_words=20000)


# In[6]:

tokenizer.fit_on_texts([" ".join([text["preoffer"], text["item"]]) for text, _ in trainset])


# In[7]:

def pad_sequences(tokenizer, sequences, maxlen=5000):
    preoffer = sequence.pad_sequences(tokenizer.sequences_to_matrix([line["preoffer"] for line, _ in sequences]), maxlen=maxlen)
    item = sequence.pad_sequences(tokenizer.sequences_to_matrix([line["item"] for line, _ in sequences]), maxlen=maxlen)
    labels = np.zeros((len(item), 3), dtype=np.int32)
    for idx, (_, label) in enumerate(sequences):
        labels[idx, label] = 1.
    return preoffer, item, labels


# In[8]:

X_train_preoffer, X_train_item, y_train = pad_sequences(tokenizer, trainset)


# In[9]:

X_train_preoffer.shape, X_train_item.shape, y_train.shape


# In[10]:

X_val_preoffer, X_val_item, y_val = pad_sequences(tokenizer, validset)
X_test_preoffer, X_test_item, y_test = pad_sequences(tokenizer, testset)


# In[11]:

from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout, concatenate, Merge
from sklearn.cross_validation import train_test_split
from keras.layers.core import Reshape, Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model


# In[12]:

sequence_length = 5000
vocabulary_size = len(tokenizer.word_index) + 1
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5


# In[43]:

# this returns a tensor
preoffer = Input(shape=(sequence_length, ), dtype='int32')
item = Input(shape=(sequence_length, ), dtype='int32')
inputs = concatenate([preoffer, item])

embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=2 * sequence_length)(inputs)
reshape = Reshape((2 * sequence_length, embedding_dim,1))(embedding)

conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)
conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', init='normal', activation='relu', dim_ordering='tf')(reshape)

maxpool_0 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
maxpool_1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
maxpool_2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)

merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
flatten = Flatten()(merged_tensor)
# reshape = Reshape((3*num_filters,))(merged_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(output_dim=3, activation='softmax')(dropout)

# this creates a model that includes
model = Model(input=[preoffer, item], output=output)


# In[44]:

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])


model.fit([X_train_preoffer, X_train_item], 
          y_train, 
          batch_size=32, epochs=1, verbose=1, 
          callbacks=[checkpoint], 
          validation_data=([X_val_preoffer, X_val_item], y_val))  # starts training


# In[ ]:

model.save("weights.hdf5")


# In[ ]:

model.evaluate([X_test_preoffer, X_test_item], y_test, batch_size=32)

