#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:


import itertools
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils


# In[3]:


a = pd.read_csv('array.csv')
p = pd.read_csv('pointers.csv')
f = pd.read_csv('function.csv')
l = pd.read_csv('loop.csv')
vali = pd.read_csv('TensorFlowData.csv')

frames = [a, p, f, l, vali]

data = pd.concat(frames)

#data = pd.read_csv('teste.csv')


# In[4]:


data


# In[5]:


data['Type'].value_counts()


# In[6]:


# Split data into train and test
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))


# In[7]:


train_posts = data['Title'][:train_size]
train_tags = data['Type'][:train_size]

test_posts = data['Title'][train_size:]
test_tags = data['Type'][train_size:]


# In[8]:


max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)


# In[9]:


tokenize.fit_on_texts(train_posts) # only fit on train
x_train = tokenize.texts_to_matrix(train_posts)
x_test = tokenize.texts_to_matrix(test_posts)


# In[10]:


# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)


# In[11]:


# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


# In[12]:



# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# In[14]:


batch_size = 512
epochs = 600


# In[15]:


# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#dpt_model = keras.models.Sequential([
   # keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(max_words,)),
   # keras.layers.Dropout(0.5),
   # keras.layers.Dense(512, activation=tf.nn.relu),
   # keras.layers.Dropout(0.5),
  #  keras.layers.Dense(1, activation=tf.nn.sigmoid)
#])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'binary_crossentropy'])


# In[16]:


# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=0.2)


# In[17]:


# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Pontuação do teste:', score[0])
print('Teste de acurácia:', score[1])


# In[52]:


# Here's how to generate a prediction on individual examples
text_labels = encoder.classes_ 

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print(test_posts.iloc[i][:50], "...")
    print('Rótulo real:' + test_tags.iloc[i])
    print("Rótulo previsto: " + predicted_label + "\n")


# In[53]:


y_softmax = model.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)


# In[54]:


# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Rótulo real', fontsize=15)
    plt.xlabel('Rótulo previsto', fontsize=15)


# In[55]:


cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(10,5))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Matriz de Confusão")
plt.show()


# In[56]:


history_dict = history.history
history_dict.keys()


# In[57]:


import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Perda do treino')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Perda da validação')
plt.title('Perca do treino e da validação')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.show()


# In[58]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Treino acc')
plt.plot(epochs, val_acc, 'b', label='Validação acc')
plt.title('Acurácia do treino e da validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.show()


# In[ ]:




