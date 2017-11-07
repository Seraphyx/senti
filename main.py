import tensorflow as tf
import keras

'''
Attention with Context:
https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

Attention Temporal:
https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d

Attentional Dense Layers:
https://github.com/philipperemy/keras-attention-mechanism

Using pre-trained word embeddings in a Keras model:
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

Blog on Word2Vec and Keras:
http://ben.bolte.cc/blog/2016/gensim.html


1) We want to tokenize each string to get a list of words, usually by making everything lowercase 
and splitting along the spaces. In contrast, lemmatization involves getting the root of each word, 
which can be helpful but is more computationally expensive (enough so that you would want to 
preprocess your text rather than do it on-the-fly).
2) Vectorize words
3) Build Model
4) Grid Search
5) Save


'''

# Import custom layers
import layers
import text