# Temp hacky workaround until pandas read_json encoding issue gets resolved in python 2.7
# https://github.com/pandas-dev/pandas/issues/15132
# The default output streams are captured and restored otherwise ipython cells wont show log messages.
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import spacy
from keras.utils.generic_utils import Progbar
from keras.utils import to_categorical

from keras_text.data import Dataset
from keras_text.processing import WordTokenizer, SentenceWordTokenizer, unicodify
from keras_text.utils import dump, load
from keras_text.models import TokenModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN


import load_data



MAX_TOKENS = 100

#==================================================================================================
# Pre-Processing
#==================================================================================================
# nlp = spacy.load('en')


# Load Data
print("=== Loading Dataset")
df = load_data.data(dataset='acllmdb')
text_raw = df.data['x']
labels = df.data['y']

# # Start Progress bar
# progbar = Progbar(len(text_raw), verbose=1, interval=0.25)

# # Getting all tokens
# token_all = []
# doc_token = []
# for doc_i, doc_text in enumerate(text_raw):
# 	doc = nlp(doc_text)
# 	token_list = []
# 	for token in doc:
# 		token_all.append(token.text)
# 		token_list.append(token.text)

# 	# Append to list
# 	doc_token.append(token.text)

# 	# Update progressbar per document level.
# 	progbar.update(doc_i)

# # All done. Finalize progressbar.
# progbar.update(len(text_raw), force=True)

# print("len(token_all)")
# print(len(token_all))
# print("len(doc_token)")
# print(len(doc_token))
# # print(df.data['x'].split()[0:10])



#==================================================================================================
# Tokenizer
#==================================================================================================

BUILD_DATASET = False

if BUILD_DATASET:
	print("=== Preview data")
	pprint(text_raw[0:4])

	print("=== Configuring tokenization scheme")
	tokenizer = WordTokenizer(
		lang='en',
		lower=True,
		lemmatize=False,
		remove_punct=True,
		remove_digits=True,
		remove_stop_words=False,
		exclude_oov=False,
		exclude_pos_tags=None,
		exclude_entities=['PERSON'])

	# Build vocabulary with training text
	print("=== Building vocabulary with training text")
	tokenizer.build_vocab(texts=text_raw, verbose=1, n_threads=4, batch_size=1000)
	# tokenizer.apply_encoding_options(max_tokens=MAX_TOKENS)


	print("=== Building dataset")
	ds = Dataset(text_raw, labels, tokenizer=tokenizer)
	ds.update_test_indices(test_size=0.1)

	print("=== Saving dataset")
	ds.save('../data/dataset/yelp_sent_dataset')
else:
	ds = load('../data/dataset/yelp_sent_dataset')


print("=== Dataset Stats")
print(ds.tokenizer.get_stats(0))

print("=== Dataset keys")
print(vars(ds).keys())

print("=== Tokenizer keys")
print(vars(ds.tokenizer).keys())

# RNN models can use `max_tokens=None` to indicate variable length words per mini-batch.
factory = TokenModelFactory(1, ds.tokenizer.token_index, max_tokens=MAX_TOKENS, embedding_type='glove.6B.100d')
word_encoder_model = YoonKimCNN()
model = factory.build_model(token_encoder_model=word_encoder_model)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()

print("=== Factory keys")
print(vars(factory).keys())
# print("=== factory.embeddings_index")
# print(factory.embeddings_index)

print("=== Model keys")
print(vars(model).keys())



#=========================================================
#=========================================================
#=========================================================

# print("ds.tokenizer.token_index")
# print(ds.tokenizer.token_index)
# # print(ds.tokenizer.token_index.shape)

# print("=== Ebedding Text")
# texts = [
#     "HELLO world hello. How are you today? Did you see the S.H.I.E.L.D?",
#     "Quick brown fox. Ran over the, building 1234?",
# ]

# # ds.tokenizer = SentenceWordTokenizer()
# # ds.tokenizer.build_vocab(texts)
# # ds.tokenizer.apply_encoding_options(max_tokens=5)
# encoded = ds.tokenizer.encode_texts(texts)
# decoded = ds.tokenizer.decode_texts(encoded, inplace=False)
# w = 1

# decoded_int = []
# for doc_encoded in decoded:
# 	print(doc_encoded)
# 	decoded_int.append([ds.tokenizer.token_index[text] for text in doc_encoded])

# print(decoded_int)





#=========================================================
#=========================================================
#=========================================================






print("\t--- Tokenizing raw inference dataset")		
print(vars(ds.tokenizer).keys())
ds_infer = ds.tokenizer.encode_texts(text_raw)
ds_decode = ds.tokenizer.decode_texts(ds_infer)

dump(ds_decode, file_name = "../data/dataset/yelp_sent_dataset_encoded")
ds_decode = load('../data/dataset/yelp_sent_dataset_encoded')


print("============ ds_decode.shape")
# print(ds_decode.shape)

# print(ds_decode[:10])

# # Convert words to vector embeddings
# ds_embedding = np.array([])
# for i, sentence in enumerate(ds_decode):
# 	print('\t\tWorking on sentence %d' % (i + 1))
# 	sentence_embedding = np.array([]).reshape(0,100)
# 	for j, word in enumerate(sentence):
# 		word_embedding = factory.embeddings_index[word.encode()]
# 		print("word_embedding.shape")
# 		print(word_embedding.shape)
# 		if word_embedding is not None:
# 			sentence_embedding = np.vstack([sentence_embedding, [word_embedding]])
# 			print(sentence_embedding.shape)

# 	ds_embedding = np.append([ds_embedding], [sentence_embedding])


print("============ Converting into index")
progbar = Progbar(len(ds_decode), verbose=1, interval=0.25)
decoded_int = np.array([]).reshape(0,100)
for position, doc_encoded in enumerate(ds_decode):
	doc_zeros   = np.zeros(100)
	doc_decoded = [ds.tokenizer.token_index[text] for text in doc_encoded]
	doc_decoded = np.array(doc_decoded)
	doc_zeros[:min(doc_decoded.size, 100)] = doc_decoded[:min(doc_decoded.size, 100)]


	# decoded_int.append(doc_zeros)
	decoded_int = np.vstack([decoded_int, [doc_zeros]])

	# Update progressbar per document level.
	progbar.update(position)

# All done. Finalize progressbar.
progbar.update(len(ds_decode), force=True)

print("decoded_int.shape")
print(decoded_int.shape)

# Convert integer into binary shape
y_binary = to_categorical(ds.y)

# Fit model
model.fit(decoded_int, y_binary,
			epochs=20,
			batch_size=128)