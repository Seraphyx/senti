from __future__ import unicode_literals

import sys
import os
from pprint import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import load_model

from keras_text.processing import WordTokenizer, SentenceWordTokenizer
from keras_text.data import Dataset
from keras_text.models.token_model import TokenModelFactory
from keras_text.models.sentence_model import SentenceModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN
from keras_text.utils import dump, load


import load_data

data_base = '../data'

MAX_SENTS = 100
MAX_TOKENS = 100


# Overcome pickle recurssion limit
sys.setrecursionlimit(10000)

'''
See: https://raghakot.github.io/keras-text/
* You may need to fix the imports in keras_text to have a period in front in the model/__init__.py file
* For Python 3 you must make the dictionary.values() to be wrapped in list()
* If you cannot download the SpaCy en model in Windows 10 then run as admin
* If you hit a recussion limit when using utils.dump then set a higher limit: sys.setrecursionlimit(10000)
'''

# tokenizer = WordTokenizer()
# tokenizer.build_vocab(texts)


def initialize_dir(root_dir):

	folder_list = ['dataset','embeddings','models']

	for subfolder in folder_list:
		subfolder_path = os.path.join(root_dir, subfolder)
		if not os.path.exists(subfolder_path):
			print("\tMaking folder %s" % subfolder)
			os.makedirs(subfolder_path)


def doc_to_sentence(doc_token_list, factory, max_sents, max_tokens):

	# Count how many sentences
	n_doc = len(doc_token_list)

	# Initialize
	ds_embedding = np.zeros((n_doc, max_sents, max_tokens))

	# Convert words to vector embeddings
	for i, doc in enumerate(doc_token_list):
		if i + 1 > n_doc:
			break
		for j, sentence in enumerate(doc):
			if j + 1 > max_sents:
				break
			for k, word in enumerate(sentence):
				if k + 1 > max_tokens:
					break
				ds_embedding[i, j, k] = word

	return ds_embedding


def doc_to_token(doc_token_list, factory, max_tokens):

	# Count how many sentences
	n_doc = len(doc_token_list)

	# Initialize
	ds_embedding = np.zeros((n_doc, max_tokens))

	# Convert words to vector embeddings
	for doc_i, doc in enumerate(doc_token_list):

		token_i = 0
		for j, sentence in enumerate(doc):
			if token_i + 1 > max_tokens:
				break;
			for token in sentence:
				if token_i + 1 > max_tokens:
					break;
				ds_embedding[doc_i, token_i] = token
				token_i += 1

	return ds_embedding


def doc_to_embedding(doc_token_list, factory, max_tokens):

	# Count how many sentences
	n_doc = len(doc_token_list)

	# Initialize
	ds_embedding = np.zeros((n_doc, max_tokens, factory.embedding_dims))

	# Convert words to vector embeddings
	for doc_i, doc in enumerate(doc_token_list):
		for j, sentence in enumerate(doc):
			token_i = 0
			sentence_embedding = np.zeros((max_tokens, factory.embedding_dims))
			for k, word in enumerate(sentence):
				token_i += 1

				if token_i > max_tokens:
					break
				word_embedding = factory.token_index[word]
				if word_embedding is not None:
					sentence_embedding[token_i - 1] = word_embedding


		# Append
		ds_embedding[doc_i] = sentence_embedding

	return ds_embedding



# Trail a sentence level model
def train_han(tokenizer):

	# Pad sentences to 500 and words to 200.
	factory = SentenceModelFactory(
		num_classes=2, 
		token_index=tokenizer.token_index, 
		max_sents=MAX_SENTS, 
		max_tokens=MAX_TOKENS, 
		embedding_type='glove.6B.100d')

	# Hieararchy
	word_encoder_model = AttentionRNN()
	sentence_encoder_model = AttentionRNN()

	# Allows you to compose arbitrary word encoders followed by sentence encoder.
	model = factory.build_model(word_encoder_model, sentence_encoder_model)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()

	return factory, model


def main():

	# Steps to perform
	BUILD_TOKENIZER = False
	BUILD_DATASET   = False
	PREP_DATA       = False
	TRAIN_MODEL     = False
	INFERENCE       = True

	# Filenames
	path_model = os.path.join(data_base, 'models', 'StackedRNN_fitted.h5')

	# Make directory if doesn't exist
	initialize_dir(data_base)

	# Read data
	print("=== Loading Data")
	data = load_data.data(dataset='acllmdb')

	# Build a token. Be default it uses 'en' model from SpaCy
	if BUILD_TOKENIZER:
		print("=== Building Tokenizer")
		tokenizer = SentenceWordTokenizer()
		tokenizer.build_vocab(data.data['x'])


	# Build a Dataset
	if BUILD_DATASET:
		print("=== Building Dataset")
		ds = Dataset(data.data['x'], data.data['y'], tokenizer=tokenizer)
		ds.update_test_indices(test_size=0.1)
		ds.save('../data/dataset/dataset_example')
	else:
		print("=== Loading Saved Dataset")
		ds = load(file_name=os.path.join(data_base, 'dataset', 'dataset_example'))
		tokenizer = ds.tokenizer

	# Prepare encoded data
	if PREP_DATA:

		print("=== Preparing Training Data")

		# Build training data
		X_train = ds.tokenizer.encode_texts(ds.X.tolist())
		# X_train = ds.tokenizer.decode_texts(X_train)

		# Save
		dump(X_train, file_name='../data/dataset/prepped_data_token')
	else:
		print("=== Loading Prepped Dataset")
		X_train = load(file_name='../data/dataset/prepped_data_token')




	# Will automagically handle padding for models that require padding (Ex: Yoon Kim CNN)
	if TRAIN_MODEL:
		print("=== Tokenizing Dataset and Padding")
		sentence_model = False
		model_cnn = False

		factory, model = train_han(tokenizer)


		if sentence_model:
			
			factory = TokenModelFactory(2, tokenizer.token_index, max_tokens=MAX_TOKENS, embedding_type='glove.6B.100d')
			sentence_encoder_model = StackedRNN()

			# Train Model
			print("=== Configuring Model")
			model = factory.build_model(token_encoder_model=sentence_encoder_model)
			model.compile(optimizer='adam', loss='categorical_crossentropy')
			model.summary()
		if model_cnn:

			factory = TokenModelFactory(1, tokenizer.token_index, max_tokens=MAX_TOKENS, embedding_type='glove.6B.100d')
			word_encoder_model = YoonKimCNN()

			# Train Model
			print("=== Configuring Model")
			model = factory.build_model(token_encoder_model=word_encoder_model)
			model.compile(optimizer='adam', loss='categorical_crossentropy')
			model.summary()


		print("=== Training Model")


		# Build training data
		# ds_embedding = doc_to_token(X_train, factory, MAX_TOKENS)
		ds_embedding = doc_to_sentence(X_train, factory, MAX_SENTS, MAX_TOKENS)
		
		print("\tTraining data has rows = %d" % ds_embedding.shape[0])
		print(ds_embedding[0])

		# Convert from array to categorical
		y_train = to_categorical(ds.y)

		# Fit
		model.fit(x=ds_embedding, y=y_train,
			epochs=25,
			batch_size=256)

		# Save
		print("\t\tSaving Model")
		model.save(path_model)
		dump(factory, file_name='../data/embeddings/factory_example')
		# dump(model, file_name='../data/models/StackedRNN')
		print("\t\tDone Saving.")
	else:
		print("=== Loading Embeddings Configuration")
		factory = load(file_name='../data/embeddings/factory_example')
		print("=== Loading Model Configuration")
		# model   = load(file_name='../data/models/StackedRNN')
		model = load_model(path_model)
		model.summary()




	# Make predictions
	if INFERENCE:
		print("=== Making Inference")
		data_infer = [
			'I thought the movie was terrible. I hated the characters.',
			'The movie was great, but the game was better',
			'Nothing in the movie was terrible. It was great.']

		data_infer = ['I hated that movie']


		# Build training data
		X_test = ds.tokenizer.encode_texts(data_infer)

		print('=== X_test')
		print(X_test)

		# Build training data
		embedding_test = doc_to_token(X_test, factory, MAX_TOKENS)

		
		print('=== embedding_test')
		print(embedding_test)

		print("\tTest data has rows = %d" % embedding_test.shape[0])

		print("\t--- Feeding tokenized text to model")
		pred = model.predict(x=embedding_test, verbose=1)
		print(pred)

		df_test = pd.DataFrame(pred)
		df_test['text'] = data_infer

		print(df_test)


if __name__ == '__main__':
	main()

	# test = np.empty((5, 2))
	# print(test)

	# t = [1,0,0,1,1,0]
	# t = np.array(t)
	# print(t)
	# t = t.reshape([t.shape[0], 1])
	# print(t)

	# test_raw = [[[1,2,3,4,5],[1,2,3]],
	# [[3,5,6,6],[9]]]

	# final_list = []
	# for sentence in test_raw:
	# 	final_list.append([y for x in sentence for y in x])

	# print(final_list)

	# test[1,1] = 99
	# print(test)

	# doc_list = np.empty(3)
	# ds_embeddings = np.zeros((5, 2, 10))
	# sentence_1 = np.empty((2, 10))
	# ds_embeddings[1] = sentence_1
	# print(ds_embeddings)
	# doc_list[0] = sentence_1
	# print(sentence_1)



