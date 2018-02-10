import sys
import os
import time

import numpy as np
from keras.utils import to_categorical
from keras.models import load_model

from keras_text.processing import WordTokenizer, SentenceWordTokenizer
from keras_text.data import Dataset
from keras_text.models.token_model import TokenModelFactory
from keras_text.models import SentenceModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN, AveragingEncoder
from keras_text.utils import dump, load
from keras_text.models.layers import *


import load_data

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


def doc_to_sentence(doc_token_list, max_sents, max_tokens):

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


def save_folder(folder_path):
	'''
	Create folder and subfolders to save results
	'''
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
		os.makedirs(folder_path + '/models')
		os.makedirs(folder_path + '/embeddings')
		os.makedirs(folder_path + '/dataset')
		os.makedirs(folder_path + '/diagnostics')
		os.makedirs(folder_path + '/results')


def build_tokenizer(data):
	print("=== Building Tokenizer")
	tokenizer = SentenceWordTokenizer()
	tokenizer.build_vocab(data.data['x'])
	return tokenizer

def build_dataset(data, tokenizer):
	print("=== Building Dataset")
	ds = Dataset(data.data['x'], data.data['y'], tokenizer=tokenizer)
	ds.update_test_indices(test_size=0.1)
	return ds

# Trail a sentence level model
def train_han(tokenizer):

	# Pad sentences to 500 and words to 200.
	factory = SentenceModelFactory(
		num_classes=2, 
		token_index=tokenizer.token_index, 
		max_sents=500, 
		max_tokens=200, 
		embedding_type='glove.6B.100d')

	# Hieararchy
	word_encoder_model = AttentionRNN()
	sentence_encoder_model = AttentionRNN()

	# Allows you to compose arbitrary word encoders followed by sentence encoder.
	model = factory.build_model(word_encoder_model, sentence_encoder_model)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()

	return model




if __name__ == '__main__':

	# Save all results into a directory
	save_directory = '../data/train_han'
	save_folder(save_directory)
	model_name = 'HAN_1'


	# Steps to perform
	BUILD_TOKENIZER = False
	BUILD_DATASET   = False
	TRAIN_MODEL     = False
	INFERENCE       = True

	NUM_CLASSES = 2
	MAX_SENTS = 100
	MAX_TOKENS = 100

	# Read data
	print("=== Loading Data")
	data = load_data.data(dataset='acllmdb')

	# Build a token. Be default it uses 'en' model from SpaCy
	if BUILD_TOKENIZER:
		tokenizer = build_tokenizer(data)


	# Build a Dataset
	if BUILD_DATASET:
		ds = build_dataset(data, tokenizer)
		ds.save(os.path.join(save_directory, 'dataset', 'dataset_example'))
	else:
		print("=== Loading Saved Dataset")
		ds = Dataset(data.data['x'], data.data['y']).load(os.path.join(save_directory, 'dataset', 'dataset_example'))
		print(ds)
		print(type(ds))
		print(vars(ds).keys())
		tokenizer = ds.tokenizer


	# Will automagically handle padding for models that require padding (Ex: Yoon Kim CNN)
	if TRAIN_MODEL:

		# Can also use `max_sents=None` to allow variable sized max_sents per mini-batch.
		print("=== Tokenizing Dataset and Padding")
		factory = SentenceModelFactory(NUM_CLASSES, tokenizer.token_index, max_sents=MAX_SENTS, max_tokens=MAX_TOKENS, embedding_type='glove.6B.100d')
		word_encoder_model = AttentionRNN()
		sentence_encoder_model = AttentionRNN()

		# Allows you to compose arbitrary word encoders followed by sentence encoder.
		print("=== Defining Model")
		model = factory.build_model(word_encoder_model, sentence_encoder_model)
		model.compile(optimizer='adam', loss='categorical_crossentropy')
		model.summary()


		# Build training data
		X_train = ds.tokenizer.encode_texts(ds.X.tolist())
		X_train = doc_to_sentence(X_train, MAX_SENTS, MAX_TOKENS)
		
		print("\tTraining data has rows = %d" % X_train.shape[0])
		print(X_train[0])
		print(X_train.shape)

		# Convert from array to categorical
		y_train = to_categorical(ds.y)

		# Fit
		model.fit(x=X_train, y=y_train,
			epochs=1,
			batch_size=512)

		# Save
		dump(factory, file_name=os.path.join(save_directory, 'embeddings', 'factory_sent'))
		model.save(os.path.join(save_directory, 'models', model_name))

	else:
		print("=== Loading Embeddings")
		factory = load(file_name=os.path.join(save_directory, 'embeddings', 'factory_sent'))
		print("=== Loading Model")
		model = load_model(os.path.join(save_directory, 'models', model_name),
			custom_objects={'AttentionLayer': AttentionLayer})


	# Make predictions
	if INFERENCE:
		print("=== Making Inference")
		data_infer = [
			'I thought the movie was terrible. The way the characters talked annoyed me.', 
			'Very fun and exciting. It was one of the best experiences I had.'
		]

		print("\t--- Tokenizing raw inference dataset")		
		print(vars(tokenizer).keys())
		start = time.time()
		x_test = tokenizer.encode_texts(data_infer)
		end = time.time()
		print("Tokenizer [%s]" % (end - start))
		start = time.time()
		x_test = doc_to_sentence(x_test, MAX_SENTS, MAX_TOKENS)
		end = time.time()
		print("Document Embedding [%s]" % (end - start))

		print("\t--- Feeding tokenized text to model")
		start = time.time()
		pred = model.predict(x=x_test, verbose=1)
		end = time.time()
		print("Inference [%s]" % (end - start))
		print(pred)
		print(data_infer)


