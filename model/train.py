from __future__ import unicode_literals

import sys
from pprint import pprint

import numpy as np


from keras_text.processing import WordTokenizer, SentenceWordTokenizer
from keras_text.data import Dataset
from keras_text.models.token_model import TokenModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN
from keras_text.utils import dump, load
from keras.utils import to_categorical

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

	# Steps to perform
	BUILD_TOKENIZER = False
	BUILD_DATASET   = False
	CONFIG_MODEL	= True
	TRAIN_MODEL     = True
	INFERENCE       = True

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
		ds = load(file_name='../data/dataset/dataset_example')
		print(ds)
		print(type(ds))
		print(vars(ds).keys())
		print(ds.X.tolist()[0:3])
		print(ds.y.tolist()[0:3])
		tokenizer = ds.tokenizer

	print(":::::::::::::::: tokenizer.get_stats")
	print(ds.tokenizer.get_stats(0))
	print(":::::::::::::::: tokenizer.get_stats")
	print(ds.tokenizer.get_stats(1))

	# Build training data
	X_train = ds.tokenizer.encode_texts(ds.X.tolist()[1:100])
	X_train = ds.tokenizer.decode_texts(ds.X)

	print(X_train.shape)
	print(X_train[1:100])

	sys.exit("Stop here")




	# Will automagically handle padding for models that require padding (Ex: Yoon Kim CNN)
	if CONFIG_MODEL:
		print("=== Tokenizing Dataset and Padding")
		factory = TokenModelFactory(1, tokenizer.token_index, max_tokens=100, embedding_type='glove.6B.100d')
		word_encoder_model = YoonKimCNN()

		# Train Model
		print("=== Configuring Model")
		model = factory.build_model(token_encoder_model=word_encoder_model)
		model.compile(optimizer='adam', loss='categorical_crossentropy')
		model.summary()

		# Save
		dump(factory, file_name='../data/embeddings/factory_example')
		dump(model, file_name='../data/models/YoonKimCNN_example')
	else:
		print("=== Loading Embeddings Configuration")
		factory = load(file_name='../data/embeddings/factory_example')
		print("=== Loading Model Configuration")
		model   = load(file_name='../data/models/YoonKimCNN_example')


	if TRAIN_MODEL:
		print("=== Training Model")

		# Build training data
		X_train = tokenizer.encode_texts(ds.X.tolist())
		X_train = tokenizer.decode_texts(ds.X)

		print(X_train[0:3])

		model.fit(X_train, ds.y,
			epochs=20,
			batch_size=128)


	# Make predictions
	if INFERENCE:
		print("=== Making Inference")
		data_infer = ['I thought the movie was terrible']

		print("\t--- Tokenizing raw inference dataset")		
		print(vars(tokenizer).keys())
		ds_infer = tokenizer.encode_texts(data_infer)
		ds_decode = tokenizer.decode_texts(ds_infer)
		print(ds_infer)
		print(ds_decode)
		print(':::::::::::::: FACTORY')
		print(type(factory))
		print(vars(factory).keys())
		print(factory.embedding_dims)
		print(factory.max_tokens)

		# Convert words to vector embeddings
		ds_embedding = np.array([])
		for i, sentence in enumerate(ds_decode):
			print('\t\tWorking on sentence %d' % (i + 1))
			sentence_embedding = np.array([]).reshape(0,100)
			for j, word in enumerate(sentence):
				word_embedding = factory.embeddings_index[word.encode()]
				print("word_embedding.shape")
				print(word_embedding.shape)
				if word_embedding is not None:
					sentence_embedding = np.vstack([sentence_embedding, [word_embedding]])
					print(sentence_embedding.shape)

			ds_embedding = np.append([ds_embedding], [sentence_embedding])

		print("::::::::::::::::: ds_embedding")
		print(ds_embedding.shape)
		print("::::::::::::::::: sentence_embedding")
		print(sentence_embedding.shape)
		print("::::::::::::::::: model")
		print(vars(model).keys())

		# print(ds_embedding[0][0])
		# print(ds_embedding[0][0].shape)
		# print(ds_embedding[0][0].T.shape)


		# Describe the model
		print("::::::::::::::::: model.summary")
		model.summary()
		print("::::::::::::::::: model.get_config")
		pprint(model.get_config())
		# model.save_weights('../data/models/YoonKimCNN_example_weights')





		print("\t--- Feeding tokenized text to model")
		y_binary = to_categorical(np.array([1, 0, 1, 0, 1, 0]))
		print(y_binary.shape)
		pred = model.predict(x=sentence_embedding, verbose=1)
		print(pred)
