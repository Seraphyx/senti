import sys
import os

from keras_text.processing import WordTokenizer
from keras_text.data import Dataset
from keras_text.models.token_model import TokenModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN
from keras_text.utils import dump, load


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



def save_folder(folder_path):
	'''
	Create folder and subfolders to save results
	'''
	if not os.path.exists(folder_path):
    	os.makedirs(folder_path)
    	os.makedirs(folder_path + '/models')
    	os.makedirs(folder_path + '/embeddings')
    	os.makedirs(folder_path + '/data')
    	os.makedirs(folder_path + '/diagnostics')
    	os.makedirs(folder_path + '/results')


def build_tokenizer(data):
	print("=== Building Tokenizer")
	tokenizer = WordTokenizer()
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


	# Steps to perform
	BUILD_TOKENIZER = False
	BUILD_DATASET   = False
	TRAIN_MODEL     = False
	INFERENCE       = True

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
		print("=== Tokenizing Dataset and Padding")
		factory = TokenModelFactory(1, tokenizer.token_index, max_tokens=100, embedding_type='glove.6B.100d')
		word_encoder_model = YoonKimCNN()

		# Train Model
		print("=== Training Model")
		model = factory.build_model(token_encoder_model=word_encoder_model)
		model.compile(optimizer='adam', loss='categorical_crossentropy')
		model.summary()

		# Save
		dump(factory, file_name='../data/embeddings/factory_example')
		dump(model, file_name='../data/models/YoonKimCNN_example')
	else:
		print("=== Loading Embeddings")
		factory = load(file_name='../data/embeddings/factory_example')
		print("=== Loading Model")
		model   = load(file_name='../data/models/YoonKimCNN_example')


	# Make predictions
	if INFERENCE:
		print("=== Making Inference")
		data_infer = ['I thought the movie was terrible', 'Very fun and exciting']

		print("\t--- Tokenizing raw inference dataset")		
		print(vars(tokenizer).keys())
		ds_infer = tokenizer.encode_texts(data_infer)
		ds_decode = tokenizer.decode_texts(ds_infer)
		print(ds_infer)
		print(ds_decode)
		print(type(factory))

		# Convert words to vector embeddings
		ds_embedding = []
		for i, sentence in enumerate(ds_decode):
			print('\t\tWorking on sentence %d' % (i + 1))
			sentence_embedding = []
			for j, word in enumerate(sentence):
				word_embedding = factory.embeddings_index[word.encode()]
				if word_embedding is not None:
					sentence_embedding.append(word_embedding)

			ds_embedding.append(sentence_embedding)

		print("\t--- Feeding tokenized text to model")
		pred = model.fit(x=ds_embedding)
		print(pred)
