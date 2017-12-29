from keras_text.processing import WordTokenizer
from keras_text.data import Dataset

import keras_text.models as test

print(dir(test))


from keras_text.models.token_model import TokenModelFactory
from keras_text.models import YoonKimCNN, AttentionRNN, StackedRNN


import load_data


'''
There are a few steps within the model pipeline that can be modified to adjust the 
'''

# tokenizer = WordTokenizer()
# tokenizer.build_vocab(texts)


if __name__ == '__main__':

	# Read data
	print("=== Loading Data")
	data = load_data.data(dataset='acllmdb')

	# # Build a token. Be default it uses 'en' model from SpaCy
	# print("=== Building Tokenizer")
	# tokenizer = WordTokenizer()
	# tokenizer.build_vocab(data.data['x'])

	# # Build a Dataset
	# print("=== Building Dataset")
	# ds = Dataset(data.data['x'], data.data['y'], tokenizer=tokenizer)
	# ds.update_test_indices(test_size=0.1)
	# ds.save('../data/dataset')

	ds = Dataset(data.data['x'], data.data['y']).load('../data/dataset')
	print(ds)
	print(type(ds))
	print(vars(ds).keys())
	tokenizer = ds.tokenizer

	# Will automagically handle padding for models that require padding (Ex: Yoon Kim CNN)
	print("=== Tokenizing Dataset and Padding")
	factory = TokenModelFactory(1, tokenizer.token_index, max_tokens=100, embedding_type='glove.6B.100d')
	word_encoder_model = YoonKimCNN()

	# 
	print("=== Model")
	model = factory.build_model(token_encoder_model=word_encoder_model)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	model.summary()
