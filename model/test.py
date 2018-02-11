import os
import keras_text
from keras_text.embeddings import *
from keras_text.embeddings import _EMBEDDINGS_CACHE


def test():

	path = os.path.dirname(keras_text.__file__)
	print(path)


	embed = get_embeddings_index('glove.6B.50d')

	print(type(embed))

	i = 0
	for key, value in embed.items():

		print(key)
		print(type(value))
		print(len(value))
		print(value)
		i += 1

		if i > 10:
			break

if __name__ == '__main__':
	_EMBEDDINGS_CACHE['haha'] = 'yoooooooooooo'

	test()