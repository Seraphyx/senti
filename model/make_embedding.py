import os
import gensim
from gensim.models import word2vec, Word2Vec, phrases, Phrases, KeyedVectors
import logging

import numpy as np
import pandas as pd

import urllib.request
import os
import zipfile
import platform

from pprint import pprint
from keras_text.embeddings import *
from keras_text.embeddings import _EMBEDDINGS_CACHE


def GetDataDir():
    plat = platform.platform()
    if "Windows" in plat:
        return "G:/Projects/senti/data/" 
    else:
        return "/Users/jasonpark/documents/project/senti/data/"

root_path = GetDataDir()

if __name__ == '__main__':

	print('_EMBEDDINGS_CACHE')
	print(_EMBEDDINGS_CACHE)

	# Load Gensim w2v
	model = gensim.models.Word2Vec.load(os.path.join(root_path, 'example', 'mymodel'))

	# Convert from gensim wv to dict
	embed = {k : model.wv[k] for k in model.wv.index2word}

	# Assign gensim to model
	_EMBEDDINGS_CACHE['mymodel'] = embed



	i = 0
	for key, value in embed.items():

		print(key)
		print(type(value))
		print(len(value))
		print(value)
		i += 1

		if i > 10:
			break