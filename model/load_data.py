import csv
import sklearn
from sklearn.datasets import load_files


data_path = "../data/raw/movie_reviews"


class data(object):

	def __init__(self, file=None, dataset=None):
		self.file = file
		self.dataset = dataset
		self.read_dataset()

	def read_tsv(self):
		with open(self.file,'rb') as tsvin:
			tsvin = csv.reader(tsvin, delimiter='\t')
		for i, row in iter(tsvin):
		    print(row)
		    if i > 10: 
		    	break

		self.data = tsvin

	def read_acllmdb(self):
		# Movie review Sentiment: http://www.nltk.org/nltk_data/
		movie_train = load_files(data_path, shuffle=True)
		self.data = {
			'x': [x.decode("utf-8") for x in movie_train.data],
			'y': movie_train.target
		}

	def read_dataset(self):
		# Toy datasets are set here
		if self.dataset == 'acllmdb':
			self.read_acllmdb()
		elif self.dataset == 'tsv':
			self.read_tsv()
		else:
			print("No dataset provided.")


if __name__ == '__main__':

	movie_train = data(dataset='acllmdb')
	print(type(movie_train.data))
	print(len(movie_train.data['x']))
	print(len(movie_train.data['y']))
	# print(movie_train.keys())
	# print(movie_train.DESCR)
	# print(movie_train.filenames)
	# print(movie_train.data[0:3])

	# ds = Dataset(data.data['x'], data.data['y']).load('../data/dataset/dataset_example')
	# import numpy as np
	# doc_zeros   = np.zeros(100)
	# print(doc_zeros.shape)
	# test = [1,2,3,4]
	# print(test)
	# test = np.array(test)
	# print(test)
	# print(test.shape)
	# doc_zeros[:test.size] = test
	# print(doc_zeros)
	# print(doc_zeros.shape)

	# print(test.size)
	# print(min(100, test.size))


