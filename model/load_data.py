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
