import csv
import sklearn
from sklearn.datasets import load_files


data_path = "../data/raw/movie_reviews"


class data(object):

	def __init__(self, file):
		self.file = file
		self.data = self.read_tsv()

	def read_tsv(self):
		with open(self.file,'rb') as tsvin:
			tsvin = csv.reader(tsvin, delimiter='\t')
		for i, row in iter(tsvin):
		    print(row)
		    if i > 10: 
		    	break

		return tsvin

movie_train = load_files(data_path, shuffle=True)
print(len(movie_train.data))
print(movie_train.keys())
print(movie_train.DESCR)
print(movie_train.filenames)
print(movie_train.data[0:3])

# target names ("classes") are automatically generated from subfolder names
print(movie_train.target_names)

# test = data(file='../data/imdb/labeledTrainData.tsv')
# print(test)