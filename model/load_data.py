import csv


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


test = data(file='../data/imdb/labeledTrainData.tsv')
print(test)