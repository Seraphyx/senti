import gensim
from gensim.models import word2vec, Word2Vec
import logging

import numpy as np

import urllib.request
import os
import zipfile
import platform



def GetDataDir():
    plat = platform.platform()
    if "Windows" in plat:
        return "G:/Projects/senti/data/example/" 
    else:
        return "/Users/jasonpark/documents/project/senti/data/"

root_path = GetDataDir()
vector_dim = 300

# Directory full of sentences
# i.e.: https://archive.ics.uci.edu/ml/machine-learning-databases/00311/
text_directory = os.path.join(root_path, 'sentences/SentenceCorpus/unlabeled_articles/arxiv_unlabeled')

def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print("Size is %s. Expected %s" % (statinfo.st_size, expected_bytes))
        print(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


class Sentences(object):
    '''
    Read all text files within a given directory.
    Each new line is considered a new sentence.
    '''
    def __init__(self, text_directory):
        self.text_directory = text_directory

    def __iter__(self):
        for text_file in os.listdir(self.text_directory):
            print("\tText File:", str(text_file))
            for line in open(os.path.join(self.text_directory, text_file)):
                yield line.split()


# convert the input data into a list of integer indexes aligning with the wv indexes
# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = f.read(f.namelist()[0]).split()
    return data

def convert_data_to_index(string_data, wv):
    index_data = []
    for word in string_data:
        if word in wv:
            index_data.append(wv.vocab[word].index)
    return index_data


def gensim_demo():
    print("Running Gensim Demo...")

    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url, 31344016)
    if not os.path.exists((root_path + filename).strip('.zip')):
        zipfile.ZipFile(root_path+filename).extractall()
    sentences = word2vec.Text8Corpus((root_path + filename).strip('.zip'))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, iter=10, min_count=10, size=300, workers=4)

    # get the word vector of "the"
    print("=== get the word vector of 'the'")
    print(model.wv['the'])

    # get the most common words
    print("=== get the most common words")
    print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])

    # get the least common words
    print("=== get the least common words")
    vocab_size = len(model.wv.vocab)
    print(model.wv.index2word[vocab_size - 1], model.wv.index2word[vocab_size - 2], model.wv.index2word[vocab_size - 3])

    # find the index of the 2nd most common word ("of")
    print("=== find the index of the 2nd most common word ('of')")
    print('Index of "of" is: {}'.format(model.wv.vocab['of'].index))

    # some similarity fun
    print("=== Similarity between 'woman' and 'man' is: %d" % model.wv.similarity('woman', 'man'))
    print("=== Similarity between 'man' and 'elephant' is: %d" % model.wv.similarity('man', 'elephant'))

    # what doesn't fit?
    print("=== Which among 'green', 'blue', 'red', and 'zebra' do not fit?")
    print(model.wv.doesnt_match("green blue red zebra".split()))

    print("=== Extract index from data")
    str_data = read_data(root_path + filename)
    str_decoded = [s.decode('utf-8') for s in str_data[:30]]
    index_data = convert_data_to_index(str_decoded, model.wv)
    index_custom = convert_data_to_index(['I','love','game','early'], model.wv)
    print("\t Original Text")
    print(str_data[:30])
    print("\t Index")
    print(index_data)
    print("\t Index (Custom)")
    print(index_custom)

    # save and reload the model
    print("=== Saving Model")
    model.save(root_path + "mymodel")


def load_model(model_path):

    print("=== Loading Model")
    model = Word2Vec.load(model_path)

    print(model.wv['computer'])

if __name__ == "__main__":

    run_opt = -4

    sent = Sentences(text_directory)
    for file in sent:
        print(file)

    if run_opt == 1:
        gensim_demo()
    elif run_opt == 2:
        model = gensim.models.Word2Vec.load(root_path + "mymodel")
        embedding_matrix = create_embedding_matrix(model)
        tf_model(embedding_matrix, model.wv)
    elif run_opt == 3:
        model = gensim.models.Word2Vec.load(root_path + "mymodel")
        embedding_matrix = create_embedding_matrix(model)
        keras_model(embedding_matrix, model.wv)
    elif run_opt == 4:
        load_model(root_path + 'mymodel')


