
#!/usr/local/bin/python3
import os

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
# from werkzeug.contrib.cache import SimpleCache


from keras_text.data import Dataset
from keras_text.utils import load
from keras.models import load_model

'''
Attention with Context:
https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

Attention Temporal:
https://gist.github.com/cbaziotis/6428df359af27d58078ca5ed9792bd6d

Attentional Dense Layers:
https://github.com/philipperemy/keras-attention-mechanism

Using pre-trained word embeddings in a Keras model:
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

Blog on Word2Vec and Keras:
http://ben.bolte.cc/blog/2016/gensim.html


1) We want to tokenize each string to get a list of words, usually by making everything lowercase 
and splitting along the spaces. In contrast, lemmatization involves getting the root of each word, 
which can be helpful but is more computationally expensive (enough so that you would want to 
preprocess your text rather than do it on-the-fly).
2) Vectorize words
3) Build Model
4) Grid Search
5) Save


'''

# Location of model and tokenizer
data_base = './data'


def doc_to_token(doc_token_list, factory, max_tokens):

	# Count how many sentences
	n_doc = len(doc_token_list)

	# Initialize
	ds_embedding = np.zeros((n_doc, max_tokens))

	# Convert words to vector embeddings
	for doc_i, doc in enumerate(doc_token_list):

		token_i = 0
		for j, sentence in enumerate(doc):
			if token_i + 1 > max_tokens:
				break;
			for token in sentence:
				if token_i + 1 > max_tokens:
					break;
				ds_embedding[doc_i, token_i] = token
				token_i += 1

	return ds_embedding




# print a nice greeting.
def say_hello(username = "World"):
	return '<p>Hello %s!</p>\n' % username

# some bits of text for the page.
header_text = '''
	<html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
instructions = '''
	<p><em>Hint</em>: This is a RESTful web service! Append a username
	to the URL (for example: <code>/Thelonious</code>) to say hello to
	someone specific.</p>\n'''
home_link = '<p><a href="/">Back</a></p>\n'
footer_text = '</body>\n</html>'

# EB looks for an 'app' callable by default.
app = Flask(__name__)

# add a rule for the index page.
app.add_url_rule('/', 'index', (lambda: header_text +
	say_hello() + instructions + footer_text))

# add a rule when the page is accessed with a name appended to the site
# URL.
app.add_url_rule('/<username>', 'hello', (lambda username:
	header_text + say_hello(username) + home_link + footer_text))



# Model request
@app.route('/model', methods=['GET','POST'])
def model():

	error = None

	if request.method == 'POST':

		# What the client sends as header type TEXT
		print('=== GETTING data from req')
		data = request.data
		print('=== GOT data from req')
		print(data)

		# Get Prediction
		print('=== GETTING prediction')
		pred = predict(data)
		print(pred)
		print('=== GOT prediction')

		return jsonify(pred)


	# Return 
	return jsonify({'message': 'Use POST with the header described here and text in the body.',
		'header': {
			'Content-Type': 'text/html',
			'charset': 'UTF-8'
		}})


def initialize():
	'''
	Cache any model dependencies here
	'''

	dir_path = os.path.dirname(os.path.realpath(__file__))
	cwd = os.getcwd()

	print('=== dir_path')
	print(dir_path)
	print('=== cwd')
	print(cwd)


	print('=== Caching Tokenizer')
	global ds, factory, model
	ds = load(file_name=os.path.join(data_base, 'server', 'dataset_example'))
	# cache.set('ds', ds)

	print('=== Caching Factory')
	factory = load(file_name=os.path.join(data_base, 'server', 'factory_example'))
	# cache.set('factory', factory)

	print('=== Caching Model')
	model = load_model(os.path.join(data_base, 'server', 'StackedRNN_fitted.h5'))
	# cache.set('model', model)
	# model.summary()




def predict(text):

	# Parameters
	MAX_TOKENS = 100
	global ds, factory, model

	# Decode Text to UTF-8
	print('=== DECODING text')
	text = text.decode('utf-8')

	# Get token id list
	print('=== TOKENIZING text')
	X_test = ds.tokenizer.encode_texts([text])

	# Build training data
	print('=== EMBEDDING text')
	embedding_test = doc_to_token(X_test, factory, MAX_TOKENS)
	print(embedding_test)

	# Get Sentiment
	print('=== PREDICTING text')
	pred = model.predict(x=embedding_test, verbose=1)

	return {
			'text': text,
			'pred': pred.tolist()
		}


with app.app_context():
    # within this block, current_app points to app.
    initialize()



# run the app.
if __name__ == "__main__":

	# Setting debug to True enables debug output. This line should be
	# removed before deploying a production app.
	app.debug = True

	# Run
	app.run(host="0.0.0.0", port=80, threaded=True)
