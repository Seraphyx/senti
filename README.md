# Senti
Exploring Sentiment Modeling

We explore how words used in the English speaking world takes on different meanings in the **gaming**.
Typically words with **polysemy** [[Hamilton](https://arxiv.org/pdf/1605.09096.pdf)] tends to have higher rates of semantic change.

To compare word embeddings like **word2vec** we can use Positive Pointwise Mutual Information (PPMI) to measure the strength of association between two words. PMI It is defined as the log ratio between the joint probability of two words and the product of their marginal probabilities. Positive PMI (PPMI) replaces negative values with 0.

We compare the embeddings from our gaming corpus and compare it with Goog'e N-Gram datasets comprising of about 6% of all books ever published. We can also compare it with COHA which is designed to be genre balanced.

## Visualizing Algorithm

To visualize semantic change for a word w<sub>i</sub> in two dimensions we employed the following procedure, which relies on the t-SNE embedding method (Van der Maaten and Hinton, 2008) as a subroutine:
1. Find the union of the word wi’s k nearest neighbors over all necessary time-points.
2. Compute the t-SNE embedding of these words on the most recent (i.e., the modern) time-point.
3. For each of the previous time-points, hold all embeddings fixed, except for the target word’s (i.e., the embedding for wi), and optimize a new t-SNE embedding only for the target word. We found that initializing the embedding for the target word to be the centroid of its k-nearest neighbors in a timepoint was highly effective.

Thus, in this procedure the background words are
always shown in their “modern” positions, which
makes sense given that these are the current meanings
of these words. This approximation is necessary,
since in reality all words are moving.

## Embedding Strategies
SGNS and PPMI.
PPMI method benefited substantially from larger contexts, so we did not remove any lowfrequency words per year from the context for that method. The other embedding approaches did not appear to benefit from the inclusion of these lowfrequency terms, so they were dropped for computational efficiency.


## Windows Linux
`C:\Users\<windows-username>\AppData\Local\lxss\home\<linux-username>`


`
pip install --upgrade tensorflow keras gensim thinc
conda config --add channels conda-forge
conda install spacy
`



## Docker

### Building Docker Image
Build with name
`docker build -t senti .`

### View images
`docker images`

### Run the container
We mount the notebook folder to our container. We expose 8888 for Jupyter Notebooks, and 5000 for SpaCy visualizations.
`docker run -it --rm -p 8888:8888 -p 5000:5000 --user root -e GRANT_SUDO=yes -v ${pwd}:/home/jovyan/work  senti`
For Liunux/OSX use:
`docker run -it --rm -p 8888:8888 -p 5000:5000 --user root -e GRANT_SUDO=yes -v `pwd`:/home/jovyan/work senti`


## Misc
Consider adding **tqdm** for progressbars in Jupyter.



# Modeling
There are various steps where we can intervene and modify to generate a better performing model. Of course, we can run through a battery of models and grid search through each model's respective hyper-parameters, but we can also change the preprocessing steps:
1. **Data Clean**
⋅⋅* **TFIDF**: Get only words with discrinatory power, or leave them in if the model is expressive enough to take advantage of the context.
..* **Named Entity Recognition** (**NER**): We can generalize the entities into categories. If, for example, we leave **Star Wars: Battlefront II** as a token we can avoid having the individual words (*Star*, *Wars*, etc.) mean something that we don't want. Also, if we tag this as a **Game (Noun)** and hash it as just that, then we can avoid biasing this entity since the mere presence of **SWBF2** will have negative sentiment otherwise even though people may talk about it favorably.
..* **Part of Speech** (**POS**): We can take the parts of speech such as Verb, Adjective, Noun, etc to give Homonyms separate tokens and perhaps even capture the Polysemic usage into its correct usage. For example, we can perhaps use it as **Duck|Noun** vs **Duck|Verb** as a token. See SpaCy or SyntaxNet. We can also use as an embedding either at the end of the token vector.
2. **Vectorize**
..* **Word2Vec**: We can use Google's Word2Vec bank or Glove's representation. We can also train our own using Skip-Gram, since our training dataset may be genre specific.
3. **Model**





