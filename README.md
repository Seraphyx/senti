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
`docker run -it --rm -p 8888:8888 -p 5000:5000 --user root -e GRANT_SUDO=yes -v ${pwd}:/home/jovyan/work senti`


## Misc
Consider adding **tqdm** for progressbars in Jupyter.