# Senti
Exploring Sentiment Modeling

We explore how words used in the English speaking world takes on different meanings in the **gaming**.
Typically words with **polysemy** [[Hamilton](https://arxiv.org/pdf/1605.09096.pdf)] tends to have higher rates of semantic change.

To compare word embeddings like **word2vec** we can use Positive Pointwise Mutual Information (PPMI) to measure the strength of association between two words. PMI It is defined as the log ratio between the joint probability of two words and the product of their marginal probabilities. Positive PMI (PPMI) replaces negative values with 0.

We compare the embeddings from our gaming corpus and compare it with Goog'e N-Gram datasets comprising of about 6% of all books ever published. We can also compare it with COHA which is designed to be genre balanced.

To visualize semantic change for a word w<sub>i</sub> in two dimensions we employed the following procedure, which relies on the t-SNE embedding method (Van der Maaten and Hinton, 2008) as a subroutine:
1. Find the union of the word wi’s k nearest neighbors over all necessary time-points.
2. Compute the t-SNE embedding of these words on the most recent (i.e., the modern) time-point.
3. For each of the previous time-points, hold all embeddings fixed, except for the target word’s (i.e., the embedding for wi), and optimize a new t-SNE embedding only for the target word. We found that initializing the embedding for the target word to be the centroid of its k-nearest neighbors in a timepoint was highly effective.

Thus, in this procedure the background words are
always shown in their “modern” positions, which
makes sense given that these are the current meanings
of these words. This approximation is necessary,
since in reality all words are moving.

`
pip install --upgrade tensorflow
pip install --upgrade keras
pip install --upgrade gensim
`