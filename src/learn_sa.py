from functools import partial
from itertools import islice
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.cluster import (MiniBatchKMeans,
                             AgglomerativeClustering)
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfTransformer,
                                             ENGLISH_STOP_WORDS)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline


here = os.path.abspath(os.path.dirname(__file__))
mergers = pd.read_pickle(os.path.join(here, 'mergers_clean.pickle'))\
    .drop('_desc', axis=1)
mergers = mergers[mergers['desc'].str.len() > 0]
text = mergers['desc'].values.tolist()

try:
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('wordnet')

STOPWORDS = ENGLISH_STOP_WORDS.union(
    frozenset(('business', 'press', 'source', 'previously', 'deal', 'company',
               'says', 'ceo', 'release', 'companies', 'merger',
               'said', 'say', 'acquisition', 'transaction', 'time', 'year',
               'president', 'assets', 'today', 'product', 'firm')))
# TOKEN = re.compile(r'\b\w{2,}\b')

# Due to ordering of sklearn's text processing routine, we can't have
#     multi-word stopwords with a lemma-tokenizer.  One solution is to
#     build a custom str -> str `processor` *first*

EXT = re.compile(r'\b(?P<ext>inc|lp|llc|corp|technol|ltd|se|plc|ag|on|vz|na|inh|grp|kgaa|sp|co|incorporated|corporation|company limited|limited liability corp(?:oration)|limited|)?\b')


def read_comp(url):
    for line in open(url):
        if '(' in line or ')' in line:
            continue
        yield re.sub('\s+', ' ', EXT.sub('', line.replace(',', '').replace('.', '')
                      .lower()).strip())

# Listed equities: NYSE, AMEX, NASDAQ, TSE, LSE, DEUTSCHE BORSE
COMP = frozenset(read_comp(os.path.join(here, 'companies.txt')))


# def proc(doc) -> str:
#     """Custom preprocessor."""
#     # Prefer lower over re.IGNORECASE because we want a lc str result
#     return COMP.sub('', doc.lower().replace("'s", ''))


def window(seq, n=3):
    """Length-n sliding window over `seq`.  Source: itertools docs."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class LemmaTokenizer(object):
    """Tokenize, remove base stop words, & lemmatize all-in-one."""
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc, stop_words):
        return tuple(self.wnl.lemmatize(i.lower()) for i in
                     re.findall(r'\b\w{3,}\b', doc)
                     if i.lower() not in stop_words)


def analyzer(doc, stop_words=None, stop_phr=None, ngram_range=(1, 4)):
    """Custom text vectorization parameter."""
    if not stop_words:
        stop_words = {}
    if not stop_phr:
        stop_phr = {}
    start, stop = ngram_range
    lt = LemmaTokenizer()
    words = lt(doc, stop_words=stop_words)
    # bind method outside loop
    join = ' '.join
    for n in range(start + 1, stop + 1):
        for ngram in window(words, n=n):
            res = join(ngram)
            if res not in stop_phr:
                yield res
    for w in words:
        yield w


# To enable passing a single callable below
analyzer_ = partial(analyzer, stop_words=STOPWORDS, stop_phr=COMP)

TF_KWARGS = dict(
    max_df=0.3,
    min_df=35,
    analyzer=analyzer_)


# tf-idf -> NMF  /  tf -> LDA
# tf/tfidf: rows=docs, features=vocab
# ---------------------------------------------------------------------

tf_vect = CountVectorizer(**TF_KWARGS)
tf = tf_vect.fit_transform(text)
tfidf = TfidfTransformer().fit_transform(tf)

# n_components -> n_topics (v <0.21)
nmf = NMF(n_components=8, random_state=444, beta_loss='kullback-leibler',
          solver='mu', max_iter=1000, alpha=0.1, l1_ratio=0.5)
nmf.fit(tfidf)

lda = LatentDirichletAllocation(n_components=8, max_iter=10,
                                learning_method='online', learning_offset=50.,
                                random_state=444, n_jobs=-1)
lda.fit(tf)

# TODO: which documents are most associated with which topics?
def get_top_words(model, feature_names, n_top_words):
    # Each component_ is a topic vector with the same length as
    #     the vocabulary.
    feature_names = np.asarray(feature_names)
    for topic_idx, topic in enumerate(model.components_, 1):
        yield topic_idx, feature_names[topic.argsort()[:-n_top_words - 1:-1]]


# LSA
# ---------------------------------------------------------------------

# http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#sphx-glr-auto-examples-text-document-clustering-py
svd = TruncatedSVD(n_components=8, random_state=444)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(tfidf)
exp_var = svd.explained_variance_ratio_.sum()


# Note: centroid coordinates (km.cluster_centers_) are n-tuples where
#       n is X.shape[1] (features)
def loop_nclusters(rng, model, X):
    print(model.__class__.__name__, ':', sep='')
    inertias = []
    mean_silhouettes = []
    i = 1  # i != n
    for n in rng:
        print('Loop %s of %s...' % (i, len(rng)))
        model.set_params(n_clusters=n)
        model.fit(X)
        if hasattr(model, 'inertia_'):
            inertias.append(model.inertia_)
        # sample_size significantly influences runtime (more than fitting)
        mean_silhouettes.append(metrics.silhouette_score(X, model.labels_,
                                                         random_state=444,
                                                         sample_size=5000))
        i += 1
    print()
    return inertias, mean_silhouettes


rng = tuple((*range(4, 11), *range(10, 21, 2)))
models = (
    MiniBatchKMeans(init='k-means++', n_init=5, batch_size=100,
                    init_size=1000, tol=0.001, random_state=444),
    AgglomerativeClustering()
    )
scores = dict.fromkeys(model.__class__.__name__ for model in models)

for model in models:
    scores[model.__class__.__name__] = loop_nclusters(rng, model, X)


figsize = (12, 6)
title = '%s as a function of clusters'
xlabel = 'Number of clusters'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
ax1.plot(rng, scores['MiniBatchKMeans'][0])
ax1.set_title(title % 'K-Means Inertia')
ax1.set_ylabel('Inertia')
ax1.set_xlabel(xlabel)
ax2.plot(rng, scores['MiniBatchKMeans'][1],
         rng, scores['AgglomerativeClustering'][1])
ax2.set_title(title % 'Silhouette Coefficient')
ax2.set_ylabel('Silhouette Coefficient')
ax2.set_xlabel(xlabel)
ax2.legend(['MB KMeans', 'Aggl. Clustering'], loc='upper right', shadow=True)
plt.savefig(os.path.join(here, 'scores.png'))


if __name__ == '__main__':
    n_top_words = 10
    # Both NMF/LDA yield word/topic matrices, but only tf_vect has
    #     feature names (both are same) because we used TfidfTransformer
    for obj, name in zip((nmf, lda), ('nmf_df.csv', 'lda_df.csv')):
        # Columns: topic num (these are dict keys); Indices: word num
        # Dictionary
        df = pd.DataFrame.from_dict(dict(get_top_words(
        obj, tf_vect.get_feature_names(), n_top_words))).add_prefix('topic')
        df.to_csv(os.path.join(here, name))

    # Count of unigrams, bigrams, ...
    gramcounts = dict(enumerate([len([i for i in tf_vect.get_feature_names()
                                 if len(i.split()) == d]) for d in (1, 2, 3, 4)],
                                1))

    with open(os.path.join(here, 'ngramcounts.txt'), 'w') as f:
        print('ngram counts:', file=f)
        print(pd.Series(gramcounts, index=range(1, 5)), '\n', sep='', file=f)
        print('quadgrams:', file=f)
        print([i for i in tf_vect.get_feature_names() if len(i.split()) == 4],
              file=f)
        print('trigrams:', file=f)
        print([i for i in tf_vect.get_feature_names() if len(i.split()) == 3],
              file=f)
