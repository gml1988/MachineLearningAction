from __future__ import print_function

from scipy.spatial import distance
from gensim import corpora, models
from collections import defaultdict
import numpy as np
import nltk.stem
import nltk.corpus
import sklearn.datasets

english_stemmer = nltk.stem.SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords.update([''])
