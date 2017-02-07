from __future__ import  print_function
import numpy as np
import gensim
from os import path

if not path.exists('wiki_lda.pkl'):
    import sys
    sys.stderr.write('')
    sys.exit(1)

id2word = gensim.corpora.Dictionary.load_from_text('')
mm = gensim.corpora.MmCorpus('')

model = gensim.models.LdaModel.load('wiki_lda.pkl')

