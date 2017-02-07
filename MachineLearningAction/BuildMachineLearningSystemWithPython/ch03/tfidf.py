import scipy as sp

def tfidf(t,d,D):
    tf = d.count(t)
    idf = sp.log(len(D) / (len([doc for doc in D if t in doc])))
    return tf * idf