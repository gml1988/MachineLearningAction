import scipy as sp
import sklearn.datasets

groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.fetch_20newsgroups(subset='train',
                                                 categories=groups)

labels = train_data.target
num_clusters = sp.unique(labels).shape[0]

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')

from sklearn.feature_extraction.text import TfidfVectorizer

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        # Call the build_analyzer of the parent class TfidfVectorizer at first
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                    stop_words='english', decode_error='ignore')

selected_rows = train_data.data[0:100]

vectorized = vectorizer.fit_transform(selected_rows)
# vectorized = vectorizer.fit_transform(train_data.data)

post_group = zip(train_data.data, train_data.target)

all = [(len(post[0]), post[0], train_data.target_names[post[1]])
       for post in post_group]
graphics = sorted([post for post in all if post[2] == 'comp.graphics'])
print(graphics[5])

noise_post = graphics[5][1]

analyzer = vectorizer.build_analyzer()
print(list(analyzer(noise_post)))

useful = set(analyzer(noise_post)).intersection(vectorizer.get_feature_names())
print(sorted(useful))

for feature_name in vectorizer.get_feature_names():
    print(feature_name)

for key in vectorizer.vocabulary():
    print(key)
