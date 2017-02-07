from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

# plot three graphs.
# The 1st dimension as x and 2nd dimension as y
# Target = 1, 2, 3
for t, marker, c in zip(range(3), ">ox", "rgb"):
    plt.scatter(features[target == t, 0],
                features[target == t, 1],
                marker=marker,
                color=c)

labels = target_names[target]

# get the petal length array for all the flower instances
petal_length = features[:, 2]

is_setosa = (labels == 'setosa')

max_setosa = petal_length[is_setosa].max()
min_non_setosa = petal_length[~is_setosa].min()
print('Maximum of setosa: {0}'.format(max_setosa))
print('Minimum of others: {0}'.format(min_non_setosa))

features = features[~is_setosa]
labels = labels[~is_setosa]

is_virginica = (labels == 'virginica')

# Find which feature we should use, which value we use as threshold
best_acc = -1.0
for fi in range(features.shape[1]):
    thresh = features[:, fi]
    for t in thresh:
        pred = (features[:, fi] > t)
        acc = (pred == is_virginica).mean()
        rev_acc = (pred == ~is_virginica).mean()
        if rev_acc > acc:
            reverse = True
            acc = rev_acc
        else:
            reverse = False

        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t
            best_reverse = reverse

print(best_fi, best_t, best_reverse, best_acc)

features_len = len(features)
features_shape = features.shape

# Cross validation, caculate the training error and testing error
from threshold import learn_model, apply_model
error = 0.0
correct = 0.0
for ei in range(len(features)):
    # select all but the one at position 'ei' as training data
    training = np.ones(len(features), bool)
    training[ei] = False
    testing = ~training
    model = learn_model(features[training], is_virginica[training])
    predictions = apply_model(model, features[testing])
    correct += np.sum(predictions == is_virginica[testing])
accuracy = correct / float(len(features))

from load import load_dataset

features, labels = load_dataset('seeds')

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)
from sklearn.cross_validation import KFold
kf = KFold(len(features), n_folds=5, shuffle=True)
means = []
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    predictions = classifier.predict(features[testing])
    current_mean = np.mean(predictions == labels[testing])
    means.append(current_mean)
print('Mean accuracy: {:.1%}'.format(np.mean(means)))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

classifier = KNeighborsClassifier(n_neighbors=1)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training, testing in kf:
    classifier.fit(features[training], labels[training])
    predictions = classifier.predict(features[testing])
    current_mean = np.mean(predictions == labels[testing])
    means.append(current_mean)
print('Mean accuracy: {:1%}'.format(np.mean(means)))