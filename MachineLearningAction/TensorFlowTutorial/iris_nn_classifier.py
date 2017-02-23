from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys

IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

my_path = os.path.abspath(__file__)
my_dir = os.path.dirname(my_path)
iris_training_file_path = os.path.join(my_dir, IRIS_TRAINING)
iris_test_file_path = os.path.join(my_dir, IRIS_TEST)

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=iris_training_file_path,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=iris_test_file_path,
    target_dtype=np.int,
    features_dtype=np.float32)

# Just like a place holder. The feature vector that will be filled into the Classifier will be
# Dimension = 4, the value type will be float
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="tmp/iris_model")

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000)

accuracy_sore = classifier.evaluate(x=test_set.data,
                                    y=test_set.target)["accuracy"]

new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))