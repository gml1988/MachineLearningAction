import tempfile
import urllib.request

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.request.urlretrieve(url="http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", filename=train_file.name)
urllib.request.urlretrieve(url="http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test", filename=test_file.name)

import pandas as pd
COLUMNS = ["", ""]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)