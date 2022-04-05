import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from glob import glob

# data import
files = glob('S*')
pacient = pd.concat(pd.read_csv(file, sep=' ', names=["time",
                                                      "forward ankle acceleration", "vert ankle acc", "ankle sides acc",
                                                      "forward feet acc", "vert feet acc", "feet sides acc",
                                                      "forward hip acc", "vert hip acc", "hip sides acc",
                                                      "tags"])
                    for file in files)
# 0 -> unrelated
# 1 -> no syndrome
# 2 -> syndrome


# filter for unrelated tags
pacient = pacient[pacient.concluzii > 0]

# selects the input and output data
# input (X) : time, acceleration sensors
# output (Y) : tags
X = pacient.drop("tags", axis=1)
y = pacient["tags"]

# defines training and testing vectors
# sets 25% of data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# scales data to have an equal proportion
scalare = StandardScaler()
X_train = scalare.fit_transform(X_train)
X_test = scalare.transform(X_test)

# MLP implementation
mlpc = MLPClassifier(hidden_layer_sizes=(10, 5), learning_rate_init=0.01)

mlpc.fit(X_train, y_train)

pred = mlpc.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))
