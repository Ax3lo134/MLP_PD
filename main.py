import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from glob import glob

fisiere_date = glob('S*')
pacient = pd.concat(pd.read_csv(file, sep=' ', names=["timp", "glezna oriz fata", "glezna vert", "glezna oriz lateral",
                                                      "picior oriz fata", "picior vert", "picior oriz lateral",
                                                      "sold oriz fata", "sold vert", "sold oriz lateral", "concluzii"])
                    for file in fisiere_date)
# 0 -> nu face parte din experiment
# 1 -> fara sindrom
# 2 -> cu sindrom 

# pacient = pd.read_csv('S01R01.txt', sep=' ', names=["timp", "glezna oriz fata","glezna vert","glezna oriz lateral",
# "picior oriz fata","picior vert","picior oriz lateral","sold oriz fata","sold vert","sold oriz lateral",
# "concluzii"])


# filtru pentru momentele de timp in care nu se desfasoara experimentul
pacient = pacient[pacient.concluzii > 0]

# am separat datele de intrare si datele de iesire.
# intrare: timpul, senzorii; iesire: daca sindromul apare (2) sau daca nu apare sindrom (1)
X = pacient.drop("concluzii", axis=1)
y = pacient["concluzii"]

# definim vectorii in care vom pastra datele de invatare si cei pe care ii vom folosi sa testam modelul de A.I.
# alegem cat la suta din datele pe care le avem la dispotitie sunt folosite pentru invatare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# folosim scalarea pentru a aduce toate variabilele la un nivel apropiat, pentru a nu exista diferente majore intre date
scalare = StandardScaler()
X_train = scalare.fit_transform(X_train)
X_test = scalare.transform(X_test)

# implementarea retelei neuronale
mlpc = MLPClassifier(hidden_layer_sizes=(10, 5), learning_rate_init=0.01)

mlpc.fit(X_train, y_train)

pred = mlpc.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))
