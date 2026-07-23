# %% Importando a biblioteca o conjunto de dados
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)

# %%
mnist.keys()

# %% Descrição dos dados
mnist.DESCR

# %% Verificando os conjuntos de dados
X, y = mnist.data, mnist.target
X

# %% Conferindo o tamanho do conjunto
X. shape
y.shape

# %% Visualizando um exemplo
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show

# %%
y[0]

# %% Convertendo Y para inteiro
import numpy as np
y = y.astype(np.uint8)

# %% Criando um conjunto de teste e treino
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[:60000:]

# %% Identificando o número 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# %% Treinando um modelo (Classificador Binário)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# %% Predizendo o valor
sgd_clf.predict([some_digit])

# %%
########################## IMPLEMENTANDO UMA VALIDAÇÃO CRUZADA ##########################

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfold = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

for train_index, test_index in skfold.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]

    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

# %% cross_val_score
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %% Classe-não-5
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X),1), dtype=bool)

never_5_clf = Never5Classifier()
score_final = cross_val_score(
    never_5_clf,
    X_train,
    y_train_5,
    cv=3,
    scoring="accuracy"
)

# %%
print(np.round(score_final, 3))

# %%
########################## MATRIZ DE CONFUSÃO ##########################
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(
    sgd_clf,
    X_train,
    y_train_5,
    cv=3
)

# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(
    y_train_5,
    y_train_pred
)
# %%
########################## PRECISÃO E REVOCAÇÃO ##########################

from sklearn.metrics import precision_score, recall_score

print("precision_score\n")
pr = precision_score(y_train_5, y_train_pred)
print(np.round(pr, 3))
print("----------------\n")

print("recall_score\n")
rs = recall_score(y_train_5, y_train_pred)
print(np.round(rs, 3))
print("----------------\n")

# %%
########################## FI SCORE ##########################
from sklearn.metrics import f1_score

f1 = f1_score(y_train_5, y_train_pred)
print(np.round(f1, 4))
# %%
