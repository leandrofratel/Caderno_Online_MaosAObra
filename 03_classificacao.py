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