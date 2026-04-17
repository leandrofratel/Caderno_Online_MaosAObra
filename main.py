# %%
import os
from zlib import crc32
import tarfile
import numpy as np
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt

# %%
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# %% Realizando o Donwload da base de dados.
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

# %% Convertendo a tabela em um DataFrame
def load_housing_data(housing_path= HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing_df =  load_housing_data()

# %% Gerando tabelas para a exploração dos dados
housing_df.hist(bins=50, figsize=(20,15))
plt.show()

# %% Separando o conjunto de treinamento
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data), )
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing_df, 0.2)

# %% Criando um identificador e estabilizando o conjunto de train/test
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing_df.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')

# %% Transformando latitude e longitude em ids
housing_with_id["id"] = housing_df["longitude"] * 1000 + housing_df["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# %% Usando o sklearn para separa os dados de train/test
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(
    housing_df,
    test_size=0.2,
    random_state=42
)

# %% Histograma de preço médio
housing_df["income_cat"] = pd.cut(
    housing_df["median_income"],
    bins=[0, 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

housing_df["income_cat"].hist()
# %%
