# Import module for data visualization
import pandas as pd
import numpy as np
import plotnine
import csv
from kmodes.kprototypes import KPrototypes
import pickle
import matplotlib.pyplot as plt

# Préparation des données
reader = csv.reader(open("encoded_mcu_features.csv", "r"), delimiter=",")
x = list(reader)
result = np.array(x)
N,nbs_cat = result.shape
print(N,nbs_cat)
cat_cols_pos = [0,1,5,7,9,10,13,14,21,22,32,35,36]
result = result[1:]
"""
for col in range(nbs_cat):
    try:
        A = result.T
        A[col].astype(float)
    except:
        cat_cols_pos.append(col)
"""

# El bow méthode
costs = []
K = range(10, 15)
for k in K:
    print("en cours, k = ",k)
    kproto = KPrototypes(n_clusters=k, init='Cao', n_init=1, verbose=2, n_jobs=-1)
    clusters = kproto.fit_predict(result, categorical=cat_cols_pos)
    costs.append(kproto.cost_)


# Visualiser la méthode du coude
plt.plot(K, costs, 'bx-')
plt.xlabel('k')
plt.ylabel('Cost')
plt.title('Méthode du coude pour déterminer le meilleur k')

# Sauvegarder la figure
plt.savefig('elbow_method.png')

# Afficher la figure si nécessaire
plt.show()