# Import module for data visualization
import pandas as pd
import numpy as np
import plotnine
import csv
from kmodes.kprototypes import KPrototypes
import pickle
# IMPORT df FROM csv FILE
# TO-DO
reader = csv.reader(open("small_edit_mcu_features.csv", "r"), delimiter=",")
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
print("col_cols_pos : ",cat_cols_pos)



cluster = 10
kprototype = KPrototypes(n_jobs = -1, n_clusters = cluster, init = 'Huang', random_state = 0)
# df_matrix à une shape de 
print("coucou1")
kprototype.fit_predict(result, categorical = cat_cols_pos)
print("coucou2")

print('Cluster initiation: {}'.format(cluster))

fsave = open("kprototype","wb")
pickle.dump(kprototype,fsave)
fsave.close()