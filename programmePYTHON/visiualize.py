# perform PCA on the numeric columns
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import pickle
import matplotlib.pyplot as pyo
from matplotlib.figure import Figure
import plotly.graph_objects as po
from kmodes.kprototypes import KPrototypes

# charge les clusters entrainé
file = open('kprototype', 'rb')
kprototype = pickle.load(file)
file.close()


# suprime les features catégoriels
cat_cols_pos = [20,21,22,23,24,25,26,27,28,29,30]
K,nbs_feat = kprototype.cluster_centroids_.shape
nbs_cols_pos = [i for i in range(nbs_feat) if not(i in cat_cols_pos)]
centroid_number = np.array([[kprototype.cluster_centroids_[k,feature] for feature in nbs_cols_pos] for k in range(K)])
print(centroid_number)



pca = PCA(n_components=3)
df_pca = pca.fit_transform(centroid_number)
df_pca = pd.DataFrame(df_pca, columns=['PC1', 'PC2', 'PC3'])
 
# create a 3D scatter plot with colored markers and text labels
fig = po.Figure(data=[po.Scatter3d(
    x=df_pca['PC1'],
    y=df_pca['PC2'],
    z=df_pca['PC3'],
    mode='markers',
    name='',  # Set the name to an empty string
    marker=dict(
        size=5,
        colorscale='Viridis',
        colorbar=dict(
            tickvals=list(centroid_number),
            ticktext=[f'Cluster {i}' for i in range(K)]
        ),
    )
)])
 
fig.update_layout(
    scene=dict(
        xaxis=dict(
            title='',
            tickvals=[],
            ticktext=[],
            visible=False
        ),
        yaxis=dict(
            title='',
            tickvals=[],
            ticktext=[],
            visible=False
        ),
        zaxis=dict(
            title='',
            tickvals=[],
            ticktext=[],
            visible=False
        )
    )
)
 
# save the plot as an HTML file
pyo.plot(fig, filename='scatter.html')
 
fig.show()