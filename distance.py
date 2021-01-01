import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objs as go

ability = pd.read_csv('data/ability.csv').drop('SOC', axis = 1)

X_embedded = TSNE(n_components=3).fit_transform(ability)
X_embedded = pd.DataFrame(X_embedded, columns = ['0', '1', '2'])

axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
)

scatter = go.Scatter3d(
    name='NAME',
    x=X_embedded['0'],
    y=X_embedded['1'],
    z=X_embedded['2'])

figure = go.Figure(data=[scatter], layout=layout)
