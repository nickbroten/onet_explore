import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
import re
from collections import Counter
import math

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.manifold import TSNE
### Load data
ability = pd.read_pickle('static/ability.pkl')
skills = pd.read_pickle('static/skills.pkl')
knowledge = pd.read_pickle('static/knowledge.pkl')
styles = pd.read_pickle('static/style.pkl')
interests = pd.read_pickle('static/interests.pkl')
values = pd.read_pickle('static/values.pkl')
style_SOC = pd.DataFrame({'SOC': styles['SOC']})
occs = pd.read_pickle('static/occs.pkl')

load_data = [ability, skills, knowledge, styles, interests, values]
clean_data = []
for index in load_data:
    d = pd.merge(style_SOC[0:967], index, on = 'SOC', how = 'left')
    clean_data.append(d)

WORD = re.compile(r"\w+")

SOC_cats = ['Management',
        'Business and Financial Operations',
        'Computers and Math',
        'Architechture and Engineering',
        'Life, Physical, and Social Science',
        'Community and Social Services',
        'Legal',
        'Education, Training, and Library',
        'Arts, Design, Entertainment, and Media',
        'Healthcare Practitioner',
        'Healthcare Support',
        'Protective Service',
        'Food Preparation',
        'Building and Grounds Cleaning and Maintenance',
        'Personal Care and Service',
        'Sales and Related',
        'Office and Administrative Support',
        'Farming, Fishing, and Forestry',
        'Construction and Extraction',
        'Installation, Maintenance, and Repair',
        'Production',
        'Transportation and Material Moving' ]


def make_labels(cats):

    ## Make a dataframe with 2-digit SOC labels
    label = []
    i = 11
    while i < 55:
        i_str = str(i)
        i = i + 2
        label.append(i_str)

    label_df = pd.DataFrame({
        'Label': label,
        'Occupation Category': cats
    })

    return label_df

def make_fig_updates(query, q1, q2, q3, q4, q5, q6):

    ## Define labels for figure
    label_df = make_labels(SOC_cats)

    ## Build weighted datasets
    weights = [q1, q2, q3, q4, q5, q6]

    data = []
    for i in range(0, len(weights)):
        clean = clean_data[i].drop('SOC', axis = 1).to_numpy() * weights[i]
        data.append(pd.DataFrame(clean))

    df = pd.concat(data, axis = 1)

    ## Run TSNE
    X_TSNE = TSNE(n_components=2, perplexity = 20).fit_transform(df)
    X_TSNE = pd.DataFrame(X_TSNE, columns = ['First component', 'Second component'])
    X_TSNE['SOC'] = style_SOC
    X_TSNE['Label'] = X_TSNE['SOC'].str.slice(start=0, stop=2, step=1)
    X_TSNE = X_TSNE.merge(label_df, on = 'Label', how = 'left')
    X_TSNE['Size'] = np.ones(len(X_TSNE)) / 2
    X_TSNE.loc[X_TSNE['SOC'] == query, 'Size'] = 8
    X_TSNE = X_TSNE.merge(occs, on = 'SOC', how = 'left')

    fig = px.scatter(
        X_TSNE,
        x = 'First component',
        y = 'Second component',
        color = 'Occupation Category',
        size = 'Size',
        hover_data = ['SOC', 'Occupation Category', 'Title'])

    fig['layout'].update(height=1000, width=1500, title='t-SNE 2D Representation of O*NET Data', font=dict(family= 'Courier New, monospace', size=24, color='black'))

    return fig

def return_fig(query, q1, q2, q3, q4, q5, q6):

    newfig = make_fig_updates(query, q1, q2, q3, q4, q5, q6)

    return(newfig)
