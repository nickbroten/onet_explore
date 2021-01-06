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
alt_titles = pd.read_pickle('static/alt_titles.pkl')
ability = pd.read_pickle('static/ability.pkl')
skills = pd.read_pickle('static/skills.pkl')
knowledge = pd.read_pickle('static/knowledge.pkl')
styles = pd.read_pickle('static/style.pkl')
interests = pd.read_pickle('static/interests.pkl')
values = pd.read_pickle('static/values.pkl')
newline = pd.DataFrame({'SOC': ['55']})
styles = pd.read_pickle('static/style.pkl')
style_SOC = pd.DataFrame({'SOC': styles['SOC']})
observe = pd.read_pickle('static/observe.pkl')
occs = pd.read_pickle('static/occs.pkl')

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
        'Transportation and Material Moving',
        'YOUR QUERY' ]

def get_cosine(dic1, dic2):
    numerator = 0
    dena = 0
    for key1,val1 in dic1.items():
        numerator += val1*dic2.get(key1, 0.0)
        dena += val1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    return numerator/math.sqrt(dena*denb)


def text_to_vector(text):

    ## Counter word frequencies
    ps = PorterStemmer()

    words = WORD.findall(text)

    stemwords = []
    for w in words:
        stem_w = ps.stem(w)
        stemwords.append(stem_w)

    return Counter(stemwords)

def make_labels(cats):

    ## Make a dataframe with 2-digit SOC labels
    label = []
    i = 11
    while i < 57:
        i_str = str(i)
        i = i + 2
        label.append(i_str)

    size = []
    for i in range(0, len(label) - 1):
        size.append(0.5)
    size.append(8)

    label_df = pd.DataFrame({
        'Label': label,
        'Text': cats,
        'Size': size
    })

    return label_df

def make_fig_updates(query, q1, q2, q3, q4, q5, q6):

    ## Define labels for figure
    label_df = make_labels(SOC_cats)
    load_data = [ability, skills, knowledge, styles, interests, values]
    clean_data = []
    for index in load_data:
        d = pd.merge(style_SOC[0:967], index, on = 'SOC', how = 'left')
        clean_data.append(d)
    ## Build weighted datasets
    weights = [q1, q2, q3, q4, q5, q6]

    label_df = make_labels(SOC_cats)
    SOC_set = []
    sim = 0
    for i in range(0, len(observe)):
        sim_est = get_cosine(dict(text_to_vector(query)), dict(observe['VECT'][i]))
        if sim_est > sim:
            SOC_set = [observe['SOC'][i]]
            sim = sim_est
        elif sim_est > 0 and sim_est == sim:
            SOC_set.append(observe['SOC'][i])
            sim = sim_est


    data = []
    for i in range(0, len(weights)):
        clean = clean_data[i].drop('SOC', axis = 1).to_numpy() * weights[i]
        q = clean_data[i][clean_data[i]['SOC'].isin(SOC_set)].drop('SOC', axis = 1).mean() * weights[i]
        df = pd.DataFrame(clean)
        df.loc[len(df)] = q.to_list()
        data.append(df)

    ## Run TSNE
    X_TSNE = TSNE(n_components=2, perplexity = 20).fit_transform(df)

    X_TSNE = pd.DataFrame(X_TSNE, columns = ['First component', 'Second component'])

    fig = px.scatter(
        X_TSNE,
        x = 'First component',
        y = 'Second component')

    fig['layout'].update(height=1000, width=1200, title='t-SNE 2D Representation of O*NET Data')

    return fig

def return_fig(query, q1, q2, q3, q4, q5, q6):

    newfig = make_fig_updates(query, q1, q2, q3, q4, q5, q6)

    return(newfig)
