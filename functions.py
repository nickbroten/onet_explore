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
style_SOC = pd.DataFrame({'SOC': styles['SOC']}).append(newline, ignore_index = True)
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

def text_to_vector(text):

    ## Counter word frequencies
    ps = PorterStemmer()

    words = WORD.findall(text)

    stemwords = []
    for w in words:
        stem_w = ps.stem(w)
        stemwords.append(stem_w)

    return Counter(stemwords)

def get_cosine(vec1, vec2):

    ## Calulate cosine similarity of two vectors
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

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



def return_closest(query, title_list):

    start = text_to_vector(query)

    cos = []
    for i in range(0, len(observe)):
        sim = get_cosine(start, observe['VECT'][i])
        cos.append(sim)

    t = pd.DataFrame({'SOC': observe['SOC'],
                              'ALT_TITLE': observe['ALT_TITLE'],
                              'SIM': cos}).dropna()

    test_set = t[t.SIM == t.SIM.max()]
    out = test_set['SOC'][test_set.index]

    #build output
    ability_c = ability[ability['SOC'].isin(out)].drop(['SOC'], axis = 1).mean()
    skills_c = skills[skills['SOC'].isin(out)].drop(['SOC'], axis = 1).mean()
    knowledge_c = knowledge[knowledge['SOC'].isin(out)].drop(['SOC'], axis = 1).mean()
    styles_c = styles[styles['SOC'].isin(out)].drop(['SOC'], axis = 1).mean()
    interests_c = interests[interests['SOC'].isin(out)].drop(['SOC'], axis = 1).mean()
    values_c = values[values['SOC'].isin(out)].drop(['SOC'], axis = 1).mean()

    means = [ability_c, skills_c, knowledge_c, styles_c, interests_c, values_c]

    return means


def clean_data(query):

    load_data = [ability, skills, knowledge, styles, interests, values]

    def make_df(file, index):

        means = return_closest(query, alt_titles)

        df = pd.DataFrame({'SOC': ['55']})
        for i in range(0, len(file.columns) - 1):
            name = str(i)
            df[name] = means[index][name]

        return df

    output = []
    for i in range(0, len(load_data)):
        d = load_data[i].append(make_df(load_data[i], i))
        d = pd.DataFrame(d.merge(style_SOC, on = 'SOC', how = 'right')).drop('SOC', axis = 1)
        output.append(d)

    return output

def make_fig_updates(query, q1, q2, q3, q4, q5, q6):

    ## Define labels for figure
    label_df = make_labels(SOC_cats)

    ## Load data
    data = clean_data(query)

    ## Build weighted datasets
    weights = [q1, q2, q3, q4, q5, q6]

    ## Concatenate weighted data
    for i in range(0, len(weights)):
        data[i] = data[i].to_numpy()
        data[i] = data[i] * weights[i]
    df = pd.concat([pd.DataFrame(data[0]),
                    pd.DataFrame(data[1]),
                    pd.DataFrame(data[2]),
                    pd.DataFrame(data[3]),
                    pd.DataFrame(data[4]),
                    pd.DataFrame(data[5])],
                    axis = 1)

    ## Run TSNE
    X_TSNE = TSNE(n_components=2, perplexity = 20).fit_transform(df)
    X_TSNE = pd.DataFrame(X_TSNE, columns = ['First component', 'Second component'])

    data = pd.concat([style_SOC, X_TSNE], axis = 1)
    data['Label'] = data['SOC'].str.slice(start=0, stop=2, step=1)
    data = data.merge(label_df, on = 'Label', how = 'left')
    data = data.merge(occs, on = 'SOC', how = 'left')

    fig = px.scatter(
        data,
        x = 'First component',
        y = 'Second component',
        color="Text",
        size = 'Size',
        hover_data=['SOC', 'Text', 'Title'])

    fig['layout'].update(height=1000, width=1800, title='t-SNE 2D Representation of O*NET Data')

    return fig

def return_fig(query, q1, q2, q3, q4, q5, q6):

    newfig = make_fig_updates(query, q1, q2, q3, q4, q5, q6)

    return(newfig)
