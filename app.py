import numpy as np
import pandas as pd
import re

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from functions import text_to_vector, get_cosine, return_closest, make_fig_updates, make_labels, return_fig, clean_data

external_stylesheets = ['https://codepen.io/chriddyp/pen/wvKVvGo.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "relative",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

GRAPH_STYLE = {
    "position": "fixed",
    "top": 100,
    "left": 400,
    "bottom": 5,
    "width": "80rem",
    "padding": "2rem 1rem"
}


# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

slider_content = html.Div(

    [

    html.H1(children='O*NET Explorer'),

    html.Div(children='''
        The O*NET is a widely used source of information on occupations.
        One limitation of the database is its size and complexity, making comparisons of occupations difficult.
        This tool intends to enable intuitive exploration of the data by projecting the database onto two-dimensional space.
        Users can vary the importance of six O*NET categories to change the weights of features used in this projection.
        Note the t-SNE algorithm is non-deterministic and small changes to weights can produce significantly different visualizations.

    '''),

    dcc.Markdown(
        '''
        **What job are you interested in?**
        '''
    ),

    html.Div([dcc.Input(id='query', value='e.g., molecular biologist', type='text')]),


    html.Br(),

    dcc.Markdown(
        '''
        **Select importance of job ability requirements:**
        '''
    ),

    dcc.Slider(
        id='a1',
        min=0,
        max=1,
        value=.5,
        step=.01,
        marks = {
         0: '0.0',
        .1: '0.1',
        .2: '0.2',
        .3: '0.3',
        .4: '0.4',
        .5: '0.5',
        .6: '0.6',
        .7: '0.7',
        .8: '0.8',
        .9: '0.9',
         1: '1.0'
        }
    ),

    dcc.Markdown(
        '''
        **Select importance of job skill requirements:**
        '''
    ),

    dcc.Slider(
        id='a2',
        min=0,
        max=1,
        value=.5,
        step=.01,
        marks = {
         0: '0.0',
        .1: '0.1',
        .2: '0.2',
        .3: '0.3',
        .4: '0.4',
        .5: '0.5',
        .6: '0.6',
        .7: '0.7',
        .8: '0.8',
        .9: '0.9',
         1: '1.0'
        }
    ),

    dcc.Markdown(
        '''
        **Select importance of job knowledge requirements:**
        '''
    ),

    dcc.Slider(
        id='a3',
        min=0,
        max=1,
        value=.5,
        step=.01,
        marks = {
         0: '0.0',
        .1: '0.1',
        .2: '0.2',
        .3: '0.3',
        .4: '0.4',
        .5: '0.5',
        .6: '0.6',
        .7: '0.7',
        .8: '0.8',
        .9: '0.9',
         1: '1.0'
        }
    ),

    dcc.Markdown(
        '''
        **Select importance of work styles:**
        '''
    ),

    dcc.Slider(
        id='a4',
        min=0,
        max=1,
        value=.5,
        step=.01,
        marks = {
         0: '0.0',
        .1: '0.1',
        .2: '0.2',
        .3: '0.3',
        .4: '0.4',
        .5: '0.5',
        .6: '0.6',
        .7: '0.7',
        .8: '0.8',
        .9: '0.9',
         1: '1.0'
        }
    ),

    dcc.Markdown(
        '''
        **Select importance of job-related interests:**
        '''
    ),

    dcc.Slider(
        id='a5',
        min=0,
        max=1,
        value=.5,
        step=.01,
        marks = {
         0: '0.0',
        .1: '0.1',
        .2: '0.2',
        .3: '0.3',
        .4: '0.4',
        .5: '0.5',
        .6: '0.6',
        .7: '0.7',
        .8: '0.8',
        .9: '0.9',
         1: '1.0'
        }
    ),

    dcc.Markdown(
        '''
        **Select importance of work values:**
        '''
    ),

    dcc.Slider(
        id='a6',
        min=0,
        max=1,
        value=.5,
        step=.01,
        marks = {
         0: '0.0',
        .1: '0.1',
        .2: '0.2',
        .3: '0.3',
        .4: '0.4',
        .5: '0.5',
        .6: '0.6',
        .7: '0.7',
        .8: '0.8',
        .9: '0.9',
         1: '1.0'
        }
    )
    ]
)

sidebar = html.Div(
    [
        slider_content
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([

    sidebar,

    dcc.Graph('graph-with-slider', style = GRAPH_STYLE)

])

@app.callback(
    Output(component_id='graph-with-slider', component_property='figure'),
    [Input(component_id='query', component_property='value'),
    Input('a1', 'value'),
    Input('a2', 'value'),
    Input('a3', 'value'),
    Input('a4', 'value'),
    Input('a5', 'value'),
    Input('a6', 'value')]
)
def update_output_div(query, a1, a2, a3, a4, a5, a6):
    fig = return_fig(query, a1, a2, a3, a4, a5, a6)
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
