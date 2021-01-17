import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.express as px
from whitenoise import WhiteNoise

from functions import make_fig_updates, make_labels, return_fig, sliders

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
server.wsgi_app = WhiteNoise(server.wsgi_app, root='static/')

occlist = pd.read_pickle('static/occs.pkl')
styles = pd.read_pickle('static/style.pkl')
style_SOC = pd.DataFrame({'SOC': styles['SOC']})
occlist = occlist.merge(style_SOC, on = 'SOC', how = 'right')
occlist.columns = ['value', 'label']

options = []
for col in occlist.columns:
    for o in range(0, len(occlist)):
        options.append({'label': occlist['label'][o].format(col, col), 'value': occlist['value'][o]})

SIDEBAR_STYLE = {
    "position": "relative",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "40rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

GRAPH_STYLE = {
    "position": "fixed",
    "top": 5,
    "left": 500,
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

        **Select an occupation to explore:**

        '''
    ),

    dcc.Dropdown(
        id = 'occ-dropdown',
        options= options,
        value= '11-3051.03'

    ),
    html.Label(id='my_label1'),

    html.Div(id='occ-output-container'),

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
        value= 0.5,
        step=.05,
        marks = {
         0: 'Not important',
        0.5: 'Somewhat important',
         1: 'Very important'
        }
    ),

    html.Br(),

    dcc.Markdown(
        '''

        **Select importance of job skill requirements:**

        '''
    ),

    dcc.Slider(
        id='a2',
        min=0,
        max=1,
        value= 0.5,
        step=.05,
        marks = {
         0: 'Not important',
        0.5: 'Somewhat important',
         1: 'Very important'
        }
    ),

    html.Br(),

    dcc.Markdown(
        '''

        **Select importance of job knowledge requirements:**

        '''
    ),

    dcc.Slider(
        id='a3',
        min=0,
        max=1,
        value= 0.5,
        step=.05,
        marks = {
         0: 'Not important',
        0.5: 'Somewhat important',
         1: 'Very important'
        }
    ),

    html.Br(),

    dcc.Markdown(
        '''

        **Select importance of work styles:**

        '''
    ),

    dcc.Slider(
        id='a4',
        min=0,
        max=1,
        value= 0.5,
        step=.05,
        marks = {
         0: 'Not important',
        0.5: 'Somewhat important',
         1: 'Very important'
        }
    ),

    html.Br(),

    dcc.Markdown(
        '''

        **Select importance of job-related interests:**

        '''
    ),

    dcc.Slider(
        id='a5',
        min=0,
        max=1,
        value= 0.5,
        step=.05,
        marks = {
         0: 'Not important',
        0.5: 'Somewhat important',
         1: 'Very important'
        }
    ),

    html.Br(),

    dcc.Markdown(
        '''

        **Select importance of work values:**

        '''
    ),

    dcc.Slider(
        id='a6',
        min=0,
        max=1,
        value= 0.5,
        step=.05,
        marks = {
         0: 'Not important',
        0.5: 'Somewhat important',
         1: 'Very important'
        }
    ),

    html.Br(),

    dcc.Markdown(
            '''

            **NOTE: job requirement weights only have three levels: low, medium, and high. Slider values between 0 and 0.25 will register as low, 0.25 and 0.75 as medium, and 0.75 and 1 as high.**

            '''
    ),

    ]
)

sidebar = html.Div(
    [slider_content],
    style=SIDEBAR_STYLE,
)

app.layout = html.Div([
    sidebar,
    html.Button('Submit', id='submit-val', n_clicks=0),
    dcc.Graph('graph-with-slider', style = GRAPH_STYLE)
])


@app.callback(
    dash.dependencies.Output(component_id='graph-with-slider', component_property='figure'),
    [dash.dependencies.Input('submit-val', 'n_clicks'),
    dash.dependencies.Input('occ-dropdown', 'value'),
    dash.dependencies.Input('a1', 'value'),
    dash.dependencies.Input('a2', 'value'),
    dash.dependencies.Input('a3', 'value'),
    dash.dependencies.Input('a4', 'value'),
    dash.dependencies.Input('a5', 'value'),
    dash.dependencies.Input('a6', 'value')],
    [dash.dependencies.State('submit-val', 'value')])
def update_output(n_clicks, query, a1, a2, a3, a4, a5, a6, state):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if n_clicks == 0:
        fig = make_fig_updates('11-3051.03', 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        return fig

    elif n_clicks > 0 and 'submit-val' in changed_id:
        fig = make_fig_updates(query, a1, a2, a3, a4, a5, a6)
        return fig

    else:
        raise PreventUpdate

if __name__ == '__main__':
    app.run_server(debug=False)
