#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:38:16 2019

@author: frank
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# from dash.exceptions import PreventUpdate
import numpy as np
import pandas as pd
import time
import joblib
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor

# encoding columns
ec_cols = ['state', 'make', 'model', 'trim', 'body', 'engine', 'drivetrain', 'fueltype', 'condition']
# non-encoding columns
nec_cols = ['year', 'mileage']

# Read in data from pre-processed pkl file
t0 = time.time()
data = pd.read_pickle('data.pkl')
print('Done reading data. Used time: {:.3f}s'.format(time.time() - t0))

# Create data with selected columns for later use
#X = data[ec_cols+nec_cols]
#y = data[['price']]

# Using Target Encoder to encode training data and testing data
t1 = time.time()
encoder = joblib.load('encoder.joblib')
print('Done loading data encoder. Used time: {:.3f}s'.format(time.time() - t1))
t2 = time.time()
rf = joblib.load('rf1.sav')
print('Done loading model. Used time: {:.3f}s'.format(time.time() - t2))
print('Done preparing data. Used time: {:.3f}s'.format(time.time() - t0))

test = data.head(100)

#all_options = {k: f.groupby('model', observed = True)['trim'].apply(set).apply(list).to_dict() for k, f in test.groupby('make', observed = True)}
all_options = joblib.load('dic.joblib')
make_list = sorted(list(all_options.keys()))
state_list = ['All state']+sorted(list(set(data.state)))
year_list = ['All year']+list(np.arange(1990,2021,1))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children='TradePending Capstone Project with MIDS'),
    html.Div(children='''
        Team members: Frank Xu, Abhishek Angadi.
        '''),
    dcc.Dropdown(id='make-dropdown',
                 options=[{'label': k, 'value': k} for k in make_list],
                 style={'width':'10vw', 'display': 'inline-block'},
                 value = 'BMW'),
    dcc.Dropdown(id='model-dropdown', 
                 style={'width':'10vw', 'display': 'inline-block'},
                 value = 'X5'),
    dcc.Dropdown(id='trim-dropdown', 
                 style={'width':'10vw', 'display': 'inline-block'},
                 value = 'All trim'),
    dcc.Dropdown(id='state-dropdown', 
                 options=[{'label': s, 'value': s} for s in state_list],
                 style={'width':'10vw', 'display': 'inline-block'},
                 value='All state'),
    dcc.Dropdown(id='year-dropdown', 
                 options=[{'label': y, 'value': y} for y in year_list],
                 style={'width':'8vw', 'display': 'inline-block'},
                 value='All year'),
    html.Button('Go', id='show',
                style={'width':'5vw', 'display': 'inline-block'}),
    html.Hr(),
    html.Div(id='display scatter plot')
])

# Model Dropdown Menu
@app.callback(
    Output('model-dropdown', 'options'),
    [Input('make-dropdown', 'value')])
def set_model_options(make):
    return [{'label': i, 'value': i} for i in all_options[make]]

@app.callback(
    Output('model-dropdown', 'value'),
    [Input('model-dropdown', 'options')])
def set_model_value(available_options):
    return available_options[0]['value']

# Trim Dropdown Menu
@app.callback(
    Output('trim-dropdown', 'options'),
    [Input('make-dropdown', 'value'),
     Input('model-dropdown', 'value')])
def set_trim_options(make, model):
    return [{'label': i, 'value': i} for i in ['All trim']+list(all_options[make][model])]

@app.callback(
    Output('trim-dropdown', 'value'),
    [Input('trim-dropdown', 'options')])
def set_trim_value(available_options):
    return available_options[0]['value']

# Return updated graph
@app.callback(
    Output('display scatter plot', 'children'),
    [Input('show', 'n_clicks')],
     state=[State('make-dropdown', 'value'),
     State('model-dropdown', 'value'),
     State('trim-dropdown', 'value'),
     State('state-dropdown', 'value'),
     State('year-dropdown', 'value')])
def update_graph(n_clicks, make, model, trim, state, year):
    if n_clicks is None:
        return html.Div('Please select vehicle information above!')
    try:
        groupby_list = ['make', 'model']
        df = data[data.make.isin([make])]
        df = df[df.model.isin([model])]
        if trim != 'All trim':
            df = df[df.trim.isin([trim])]
            groupby_list.append('trim')
        if year != 'All year':
            df = df[df.year.isin([year])]
            groupby_list.append('year')
        if state != 'All state':
            df = df[df.state.isin([state])]
            groupby_list.append('state')
        df_X = df[ec_cols+nec_cols]
        df_X_ec = encoder.transform(df_X)
        df2 = df_X_ec.groupby(groupby_list).agg('mean').reset_index()[ec_cols+nec_cols]
        df2 = pd.concat([df2]*50, ignore_index = True)
        df2.mileage = np.arange(1000,250000,5000)
        return dcc.Graph(id='scatter',
                         figure={'data': [
                                    dict(x=df.mileage,
                                         y=df.price,
                                         name='Real transactions',
                                         mode='markers',
                                         marker={'size': 5, 'opacity': 0.6}),
                                    dict(x = df2.mileage,
                                         #y = np.arange(10000,35000,1000),
                                         y = rf.predict(df2),
                                         name='Price prediction',
                                         mode='lines')],
                                'layout': 
                                    dict(autosize=False,
                                         width=800,
                                         height=600,
                                         title='Price Prediction of a {} {} {} from state {} of year {}'.format(make,model,trim,state,year),
                                         xaxis={'title': 'Mileage',
                                                'range':[0,250000]},
                                         yaxis={'title': 'Price'})
                        })
    except:
        return html.Div('There is no vehicle data for selected info')

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8888, debug=True, use_reloader=False)
