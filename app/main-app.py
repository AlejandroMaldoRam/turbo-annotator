# Entry point of the application
from dash import html, Dash, dcc, Input, Output, State, callback, dash_table, ALL
import dash_bootstrap_components as dbc
import pandas as pd
from glob import glob
import numpy as np
import datetime
import time
import cv2
import base64
from PIL import Image

from SAMDetector import SAMDetector

#--------------SERVER PREPARATION-----
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
device = 'cuda'

#--------------DATA AND MODELS--------
#detector = TemplateProductDetector("/home/amaldonado/Code/cv-order-validation/dataset/ds1/train", "/home/amaldonado/Code/sam-test/models/sam_vit_b_01ec64.pth","vit_b","/home/amaldonado/Code/cv-order-validation/models/mobilenet_v3_large.tflite")
detector = SAMDetector.SAMDetector("/home/amaldonado/Code/sam-test/models/sam_vit_l_0b3195.pth","vit_l")
print("Loading catalog...")
df_labels = pd.read_csv("/home/amaldonado/Code/cv-order-validation/dataset/ds1/train/labels.csv")
df_labels['name'] = df_labels['Producto']+'|'+df_labels['Marca']+ '|'+df_labels['Empaque']
names = df_labels['name'].tolist()

print("Ready.")

#--------------FRONTEND-------------
app.layout = html.Div([
    html.H1("TURBO ANNOTATOR"),
    html.Div([
        html.H3("Annotator config."),
        dbc.InputGroup([
            dbc.InputGroupText("Folder with raw images: "), dbc.Input(placeholder='Folder dir.', id='folder_input')
        ]),
        dbc.InputGroup([
            dbc.DropdownMenu([
                dbc.DropdownMenuItem("ViT H (High)", id='vit_h_item'),
                dbc.DropdownMenuItem("ViT L (Medium)", id='vit_l_item'),
                dbc.DropdownMenuItem("ViT B (Low)", id='vit_b_item'),
            ], label='Model'), 
            dbc.Input(placeholder="Selected model...", id='model_selection_input')
        ])
    ]),
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row(html.Div(id='current-img-div')),
                dbc.Row([
                    dbc.ButtonGroup([dbc.Button("Prev", id='prev-img-button'), dbc.Button("Save", id='save-img-button'), dbc.Button("Next", id='next-img-button')])
                ]),
                dbc.Row([
                    dbc.Progress(id='img-progress')
                ]),
                dbc.Row([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("About the image"),
                            html.Div(id='image-info-div')
                        ])
                    )
                ])
            ]),
            dbc.Col([
                dbc.Row(html.Div(id='current-object-div')),
                dbc.Row([
                    dbc.ButtonGroup([dbc.Button("Prev", id='prev-obj-button'), dbc.Button("Save", id='save-obj-button'), dbc.Button("Next", id='next-obj-button')])
                ]),
                dbc.Row([
                    dbc.Progress(id='obj-progress')
                ]),
                dbc.Row([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("About the object"),
                            html.Div(id='obj-info-div')
                        ])
                    )
                ])
            ])
        ])
    ]),

])

#-------------BACKEND---------------

    
if __name__ == '__main__':
    app.run(debug=True)
