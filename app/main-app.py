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
import os
import json

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
            dbc.InputGroupText("Folder with raw images (Glob format): "), dbc.Input(placeholder='Folder dir.', id='folder-input')
        ]),
        dbc.InputGroup([
            dbc.DropdownMenu([
                dbc.DropdownMenuItem("ViT H (High)", id='vit_h_item'),
                dbc.DropdownMenuItem("ViT L (Medium)", id='vit_l_item'),
                dbc.DropdownMenuItem("ViT B (Low)", id='vit_b_item'),
            ], label='Model'), 
            dbc.Input(placeholder="Selected model...", id='model-selection-input')
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
    dcc.Store(id='files-storage'),
    dcc.Store(id='labels-storage'),
    dcc.Store(id='current-img-storage'),
    dcc.Store(id='current-obj-storage')
])

#-------------BACKEND---------------


@app.callback(
    [Output('image-info-div', 'children'),
     Output('files-storage', 'data'),
     Output('labels-storage', 'data'),
     Output('current-img-storage', 'data')],
    [Input('folder-input', 'value')]
)
def open_folder(folder_addr):
    if folder_addr is not None:
        files = glob(folder_addr)
        #print(files)
        if len(files)>0:
            result = [html.H5("Images in dataset: "), html.P("{}".format(len(files)))]
            files_json = json.dumps([*files])
            current_img_json = json.dumps([0])
            return [result, files_json, None, current_img_json]
        else:
            return ["Images not found.", None, None, None]
    else:
        return ["Not valid address.", None, None, None]

@app.callback(
    Output('current-img-div', 'children'),
    Input('next-img-button', 'n_clicks'),
    Input('prev-img-button', 'n_clicks'),
    State('files-storage', 'data')
)
def display_image(n_clicks_next, n_clicks_prev, files_data):
    if files_data:
        files_list = json.loads(files_data)
        n_images = len(files_list)
        index=0
        n_next = n_clicks_next if n_clicks_next is not None else 0
        n_prev = n_clicks_prev if n_clicks_prev is not None else 0
        print("Next: ", n_next)
        print("Prev: ", n_prev)
        #if n_clicks_next is None:
        #    index=0
        #elif n_clicks_prev is not None:
        dif = n_next - n_prev
        if dif>=0:
            index = min(n_images-1, dif)
        else:
            index = 0
        image = cv2.imread(files_list[index])
        _, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer.tobytes())
        dataURI = 'data:image/jpeg;base64,' + str(jpg_as_text, 'ascii')
        return [html.P("Image {}".format(files_list[index])),
                html.Img(src=dataURI)]
    else:
        return "No images"
    

if __name__ == '__main__':
    app.run(debug=True)
