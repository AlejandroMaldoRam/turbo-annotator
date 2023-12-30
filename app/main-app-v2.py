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
from pprint import pprint

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
detections_list = []
print("Ready.")

#--------------FRONTEND-------------
app.layout = html.Div([
    html.H1("TURBO ANNOTATOR"),
    html.Div([
        html.H3("Annotator config."),
        dbc.InputGroup([
            dbc.InputGroupText("Folder with detection file: "), dbc.Input(placeholder='Folder dir.', id='folder-input')
        ])
    ]),
    html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.ButtonGroup([dbc.Button("Prev", id='prev-img-button'), dbc.Button("Save", id='save-img-button'), dbc.Button("Next", id='next-img-button')])
                ]),
                dbc.Row([
                    dbc.Progress(id='img-progress')
                ]),
                dbc.Row(html.Div(id='current-img-div')),
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
                dbc.Row([
                    dbc.ButtonGroup([dbc.Button("Prev", id='prev-obj-button'), dbc.Button("Save", id='save-obj-button'), dbc.Button("Next", id='next-obj-button')])
                ]),
                dbc.Row([
                    dbc.Progress(id='obj-progress')
                ]),
                dbc.Row(html.Div(id='current-object-div')),
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
    dcc.Store(id='detections-storage'),
    dcc.Store(id='labels-storage'),
    dcc.Store(id='current-img-storage')
])

#-------------BACKEND---------------


@app.callback(
    Output('image-info-div', 'children'),
    Output('detections-storage', 'data'),
    Input('folder-input', 'value')
)
def open_folder(folder_addr):
    if folder_addr is not None:
        f = open(folder_addr, 'r')
        detections_list = json.load(f)
        f.close()
        n_images = len(detections_list)
        
        if n_images>0:
            result = [html.H5("Images in dataset: "), html.P("{}".format(n_images))]
            
            return [result, json.dumps(detections_list)]
        else:
            return ["Images not found.", None]
    else:
        return ["Not valid address.", None]

@app.callback(
    Output('current-img-div', 'children'),
    Output('obj-info-div', 'children'),
    Output('img-progress', 'value'),
    Output('current-img-storage', 'data'),
    Input('next-img-button', 'n_clicks'),
    Input('prev-img-button', 'n_clicks'),
    State('detections-storage', 'data')
)
def display_image(n_clicks_next, n_clicks_prev, detections_json):
    if detections_json:
        detections_list = json.loads(detections_json)
        print("images: ", len(detections_list))
        if len(detections_list)>0:
            # get index
            n_images = len(detections_list)
            index=0
            n_next = n_clicks_next if n_clicks_next is not None else 0
            n_prev = n_clicks_prev if n_clicks_prev is not None else 0
            print("Next: ", n_next)
            print("Prev: ", n_prev)
            dif = n_next - n_prev
            if dif>=0:
                index = min(n_images-1, dif)
            else:
                index = 0
            print("Index: ", index)
            # get image according to index
            image = cv2.imread(detections_list[index]['image_addr'])
            detections = detections_list[index]['detections']

            # draw detections
            detections_results = detector.draw_results(image, detections)
            _, buffer = cv2.imencode('.jpg', detections_results)
            jpg_as_text = base64.b64encode(buffer.tobytes())
            dataURI = 'data:image/jpeg;base64,' + str(jpg_as_text, 'ascii')
            progress_img = int(index/n_images*100)
            
            # ensamble results
            return [[html.P("Image {}".format(detections_list[index]['image_addr'])),
                    html.Img(src=dataURI)], 
                    [html.H5("Detected objects: "),
                        html.P("{}".format(len(detections)))],
                    progress_img,
                    json.dumps([index])]
        else:
            return ["No images", "No objects", 0, json.dumps([-1])]
    else:
        return ["No images", "No objects", 0, json.dumps([-1])]


@app.callback(
    Output('current-object-div', 'children'),
    Output('obj-progress', 'value'),
    Input('next-obj-button', 'n_clicks'),
    Input('prev-obj-button', 'n_clicks'),
    State('detections-storage', 'data'),
    State('current-img-storage', 'data')
)
def display_object(n_clicks_next, n_clicks_prev, detections_json, current_img_json):
    if detections_json:
        detections_list = json.loads(detections_json)
        if len(detections_list)>0:
            current_img = json.loads(current_img_json)[0]
            detections = detections_list[current_img]['detections']
            n_objects = len(detections)
            
            index=0
            n_next = n_clicks_next if n_clicks_next is not None else 0
            n_prev = n_clicks_prev if n_clicks_prev is not None else 0
            print("Next: ", n_next)
            print("Prev: ", n_prev)
            dif = n_next - n_prev
            if dif>=0:
                index = min(n_objects-1, dif)
            else:
                index = 0
            
            detection = detections[index]
            print("Object: ", detection)

            image = cv2.imread(detections_list[current_img]['image_addr'])
            detected_object = detector.extract_object(image, detection)

            _, buffer = cv2.imencode('.jpg', detected_object)
            jpg_as_text = base64.b64encode(buffer.tobytes())
            dataURI = 'data:image/jpeg;base64,' + str(jpg_as_text, 'ascii')
            progress_obj = int(index/n_objects*100)
            
            # ensamble results
            return [html.Img(src=dataURI), progress_obj]
        else:
            return ["No objects", 0]    
    else:
        return ["No objects", 0]

if __name__ == '__main__':
    app.run(debug=True)
