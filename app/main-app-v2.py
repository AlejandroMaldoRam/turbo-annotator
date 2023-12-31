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
df_labels = pd.read_csv("/home/amaldonado/Datasets/MB/DS1/labels.csv")
df_labels['name'] = df_labels['ID'].astype('str')+'|'+df_labels['Producto']+'|'+df_labels['Marca']+ '|'+df_labels['Empaque']
names = df_labels['name'].tolist()
ids = df_labels['ID'].tolist()
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
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.ButtonGroup([dbc.Button("Prev", id='prev-obj-button'), dbc.Button("Save", id='save-obj-button'), dbc.Button("Next", id='next-obj-button')])
                ]),
                dbc.Row([
                    dbc.Progress(id='obj-progress')
                ]),
                dbc.Row([
                    dcc.Dropdown(options=names, id='label-dropdown')
                ]),
                dbc.Row([
                    html.P("Current label: -", id='label-p')
                ]),
                dbc.Row([
                    dbc.Col(html.Div(id='current-object-div')),
                    dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("About the task"),
                            html.Div(id='obj-info-div')
                        ])
                    )
                ])
                ])
            ])
        ]),
        dbc.Row([
            html.P("Last saved: Never", id='last-saved-p')
        ])
    ]),
    dcc.Store(id='detections-storage'),
    dcc.Store(id='labels-storage'),
    dcc.Store(id='current-img-storage'),
    dcc.Store(id='current-obj-storage'),
    dcc.Store(id='annotations-storage'),
    dcc.Store(id='current-labels-storage')
])

#-------------BACKEND---------------

def resize_strip_to_width(image, width):
    h,w,c = image.shape
    if w > width:
        ratio = width/w
        new_image = cv2.resize(image, (0,0), fx=ratio, fy=ratio)
        return new_image
    else:
        return image

def count_labels(annotations_list, key):
    labels = []
    #print("count: ", annotations_list)
    for d in annotations_list['detections']:
        if key in d:
            labels.append(d[key])
    return labels

def get_summarized_labels(labels_list):
    labels_dict = {}
    for l in labels_list:
        if l in labels_dict:
            labels_dict[l] += 1
        else:
            labels_dict[l] = 1
    return labels_dict

@app.callback(
    Output('image-info-div', 'children', allow_duplicate=True),
    Output('detections-storage', 'data'),
    Output('annotations-storage', 'data', allow_duplicate=True),
    #Output('current-labels-storage', 'data', allow_duplicate=True),
    Input('folder-input', 'value'),
    prevent_initial_call=True
)
def open_folder(folder_addr):
    if folder_addr is not None:
        f = open(folder_addr, 'r')
        detections_list = json.load(f)
        f.close()
        n_images = len(detections_list)
        #labels_list = []

        annotations_path = "/".join(folder_addr.split("/")[:-1])+"/annotations.json"
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                annotations_list = json.load(f)
                print("Annotations files opened!")
                #labels_list = count_labels(annotations_list, "class_name")
        else:
            annotations_list = detections_list
            print("There are no previous annotations")
        
        if n_images>0:
            result = [html.H5("Images in dataset: "), html.P("{}".format(n_images))]
            
            return [result, json.dumps(detections_list), json.dumps(annotations_list)]
        else:
            return ["Images not found.", None, None]
    else:
        return ["Not valid address.", None, None]

@app.callback(
    Output('current-img-div', 'children'),
    Output('image-info-div', 'children', allow_duplicate=True),
    Output('img-progress', 'value'),
    Output('current-img-storage', 'data'),
    Output('current-labels-storage', 'data', allow_duplicate=True),
    Input('next-img-button', 'n_clicks'),
    Input('prev-img-button', 'n_clicks'),
    State('annotations-storage', 'data'),
    prevent_initial_call=True
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

            labels_list = []
            labels_list = count_labels(detections_list[index], "class_name")
            
            # ensamble results
            return [[html.P("Image {}".format(detections_list[index]['image_addr'])),
                    html.Img(src=dataURI)], 
                    [html.H5("Images in dataset: "),
                    html.P("{}".format(len(detections_list))),
                    html.H5("Detected objects: "),
                    html.P("{}".format(len(detections)))],
                    progress_img,
                    json.dumps([index]),
                    json.dumps(labels_list)]
        else:
            return ["No images", "No objects", 0, json.dumps([-1]), None]
    else:
        return ["No images", "No objects", 0, json.dumps([-1]), None]


@app.callback(
    Output('current-object-div', 'children'),
    Output('obj-progress', 'value'),
    Output('current-obj-storage', 'data'),
    Output('label-p', 'children', allow_duplicate=True),
    Output('obj-info-div', 'children'),
    Input('next-obj-button', 'n_clicks'),
    Input('prev-obj-button', 'n_clicks'),
    State('annotations-storage', 'data'),
    State('current-img-storage', 'data'),
    State('current-labels-storage', 'data'),
    prevent_initial_call=True
)
def display_object(n_clicks_next, n_clicks_prev, detections_json, current_img_json, current_labels_json):
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
            #print("Object: ", detection)
            current_label = detection.get('class_name', 'No class selected')

            image = cv2.imread(detections_list[current_img]['image_addr'])
            detected_object = detector.extract_object(image, detection)
            detected_object = resize_strip_to_width(detected_object, 600)


            _, buffer = cv2.imencode('.jpg', detected_object)
            jpg_as_text = base64.b64encode(buffer.tobytes())
            dataURI = 'data:image/jpeg;base64,' + str(jpg_as_text, 'ascii')
            progress_obj = int(index/n_objects*100)

            labels_list = json.loads(current_labels_json)
            results_dict = get_summarized_labels(labels_list)
            if len(results_dict)>0:
                results_table = dash_table.DataTable([{"Producto":k, "Cantidad": v} for k, v in list(results_dict.items())])
            else:
                results_table = "No labels"
            
            # ensamble results
            return [html.Img(src=dataURI, height=600), progress_obj, json.dumps([index]), current_label, results_table]
        else:
            return ["No objects", 0, json.dumps([-1]), "Current label: ---", 'No info']    
    else:
        return ["No objects", 0, json.dumps([-1]), "Current label: ---", 'No info']
    
@app.callback(
    Output('label-p', 'children', allow_duplicate=True),
    Output('annotations-storage', 'data', allow_duplicate=True),
    Output('current-labels-storage', 'data', allow_duplicate=True),
    Input('label-dropdown', 'value'),
    State('annotations-storage', 'data'),
    State('current-img-storage', 'data'),
    State('current-obj-storage', 'data'),
    State('current-labels-storage', 'data'),
    prevent_initial_call=True
)
def label_object(value, annotations_json, current_img_json, current_obj_json, current_labels_json):
    if annotations_json:
        current_img = json.loads(current_img_json)[0]
        current_obj = json.loads(current_obj_json)[0]
        annotations_list = json.loads(annotations_json)
        labels_list = json.loads(current_labels_json)
        if current_img != -1 and current_obj != -1:
            print("Value: ", value)
            if value:
                annotations_list[current_img]['detections'][current_obj]['class_id'] = int(value.split('|')[0])
                annotations_list[current_img]['detections'][current_obj]['class_name'] = value
                labels_list.append(value)
            return ["Current label: {}".format(value), json.dumps(annotations_list), json.dumps(labels_list)]    
        else:
            return ["Current label: --", json.dumps(annotations_list), json.dumps(labels_list)]    
    else:
        return ["Current label: --", None, None]
    
@app.callback(
    Output('last-saved-p', 'children'),
    Input('save-img-button', 'n_clicks'),
    Input('save-obj-button', 'n_clicks'),
    State('annotations-storage', 'data'),
    State('folder-input', 'value')
)
def save_annotations(n_clicks_img, n_clicks_obj, annotations_json, folder_addr):
    if annotations_json:
        annotations_list = json.loads(annotations_json)
        n_img = n_clicks_img if n_clicks_img else 0
        n_obj = n_clicks_obj if n_clicks_obj else 0
        if n_img>0 or n_obj>0:
            annotations_path = "/".join(folder_addr.split("/")[:-1])+"/annotations.json"
            with open(annotations_path, "w") as f:
                json.dump(annotations_list, f, indent=4)
            last_saved = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            return "Last saved: {}".format(last_saved)
        else:
            return "Last saved: Never"
    else:
        return "Last saved: Never"

# TODO: Add a table with all the labeled objects in the annotation file. For this I need to add a storge for the labeld objects
if __name__ == '__main__':
    app.run(debug=True)
