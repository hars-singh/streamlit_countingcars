import os
import glob
import numpy as np
import pandas as pd
import torch
import os
import random
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
%matplotlib inline
import wandb
import cv2
import itertools
import copy


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
import xmltodict
import json

@st.cache(allow_output_mutation=True)
def load_model():
  gdown.download(id =_0Y9F3dlGxAtOaD4Xa3y6mz45GvSEe18,output = 'model_final.pth', quiet=False)
  gdown.download(id = 1-0rskMHiZwWT19IJxY0CnL_X5BtKmW8Q, output = "faster_rcnn_X_101_32x8d_FPN_3x.yaml", quiet=False)
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file( "faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
  cfg.MODEL.WEIGHTS = os.path.join( "model_final.pth")
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
  predictor = DefaultPredictor(cfg)
  return predictor
with st.spinner('Model is being loaded..'):
  predictor=load_model()

import warnings
classes = ['car']
# Ignore DeprecationWarning warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

font = cv2.FONT_HERSHEY_SIMPLEX     
fontScale = 0.3
color = (255, 0, 0)
thickness = 1

colors = {
    0:(255,0,0),
    1:(0,255,255),
    2:(0,0,255),
    3:(255,255,255),
    4:(255,255,0)
}
inf_path = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

img = cv2.imread(inf_path)
outputs = model(img)
out = outputs["instances"].to("cpu")
scores = out.get_fields()['scores'].numpy()
boxes = out.get_fields()['pred_boxes'].tensor.numpy().astype(int)
labels= out.get_fields()['pred_classes'].numpy()
boxes = boxes.astype(int)
boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
im /= 255.0

for b,s,l in zip(boxes,scores,labels):
    cv2.rectangle(im, (b[0],b[1]), (b[0]+b[2],b[1]+b[3]), colors[l], thickness)
    cv2.putText(im, '{}'.format(classes[l]), (b[0],b[1]-3), font, fontScale, colors[l], thickness)
        
# plt.figure(figsize=(12,12))
# plt.imshow(im)
st.image(im)
# st.write('Writing pred_classes/pred_boxes output')
# st.write(outputs["instances"].pred_classes)
# st.write(outputs["instances"].pred_boxes)

# st.write('Using Vizualizer to draw the predictions on Image')
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# st.image(out.get_image()[:, :, ::-1])
