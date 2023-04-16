from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2

model = YOLO(r"C:\Users\Vidya\Downloads\runs\detect\yolov8n_v8_50e8\weights\best.pt")
model.predict(source=r"C:\Users\Vidya\Downloads\istockphoto-1319663198-640_adpp_is.mp4", classes= 0, save = True)
