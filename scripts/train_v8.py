from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
results = model.train(
   data=r"C:\Users\Vidya\Downloads\pothole_dataset_v8\pothole_dataset_v8\pothole.yaml",
   imgsz=1280,
   epochs=1,
   batch=1,
   name='yolov8n_v8_50e')

 """  C:\Users\Vidya\Downloads\runs\detect\yolov8n_v8_50e8\weights\best.pt """