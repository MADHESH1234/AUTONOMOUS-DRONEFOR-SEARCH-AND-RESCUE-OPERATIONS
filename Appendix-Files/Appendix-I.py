from inference import get_model
import supervision as sv
import cv2
import numpy as np
from imutils.video import FPS
import imutils

use_gpu = True
live_video = False
confidence_level = 0.5

MODEL_ID = "YOUR_ROBOFLOW_MODEL_ID/VERSION"
API_KEY = "YOUR_ROBOFLOW_API_KEY"
CAFFE_PROTOTXT = "PATH/TO/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL = "PATH/TO/MobileNetSSD_deploy.caffemodel"
VIDEO_SOURCE_PATH = "PATH/TO/YOUR/LOCAL/test_video.mp4" 

fps = FPS().start()

model = get_model(model_id=MODEL_ID, api_key=API_KEY)

CLASSES = ["Human"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTXT, CAFFE_MODEL)

if use_gpu:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print("[INFO] accessing video stream...")
video_source = 0 if live_video else VIDEO_SOURCE_PATH
cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open video source. Check if the path is correct or if the camera is connected.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = int(cap.get(cv2.CAP_PROP_FPS))

out = None
if not live_video:
    out = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_video,
        (frame_width, frame_height)
    )

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error in capturing frame.")
        break
        
    frame = imutils.resize(frame, width=1000)
    (h, w) = frame.shape[:2]
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        results = model.infer(rgb_frame)[0]
        detections = sv.Detections.from_inference(results)
        
        annotated_frame = bounding_box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
    except Exception as e:
        print(f"Error during YOLO/Roboflow inference: {e}. Skipping this model's annotations.")
        annotated_frame = frame.copy()


    
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections_ssd = net.forward()
    
    for i in np.arange(0, detections_ssd.shape[2]):
        confidence = detections_ssd[0, 0, i, 2]
        
        if confidence > confidence_level:
            ssd_class_index = int(detections_ssd[0, 0, i, 1])
            
            idx = 0 
            
            box = detections_ssd[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label_text = f"SSD Detection: {CLASSES[idx]}: {confidence * 100:.2f}%"
            
            cv2.rectangle(annotated_frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(annotated_frame, label_text, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)
            
    
    if not live_video and out is not None:
        resized_out_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
        out.write(resized_out_frame)

    cv2.imshow("Live Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

fps.stop()
print(f"[INFO] elapsed time: {fps.elapsed():.2f}")
print(f"[INFO] approx. FPS: {fps.fps():.2f}")

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
