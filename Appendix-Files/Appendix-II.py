from inference import get_model
import supervision as sv
import cv2
import numpy as np
from imutils.video import FPS
import imutils
from pymavlink import mavutil 
import time 
import math 
import random 

use_gpu = True
confidence_level = 0.5

MODEL_ID = "YOUR_ROBOFLOW_MODEL_ID/VERSION"
API_KEY = "YOUR_ROBOFLOW_API_KEY"

CAFFE_PROTOTXT = "PATH/TO/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL = "PATH/TO/MobileNetSSD_deploy.caffemodel"

MAVLINK_PORT = 'COM18' 
MAVLINK_BAUD = 57600
RADIUS = 4 
ALTITUDE = 2  
CIRCLE_POINTS = 36  

def set_mode(master, mode): 
    mode_id = master.mode_mapping().get(mode) 
    if mode_id is None: 
        print(f"Mode {mode} not available") 
        return 
    master.mav.set_mode_send( 
        master.target_system, 
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 
        mode_id 
    ) 
    print(f"Mode changed to {mode}") 
    time.sleep(2) 

def arm_and_takeoff(master, target_altitude): 
    print("Arming motors...") 
    master.mav.command_long_send( 
        master.target_system, 
        master.target_component, 
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 
        1, 0, 0, 0, 0, 0, 0 
    ) 
    while True: 
        msg = master.recv_match(type='HEARTBEAT', blocking=True) 
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED 
        if armed: 
            print("Drone armed!") 
            break 
        time.sleep(1) 
    print (f"Taking off to {target_altitude} meters...") 
    master.mav.command_long_send( 
        master.target_system, 
        master.target_component, 
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 
        0, 
        0, 0, 0, 0, 0, 0, target_altitude 
    ) 
    while True: 
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True) 
        if msg and msg.relative_alt >= target_altitude * 1000 * 0.95: 
            print("Takeoff successful! Reached target altitude.") 
            break 
        time.sleep(1) 

def get_current_location(master): 
    while True: 
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True) 
        if msg: 
            lat = msg.lat / 1e7 
            lon = msg.lon / 1e7 
            return lat, lon 

def send_location(lat, lon): 
    print(f"Sending location - Latitude: {lat}, Longitude: {lon}") 

def return_to_launch(master): 
    print ("Returning to launch (RTL)...") 
    master.mav.command_long_send( 
        master.target_system, 
        master.target_component, 
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, 
        0, 0, 0, 0, 0, 0, 0, 0
    ) 
    print("Set_Navigation='HOME'")


def initialize_detection_models():
    global model, net, CLASSES, COLORS
    
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)
    CLASSES = ["Human"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromCaffe(CAFFE_PROTOTXT, CAFFE_MODEL)

    if use_gpu:
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    return model, net

def scan_for_human(master, frame, model, net, confidence_level): 
    
    h, w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_frame = frame.copy() 

    try:
        results = model.infer(rgb_frame)[0]
        detections = sv.Detections.from_inference(results)
        
        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
        
    except Exception as e:
        print(f"Error during YOLO/Roboflow inference: {e}. Skipping this model's annotations.")
        
    
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections_ssd = net.forward()
    
    human_detected_flag = False
    
    for i in np.arange(0, detections_ssd.shape[2]):
        confidence = detections_ssd[0, 0, i, 2]
        
        if confidence > confidence_level:
            human_detected_flag = True 
            
            idx = 0 
            
            box = detections_ssd[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label_text = f"SSD Detection: {CLASSES[idx]}: {confidence * 100:.2f}%"
            
            cv2.rectangle(annotated_frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(annotated_frame, label_text, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)
    
    if human_detected_flag or (detections.xyxy.shape[0] > 0 if 'detections' in locals() else False):
        print("[ALERT] Human detected!")
        lat, lon = get_current_location(master)
        send_location(lat, lon)
        
    return human_detected_flag, annotated_frame


def fly_in_circle(master, radius, altitude, points=36): 
    print ("Flying in a circular path and scanning for humans...") 
    
    lat0, lon0 = get_current_location(master) 
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("Error: Could not open live video source (index 0). Cannot proceed with detection.")
        return

    fps = FPS().start()
    
    detection_model, ssd_net = initialize_detection_models()

    for i in range(points): 
        angle = (2 * math.pi / points) * i 
        dx = radius * math.cos(angle) 
        dy = radius * math.sin(angle) 
        dlat = dx / 111320 
        dlon = dy / (111320 * math.cos(math.radians(lat0))) 
        waypoint_lat = lat0 + dlat 
        waypoint_lon = lon0 + dlon 
        
        master.mav.set_position_target_global_int_send( 
            int(time.time()), 
            master.target_system, 
            master.target_component, 
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            int(0b110111111000), 
            int(waypoint_lat * 1e7), 
            int(waypoint_lon * 1e7), 
            altitude, 0, 0, 0, 
            0, 0, 0, 
            0, 0 
        ) 
        print(f"Moving to waypoint {i+1}/{points} at ({waypoint_lat:.6f}, {waypoint_lon:.6f})") 
        time.sleep(2)  
        
        ret, frame = cap.read() 
        if ret: 
            frame = imutils.resize(frame, width=1000)
            
            human_detected, annotated_frame = scan_for_human(
                master, frame, detection_model, ssd_net, confidence_level
            )

            cv2.imshow('Live Detection and Tracking', annotated_frame) 
            
            if human_detected:
                print("Human detected during flight. Returning to launch...") 
                cap.release() 
                cv2.destroyAllWindows() 
                return_to_launch(master) 
                return 

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
        
        fps.update()


    print ("Circular path completed.") 
    fps.stop()
    print(f"[INFO] elapsed time: {fps.elapsed():.2f}")
    print(f"[INFO] approx. FPS: {fps.fps():.2f}")
    cap.release() 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    try:
        master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD) 
        print ("Waiting for heartbeat...") 
        master.wait_heartbeat() 
        print ("Heartbeat received! System is alive.") 

        set_mode(master, 'GUIDED')
        arm_and_takeoff(master, ALTITUDE)
        
        fly_in_circle(master, RADIUS, ALTITUDE, CIRCLE_POINTS)
        
        return_to_launch(master)

    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")
        print("Please ensure your drone is connected via telemetry and the MAVLink port/baud are correct.")
    out.release()
cap.release()