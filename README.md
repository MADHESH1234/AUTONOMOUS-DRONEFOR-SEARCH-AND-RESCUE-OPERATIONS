# ✈️ Autonomous Drone for Search and Rescue Operations
This project presents an autonomous drone system designed to enhance the effectiveness of Search and Rescue (SAR) missions during natural and human-made disasters. The system utilizes Machine Learning (ML) and Computer Vision to autonomously detect and locate people in distress based on visual and thermal patterns. Equipped with high-definition cameras, sensors, and GPS, it performs disaster area assessments, sends real-time locations to rescue teams, and can deliver critical supplies.

## ✨ Key Features
- **Autonomous Navigation:** The drone leverages GPS and IMU (Inertial Measurement Unit) for self-guided, precision flight, allowing it to navigate complex and hazardous environments, including difficult terrains and harsh weather conditions, with minimal human oversight.
- **AI-Powered Human Detection:** The core system uses a specialized YOLOv5 NAS model trained on a disaster-specific dataset.
- **Performance Metrics:** Achieves a mAP of 96.8%, Precision of 95.8%, and Recall of 96.4% in human identification.
- **Real-Time Data Transmission:** Pinpoints the exact location of detected victims using GPS and transmits this data, along with live video streams and status reports, to a central command center over a secure, low-latency network.
- **Critical Payload Delivery:** Features a Rescue Assistance Mode to autonomously deliver essential supplies (emergency medical kits, food, water) directly to victims.
- **Scalability:** The system is flexible and suitable for various disaster scenarios, including fires, floods, and earthquakes.

## 🛠️ Technical Specifications
The prototype is built on an F450 frame configuration and includes a Pixhawk flight controller for autonomy.

| Metric              | Specification                                |
|----------------------|----------------------------------------------|
| Detection Model      | YOLOv5 NAS (Custom Trained)                 |
| Detection Accuracy   | mAP: 96.8%                                  |
| Frame                | F450/Q450 Quadcopter Frame                  |
| Flight Controller    | Pixhawk 2.4.8 (PX4 32 Bit Autopilot)        |
| Battery              | DOGCOM 4S 3300mAh 80C 14.8V LiPo            |
| Estimated Flight Time| ~26.56 minutes                              |
| Total Weight (w/ Payload) | ~2.5 kg (1 kg payload capacity)        |
| Telemetry            | 433Mhz 100mW Radio Telemetry                |

## 🚀 Getting Started
This project involves hardware, flight control via MAVLink (using pymavlink), and a computer vision processing pipeline.

### Prerequisites
You will need Python and the libraries listed in the requirements.txt.

- Python 3.x  
- OpenCV (cv2)  
- imutils  
- supervision  
- pymavlink  

### Installation
Clone the repository:
```bash
git clone [https://github.com/MADHESH1234/AUTONOMOUS-DRONEFOR-SEARCH-AND-RESCUE-OPERATIONS.git]
```

Install Python dependencies:
```bash
pip install opencv-python imutils python-supervision pymavlink
```

### Running the System
The project is divided into two primary operational components:

1. **Detection Module (appendix_I.py):** Handles camera input, runs the YOLOv5 NAS model, and annotates the video stream.  
2. **Flight Control Module (appendix_II.py):** Manages MAVLink connection, handles arming, takeoff, waypoint setting (simulated circular path), and RTL (Return to Launch) functions.

To run the core detection pipeline:
```bash
# Ensure you have your trained model and configuration files configured correctly within the script paths ie.appedix_I
python appendix_I.py
```

To operate the drone (requires connected hardware/simulator):
```bash
# Modify appendix_II.py to match your flight controller's connection string (e.g., 'COM18', baud=57600)
python appendix_II.py
```

## 📂 Project Files
| File              | Description                                                           |
|-------------------|-----------------------------------------------------------------------|
| PHASE 2 REPORT.pdf| The complete project report detailing design, hardware, methodology, and ML result analysis. |
| appendix_I.py     | Python code for the Machine Learning Detection module (CV processing, annotation). |
| appendix_II.py    | Python code for the Autonomous Flight Navigation and MAVLink communication layer. |

## 🧑‍💻 Team
This project was submitted in partial fulfillment for the award of the degree of Bachelor of Engineering in Robotics and Automation.

- **M. MADHESHWARAN (2116211601027)**  
