import cv2
import torch
import pandas as pd
import time
import os

# Function to select the appropriate device
def get_device():
    if torch.cuda.is_available():
        print("CUDA device is available. Using GPU.")
        return 'cuda:0'  # Explicitly specify the first CUDA device
    else:
        print("CUDA device is not available. Using CPU.")
        return 'cpu'

# Load YOLOv5 model with force_reload to ensure cache issues are resolved
def load_model(model_path, device):
    print("Loading model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device, force_reload=True)
    print("Model loaded successfully.")
    return model

# Initialize device and load model
device = get_device()
model_path = 'yolov5s.pt'  # Path to your YOLOv5 model
model = load_model(model_path, device)

# Initialize video capture from CCTV feed
username = 'username'
password = 'password'
ip = '192.168.1.225'
port = '554'
channel = '101'

stream_url = f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}"
cap = cv2.VideoCapture(stream_url)

# Check if the video stream was opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream. Check RTSP URL and camera settings.")
    exit()

print("Video stream opened successfully.")

# Define classes to detect (including 'auto' with class ID 4)
classes_of_interest = [3, 2, 4, 5]  # bike: 1, car: 2, auto: 4, bus: 5
class_names = {3: 'Bike', 2: 'Car', 4: 'Auto', 5: 'Bus'}

# Define CSV file path and ensure the directory exists
csv_file = 'vehicle_counts.csv'
directory = os.path.dirname(csv_file)
if not os.path.exists(directory) and directory:
    os.makedirs(directory)

# Initialize CSV file
if not os.path.isfile(csv_file):
    df = pd.DataFrame(columns=['Time', 'Bike', 'Car', 'Auto', 'Bus', 'Image Path'])
else:
    df = pd.read_csv(csv_file)

# Track time and initialize counters
start_time = time.time()
last_save_time = start_time
counter = {3: 0, 2: 0, 4: 0, 5: 0}
tracked_objects = {}  # To keep track of objects and their IDs

# Initialize image capture
image_save_interval = 10  # Time in seconds
image_counter = 0  # To create unique image filenames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        continue

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(rgb_frame)

    # Get detections
    detections = results.pandas().xyxy[0]

    # Initialize current frame counters
    current_counters = {3: 0, 2: 0, 4: 0, 5: 0}

    # Process detections
    for _, row in detections.iterrows():
        class_id = int(row['class'])
        if class_id in classes_of_interest:
            # Create a unique ID for each detected vehicle
            detection_id = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            
            if detection_id not in tracked_objects:
                # Increment the count for this class
                current_counters[class_id] += 1
                # Add to tracked objects
                tracked_objects[detection_id] = class_id

                # Draw bounding boxes on the frame
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{model.names[class_id]}"
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update global counters
    for key in counter:
        counter[key] = current_counters[key]

    # Display counters on the frame
    info = ' | '.join([f"{class_names[class_id]}: {count}" for class_id, count in counter.items()])
    cv2.putText(frame, info, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Live Detection', frame)

    # Save counts and image every 10 seconds
    elapsed_time = time.time() - last_save_time
    if elapsed_time >= image_save_interval:
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        image_filename = f'image_{timestamp}.jpg'
        image_path = os.path.join(directory, image_filename)
        
        # Save the frame as an image
        cv2.imwrite(image_path, frame)

        # Save counts to CSV
        new_row = pd.DataFrame([{
            'Time': timestamp,
            'Bike': counter[3],
            'Car': counter[2],
            'Auto': counter[4],
            'Bus': counter[5],
            'Image Path': image_path
        }])
        try:
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(csv_file, index=False)
        except PermissionError as e:
            print(f"Error writing to CSV file: {e}")

        # Reset timer
        last_save_time = time.time()

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
