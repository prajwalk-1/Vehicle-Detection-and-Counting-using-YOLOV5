# YOLOv5 Vehicle Detection and Counting

This project implements real-time vehicle detection and counting using the YOLOv5 model. The script captures video from a CCTV feed, performs vehicle detection, counts vehicles of specific classes (bike, car, auto, bus), and logs the counts along with the timestamp to a CSV file. Additionally, it saves frames with detected vehicles as images.

![vehicle_track](https://github.com/user-attachments/assets/c1619782-359d-44eb-aaa3-8ef3cd5d5ee9)

![WhatsApp Image 2024-07-31 at 12 52 49 PM](https://github.com/user-attachments/assets/44c2b9df-7ccc-4e04-ad2a-984d1a68b9f8)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
  - [Device Selection](#device-selection)
  - [Model Loading](#model-loading)
  - [Video Stream Initialization](#video-stream-initialization)
  - [Detection and Counting](#detection-and-counting)
  - [Results Saving](#results-saving)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prajwalk-1/Vehicle-Detection-and-Counting-using-YOLOV5.git
   cd yolov5-vehicle-detection
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv5 model:**
   Download the YOLOv5 model file (e.g., `yolov5s.pt`) from the [YOLOv5 release page](https://github.com/ultralytics/yolov5/releases) and place it in the project directory.

## Usage

Run the script:
```bash
python vehicle_detection.py
```

Make sure to update the script with the correct RTSP URL, username, and password for your CCTV feed.

## Code Explanation

### Device Selection

The script begins by selecting the appropriate device (CPU or GPU) based on the availability of CUDA.

```python
def get_device():
    if torch.cuda.is_available():
        print("CUDA device is available. Using GPU.")
        return 'cuda:0'
    else:
        print("CUDA device is not available. Using CPU.")
        return 'cpu'
```

### Model Loading

The YOLOv5 model is loaded with the specified device. The `force_reload=True` parameter ensures that the model is reloaded to avoid cache issues.

```python
def load_model(model_path, device):
    print("Loading model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device=device, force_reload=True)
    print("Model loaded successfully.")
    return model
```

### Video Stream Initialization

The script initializes video capture from a CCTV feed using OpenCV. It constructs the RTSP URL with the provided username, password, IP address, port, and channel.

```python
stream_url = f"rtsp://{username}:{password}@{ip}:{port}/Streaming/Channels/{channel}"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream. Check RTSP URL and camera settings.")
    exit()
```

### Detection and Counting

The script processes each frame from the video stream, performs vehicle detection using the YOLOv5 model, and counts vehicles of specific classes (bike, car, auto, bus). It maintains a dictionary of tracked objects to avoid counting the same vehicle multiple times.

```python
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.pandas().xyxy[0]

    current_counters = {3: 0, 2: 0, 4: 0, 5: 0}
    for _, row in detections.iterrows():
        class_id = int(row['class'])
        if class_id in classes_of_interest:
            detection_id = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            if detection_id not in tracked_objects:
                current_counters[class_id] += 1
                tracked_objects[detection_id] = class_id
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{model.names[class_id]}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for key in counter:
        counter[key] = current_counters[key]

    info = ' | '.join([f"{class_names[class_id]}: {count}" for class_id, count in counter.items()])
    cv2.putText(frame, info, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('YOLOv5 Live Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Results Saving

The script saves the vehicle counts and the frame with detected vehicles to a CSV file and as images every 10 seconds.

```python
elapsed_time = time.time() - last_save_time
if elapsed_time >= image_save_interval:
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    image_filename = f'image_{timestamp}.jpg'
    image_path = os.path.join(directory, image_filename)
    cv2.imwrite(image_path, frame)

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

    last_save_time = time.time()
```

## Dependencies

- Python 3.10
- OpenCV
- PyTorch
- pandas

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
