# Real-Time Object Detection

A lightweight Python application that performs **real-time object detection** from a live camera feed using [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics) and OpenCV. Detected objects are drawn with bounding boxes, class labels, confidence scores, and an on-screen FPS counter.

---

## Author

**Gresa Bruqi**

---

## Features

- Live webcam inference with YOLOv8 nano (`yolov8n.pt`)
- Configurable confidence threshold
- Per-class color-coded bounding boxes and readable labels (with background contrast)
- Smoothed FPS overlay for performance monitoring
- GPU acceleration when CUDA is available (automatic fallback to CPU)
- Simple controls: press **Q** to quit

---

## Requirements

- Python 3.8 or newer
- Webcam or compatible video capture device
- (Optional) NVIDIA GPU with CUDA for faster inference

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <your-repository-url>
   cd RealTimeObjectDetection
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv venv
   ```

   Windows:

   ```bash
   venv\Scripts\activate
   ```

   macOS / Linux:

   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   On first run, Ultralytics will automatically download the `yolov8n.pt` weights if they are not already present.

---

## Usage

Start the detection pipeline:

```bash
python main.py
```

A window titled **YOLOv8 Real-time Object Detection** will open. Press **Q** to exit cleanly.

---

## Configuration

Edit the constants at the top of `main.py`:

| Parameter        | Default        | Description                                      |
|------------------|----------------|--------------------------------------------------|
| `SOURCE`         | `0`            | Video source index (`0` = default webcam)        |
| `MODEL_WEIGHTS`| `"yolov8n.pt"` | YOLOv8 model weights file                        |
| `CONF_THRESHOLD` | `0.2`          | Minimum detection confidence (0.0–1.0)           |

To use a video file instead of a camera, set `SOURCE` to the file path (e.g. `"sample.mp4"`).

---

## Project Structure

```
RealTimeObjectDetection/
├── main.py              # Application entry point and detection loop
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Dependencies

| Package          | Role                                      |
|------------------|-------------------------------------------|
| `ultralytics`    | YOLOv8 model loading and inference        |
| `opencv-python`  | Video capture, drawing, and display       |

---

## License

This project is provided as-is for educational and personal use. Specify a license file if you intend to distribute or open-source the repository publicly.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
