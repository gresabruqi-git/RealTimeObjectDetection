

import time
from typing import Tuple

import cv2
from ultralytics import YOLO



SOURCE = 0  

MODEL_WEIGHTS = "yolov8n.pt"

CONF_THRESHOLD = 0.2

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
BOX_THICKNESS = 2
FPS_SMOOTHING = 0.9  

def get_color(index: int) -> Tuple[int, int, int]:
    
   
    palette = [
        (255, 0, 0),       
        (0, 255, 0),      # 
        (0, 0, 255),     
        (255, 255, 0),    
        (255, 0, 255),    
        (0, 255, 255),   
        (128, 0, 128),    
        (0, 128, 255),    
        (128, 128, 0),   
        (0, 128, 128),    
    ]
    return palette[index % len(palette)]


def draw_label_with_bg(
    frame,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    scale: float = TEXT_SCALE,
    thickness: int = TEXT_THICKNESS,
):
    
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
   
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 4),
        (x + text_width + 4, y),
        color,
        thickness=cv2.FILLED,
    )
   
    brightness = int(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    cv2.putText(frame, text, (x + 2, y - 4), FONT, scale, text_color, thickness, cv2.LINE_AA)


def main():
   
    model = YOLO(MODEL_WEIGHTS)


    try:
        if model.device.type == "cuda":
            model.to(dtype=model.model.dtype)  
    except Exception:
        pass


    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {SOURCE}")


   
    class_names = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else model.names

   

    last_time = time.time()
    fps = 0.0

    window_name = "YOLOv8 Real-time Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run frame e bon
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        res = results[0]

    
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            confidences = res.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confidences):
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                color = get_color(cls_id)
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, BOX_THICKNESS)

#meret emri klases
                class_name = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else (
                    class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
                )
                label = f"{class_name} {conf:.2f}"
                draw_label_with_bg(frame, label, x1i, y1i, color)


        now = time.time()
        inst_fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now
        fps = FPS_SMOOTHING * fps + (1.0 - FPS_SMOOTHING) * inst_fps
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 28),
            FONT,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (12, 28),
            FONT,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



