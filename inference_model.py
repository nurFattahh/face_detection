import torch
import cv2
import time

# load dataset
MODEL_PATH = 'best_windows.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

class_names = ['Fattah']

cap = cv2.VideoCapture(0)
prev_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Detection
    results = model(frame)
    detections = results.xyxy[0]  

    # Filter hanya yang confident > 0.3
    detections = [det for det in detections if det[4] >= 0.3]
    if detections:
        best = max(detections, key=lambda x: x[4])  # ambil deteksi dengan confidence tertinggi

        x1, y1, x2, y2 = map(int, best[:4])
        conf = float(best[4])
        cls = int(best[5])

        label = f'{class_names[cls]} {conf:.2f}'
        color = (0, 255, 0)  # hijau untuk Fattah

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Action jika terdeteksi fattah


    # Show FPS
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow("Deteksi Fattah", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
