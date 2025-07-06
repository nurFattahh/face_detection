import torch
import cv2
import time

# ========== Load model hasil training ==========
MODEL_PATH = 'best_windows.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# ========== Kelas sesuai data.yaml ==========
class_names = ['Fattah']

# ========== Baca gambar ==========
image_path = 'foto2.jpg'  # Ganti dengan file gambar kamu
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

# ========== Resize proporsional ==========
max_width = 640
h, w = image.shape[:2]
scale = max_width / w
image = cv2.resize(image, (int(w * scale), int(h * scale)))

start_time = time.time()

# ========== Deteksi ==========
results = model(image)
detections = results.xyxy[0]

# ========== Filter confidence > 0.3 ==========
detections = [det for det in detections if det[4] >= 0.3]
if detections:
    best = max(detections, key=lambda x: x[4])  # Deteksi dengan confidence tertinggi

    x1, y1, x2, y2 = map(int, best[:4])
    conf = float(best[4])
    cls = int(best[5])

    label = f'{class_names[cls]} {conf:.2f}'
    color = (0, 255, 0)

    # Bounding box & label
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    print(f"Deteksi: {label}")
else:
    print("Tidak ada deteksi dengan confidence > 0.3")

# ========== Tampilkan gambar dengan FPS ==========

cv2.imshow("Hasil Deteksi Gambar", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
