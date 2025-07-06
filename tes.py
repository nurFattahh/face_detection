import torch
import cv2
import os

# Pastikan file model ada di direktori yang sama dengan script ini
MODEL_PATH = 'best_windows.pt'  # Ganti dengan nama file model Anda (best.pt atau best_windows.pt)

# Inisialisasi model YOLOv5
def load_model():
    try:
        # Method 1: Menggunakan torch.hub
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        
        # Atau Method 2: Menggunakan ultralytics (uncomment jika method 1 error)
        # from ultralytics import YOLO
        # model = YOLO(MODEL_PATH)
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Pastikan:")
        print(f"1. File model {MODEL_PATH} ada di folder ini")
        print("2. Anda sudah install requirements: pip install torch torchvision opencv-python ultralytics")
        return None

# Fungsi utama untuk deteksi real-time
def main():
    # Load model
    model = load_model()
    if model is None:
        return

    # Inisialisasi webcam
    cap = cv2.VideoCapture(0)  # Ganti dengan path video jika ingin deteksi dari file
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek
        results = model(frame)  # Untuk torch.hub
        # results = model.predict(frame)  # Untuk ultralytics YOLO

        # Visualisasi hasil deteksi
        if hasattr(results, 'xyxy'):  # Untuk torch.hub format
            detections = results.xyxy[0]
            for *box, conf, cls in detections:
                x1, y1, x2, y2 = map(int, box)
                label = f'Fattah {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:  # Untuk ultralytics YOLO format
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = f'Fattah {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Tampilkan hasil
        cv2.imshow('Deteksi Fattah', frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Verifikasi file model ada
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: File model '{MODEL_PATH}' tidak ditemukan!")
        print("Pastikan:")
        print(f"1. File model ada di folder: {os.getcwd()}")
        print("2. Nama file sesuai (perhatikan huruf besar/kecil)")
    else:
        print(f"Model ditemukan: {os.path.abspath(MODEL_PATH)}")
        main()