import cv2
from ultralytics import YOLO
import math

# --- 1. AYARLAR ---
# Hazır eğitilmiş YOLOv8 nano modelini indirip yükledim.
# İlk çalıştırmada internetten otomatik indirecek (yaklaşık 6MB).
model = YOLO("yolov8n.pt")

# Sadece "cell phone" sınıfını tespit etmek istiyorum.
# COCO veri setinde 'cell phone' sınıfının ID'si 67'dir.
TARGET_CLASS_ID = 67 

# --- 2. KAMERAYI AÇ ---
cap = cv2.VideoCapture(0)
# Kamera çözünürlüğünü düşürerek hızı artırdım. 
cap.set(3, 640)
cap.set(4, 480)

print("Telefon Takip Sistemi Başlatıldı... Çıkış için 'q' basınız.")

while True:
    success, img = cap.read()
    if not success:
        print("Kamera okunamadı.")
        break

    # --- 3. NESNE TESPİTİ (YOLO) ---
    # stream=True: Video akışı için daha verimli çalışır.
    results = model(img, stream=True, verbose=False)

    phone_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Sınıf ID'sini al
            cls = int(box.cls[0])
            
            # Eğer tespit edilen nesne 'cell phone' ise
            if cls == TARGET_CLASS_ID:
                phone_detected = True
                
                # Koordinatları al
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Güven skorunu al (Ne kadar emin?)
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Kutuyu çiz (Kırmızı)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Etiketi yaz
                label = f"Telefon Yasak! ({conf})"
                cv2.putText(img, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Eğer telefon varsa ekrana genel uyarı bas demektir.
    if phone_detected:
        cv2.putText(img, "DIKKAT: TELEFON KULLANIMI TESPIT EDILDI!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- 4. GÖSTER ---
    cv2.imshow('Telefon Tespiti', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()