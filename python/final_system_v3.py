import cv2
import mediapipe as mp
import numpy as np
import time
import math
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os
import threading
from playsound import playsound
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- 1. AYARLAR VE SABİTLER ---
EAR_THRESHOLD = 0.22  # Uyuklama eşiği
EAR_FRAMES = 15       # Kaç kare kapalı kalırsa alarm çalsın?
YOLO_SKIP_FRAMES = 5  # YOLO her 5 karede bir çalışsın (Hız için)
ALARM_FILE = "alarm.mp3" # Klasördeki ses dosyasının adı

# Askeri Tema Renkleri (BGR Formatında)
COLOR_GREEN = (0, 255, 0)    # Normal durum
COLOR_RED = (0, 0, 255)      # Kritik/Alarm
COLOR_ORANGE = (0, 165, 255) # Uyarı (Telefon)
COLOR_HUD_BG = (20, 20, 20)  # HUD Arka plan (Koyu Gri)

# Grafik Ayarları
EAR_HISTORY_LEN = 100 # Grafikte gösterilecek son kaç veri?
ear_history = [0.3] * EAR_HISTORY_LEN # Başlangıç verisi

# Değişkenler
log_data = []
blink_counter = 0
frame_count = 0
phone_detected_buffer = False
alarm_on = False

# Modelleri Yükle
mp_face_mesh = mp.solutions.face_mesh
print("Yapay Zeka (YOLO) Modeli Yükleniyor...")
model = YOLO("yolov8n.pt")
TARGET_CLASS_ID = 67  # Cell phone

# --- 2. FONKSİYONLAR ---
def play_alarm_sound():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        try:
            playsound(ALARM_FILE)
        except Exception as e:
            print(f"Ses hatası: {e}")
        alarm_on = False

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))
    p2_p6 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    p3_p5 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    p1_p4 = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def create_live_plot(data_history):
    """Matplotlib ile EAR grafiğini çizip OpenCV görüntüsüne dönüştürür (GÜNCELLENDİ)"""
    fig, ax = plt.subplots(figsize=(4, 2), dpi=80) # Grafik boyutu
    
    # Arka planı şeffaf/uyumlu yap
    fig.patch.set_facecolor('#141414') # Koyu gri arka plan
    ax.set_facecolor('#141414')
    
    # Grafiği çiz
    ax.plot(data_history, color='#00FF00', linewidth=2) # Yeşil çizgi
    
    # Eksen ayarları
    ax.set_ylim(0, 0.5) # EAR genelde 0.0 - 0.4 arasındadır
    ax.set_title("CANLI YORGUNLUK ANALIZI (EAR)", color='white', fontsize=10)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Çizimi görüntüye çevir
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # --- DÜZELTME BURADA ---
    # tostring_rgb yerine buffer_rgba kullanıyoruz
    buf = canvas.buffer_rgba()
    width, height = canvas.get_width_height()
    image_plot = np.frombuffer(buf, dtype='uint8').reshape(int(height), int(width), 4) # RGBA (4 kanal)
    
    # RGBA'dan BGR'ye çevir (OpenCV için)
    image_plot = cv2.cvtColor(image_plot, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig) # Belleği temizle
    return image_plot

def draw_hud(image, status_text, status_color, ear_val):
    h, w, _ = image.shape
    
    # 1. Üst Bilgi Paneli (Yarı saydam)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # 2. Alt Bilgi Paneli (Yarı saydam)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h-40), (w, h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # Tarih ve Saat
    dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    cv2.putText(image, f"OPERASYON: {dt_string}", (20, 30), 
                cv2.FONT_HERSHEY_PLAIN, 1.2, COLOR_GREEN, 1)
    
    # Sistem Durumu (Büyük ve Renkli)
    cv2.putText(image, f"DURUM: {status_text}", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Sağ Üst: EAR Değeri
    cv2.putText(image, f"EAR: {ear_val:.2f}", (w - 180, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
    
    # Alt Panel: Sistem Bilgisi
    cv2.putText(image, "SISTEM AKTIF - KAYIT ALINIYOR...", (w - 350, h - 15), 
                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)
    
    # 3. Nişangah (Kameranın ortasına)
    cx, cy = w // 2, h // 2
    length = 20
    gap = 10
    cv2.line(image, (cx - length - gap, cy), (cx - gap, cy), COLOR_GREEN, 1)
    cv2.line(image, (cx + gap, cy), (cx + length + gap, cy), COLOR_GREEN, 1)
    cv2.line(image, (cx, cy - length - gap), (cx, cy - gap), COLOR_GREEN, 1)
    cv2.line(image, (cx, cy + gap), (cx, cy + length + gap), COLOR_GREEN, 1)
    cv2.circle(image, (cx, cy), 2, COLOR_RED, -1)

def log_incident(type_name):
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    if log_data and log_data[-1]['Saat'] == time_str and log_data[-1]['Ihlal_Turu'] == type_name:
        return
    log_data.append({'Tarih': now.strftime("%Y-%m-%d"), 'Saat': time_str, 'Ihlal_Turu': type_name})

# --- 3. ANA DÖNGÜ ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280) 
cap.set(4, 720)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success: continue
        
        frame_count += 1
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image.flags.writeable = True
        img_h, img_w, _ = image.shape
        
        current_status = "GUVENLI"
        status_color = COLOR_GREEN
        current_ear = 0.3 # Varsayılan

        # --- YÜZ VE GÖZ TAKİBİ ---
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                LEFT = [362, 385, 387, 263, 373, 380]
                RIGHT = [33, 160, 158, 133, 153, 144]
                
                left_ear = calculate_ear(lm, LEFT, img_w, img_h)
                right_ear = calculate_ear(lm, RIGHT, img_w, img_h)
                current_ear = (left_ear + right_ear) / 2.0
                
                for idx in LEFT + RIGHT:
                    pt = lm[idx]
                    cv2.circle(image, (int(pt.x * img_w), int(pt.y * img_h)), 1, (0, 255, 255), -1)

                if current_ear < EAR_THRESHOLD:
                    blink_counter += 1
                    if blink_counter >= EAR_FRAMES:
                        current_status = "!!! UYUYOR !!!"
                        status_color = COLOR_RED
                        log_incident("YORGUNLUK (UYUMA)")
                        if not alarm_on:
                            threading.Thread(target=play_alarm_sound).start()
                else:
                    blink_counter = 0

        # --- TELEFON TESPİTİ ---
        if frame_count % YOLO_SKIP_FRAMES == 0:
            yolo_results = model(image, verbose=False)
            phone_detected_buffer = False
            for r in yolo_results:
                for box in r.boxes:
                    if int(box.cls[0]) == TARGET_CLASS_ID:
                        phone_detected_buffer = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_ORANGE, 2)
                        len_line = 20
                        cv2.line(image, (x1, y1), (x1 + len_line, y1), COLOR_ORANGE, 4)
                        cv2.line(image, (x1, y1), (x1, y1 + len_line), COLOR_ORANGE, 4)
                        cv2.line(image, (x2, y2), (x2 - len_line, y2), COLOR_ORANGE, 4)
                        cv2.line(image, (x2, y2), (x2, y2 - len_line), COLOR_ORANGE, 4)

        if phone_detected_buffer:
            current_status = "!!! TELEFON TESPIT !!!"
            status_color = COLOR_ORANGE
            log_incident("DIKKAT (TELEFON)")
            if not alarm_on:
                threading.Thread(target=play_alarm_sound).start()

        # --- GRAFİK GÜNCELLEME ---
        ear_history.pop(0) 
        ear_history.append(current_ear) 
        
        if frame_count % 3 == 0:
            plot_img = create_live_plot(ear_history)
            
        if 'plot_img' in locals():
            ph, pw, _ = plot_img.shape
            offset_x = img_w - pw - 20
            offset_y = img_h - ph - 50
            if offset_x > 0 and offset_y > 0:
                image[offset_y:offset_y+ph, offset_x:offset_x+pw] = plot_img

        # --- HUD ÇİZİMİ ---
        draw_hud(image, current_status, status_color, current_ear)
        
        cv2.imshow('Askeri Operasyonel Takip Sistemi v3.1', image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# Raporu Kaydet
if log_data:
    # Pandas DataFrame oluştur
    df = pd.DataFrame(log_data)
    
    # 'results' klasörünü kontrol et, yoksa oluştur
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Klasör oluşturuldu: {results_dir}")
    
    # Dosya adını ve yolunu belirle
    # '..' bir üst klasöre çık demektir. Kod 'python' klasöründe olduğu için bir üste çıkıp 'results'a girmeli.
    filename = f"{results_dir}/gorev_raporu_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    # CSV olarak kaydet
    df.to_csv(filename, index=False)
    print(f"\n[INFO] Görev Raporu Başarıyla Kaydedildi: {filename}")
    print("-" * 30)
    print("Son Kaydedilen İhlaller:")
    print(df.tail()) # Son 5 kaydı göster
    print("-" * 30)
else:
    print("\n[INFO] Görev süresince herhangi bir ihlal tespit edilmedi.")