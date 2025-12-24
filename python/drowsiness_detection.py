import cv2
import mediapipe as mp
import numpy as np
import time

# --- 1. AYARLAR VE SABİTLER ---
EAR_THRESHOLD = 0.22  # Göz açıklık oranı bu değerin altına düşerse göz kapalı sayılır
EAR_FRAMES = 10       # Kaç kare boyunca göz kapalı kalırsa uyarı verilsin? (Yaklaşık 2-3 saniye)

# MediaPipe araçlarını başlattım.
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Gözlerin Landmark İndeksleri (MediaPipe standardına göre)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --- 2. YARDIMCI FONKSİYONLAR ---
def calculate_ear(landmarks, eye_indices, img_w, img_h):
    """
    Göz açıklık oranını (EAR) hesaplar.
    """
    # Göz noktalarının koordinatlarını aldım
    # p1, p4: Gözün yatay uçları (Sol, Sağ)
    # p2, p6 ve p3, p5: Gözün dikey uçları (Üst, Alt)
    
    # Koordinatları piksel değerine çevirdim
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * img_w), int(lm.y * img_h)))
    
    p2_p6 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5])) # Dikey 1
    p3_p5 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4])) # Dikey 2
    p1_p4 = np.linalg.norm(np.array(coords[0]) - np.array(coords[3])) # Yatay
    
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return ear

# --- 3. ANA DÖNGÜ ---
cap = cv2.VideoCapture(0)
blink_counter = 0 # Gözlerin kaç kare boyunca kapalı kaldığını sayan kod
alarm_active = False

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    print("Yorgunluk Takip Sistemi Başlatıldı... Çıkış için 'q' basınız.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Kamera okunamadı.")
            continue

        # Görüntüyü hazırla
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        img_h, img_w, _ = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Tüm yüz noktaları
                landmarks = face_landmarks.landmark
                
                # Sol ve Sağ Göz EAR Hesapla
                left_ear = calculate_ear(landmarks, LEFT_EYE, img_w, img_h)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, img_w, img_h)
                
                # İki gözün ortalamasını al
                avg_ear = (left_ear + right_ear) / 2.0
                
                # --- YORGUNLUK KONTROLÜ ---
                if avg_ear < EAR_THRESHOLD:
                    blink_counter += 1
                    
                    # Eğer sayaç eşik değeri geçerse (Örn: 40 kare boyunca kapalıysa)
                    if blink_counter >= EAR_FRAMES:
                        alarm_active = True
                        cv2.putText(image, "UYARI: YORGUNLUK TESPIT EDILDI!", (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        print(f"ALARM! EAR: {avg_ear:.2f}")
                else:
                    blink_counter = 0
                    alarm_active = False
                
                # Ekrana EAR değerini yazdır (Test için)
                cv2.putText(image, f"EAR: {avg_ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Göz çevrelerini çiz (Görsellik için)
                for idx in LEFT_EYE + RIGHT_EYE:
                    lm = landmarks[idx]
                    cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), 2, (0, 255, 0), -1)

        # Görüntüyü göster
        cv2.imshow('Surucu Yorgunluk Tespiti', cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()