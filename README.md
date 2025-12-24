# 🛡️ Askeri Kara Araçlarında Operasyonel Güvenlik İçin  
## Yapay Zeka Destekli Sürücü Takip Sistemi

> *“Operasyonel süreklilik, personelin güvenliği ile başlar.”*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Görüntü%20İşleme-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Nesne%20Tespiti-red)
![Pandas](https://img.shields.io/badge/Pandas-Veri%20Analizi-purple)
![Durum](https://img.shields.io/badge/Durum-Prototip-success)

---

## 📌 Proje Tanımı

**Yapay Zeka Destekli Sürücü Takip Sistemi**, askeri kara araçlarında uzun süreli intikaller ve zorlu görev koşulları sırasında sürücü kaynaklı riskleri en aza indirmek amacıyla geliştirilmiş **gerçek zamanlı, otonom bir güvenlik ve karar destek sistemidir**.

Sistem; **yorgunluk**, **dikkat dağınıklığı** ve **kural ihlallerini** bilgisayarlı görü ve derin öğrenme yöntemleriyle tespit ederek operasyonel güvenliği artırmayı hedefler.

---

## 🎯 Projenin Amacı

Askeri lojistik ve operasyonel süreçlerde **insan faktörü** kritik öneme sahiptir. Yorgunluk ve dikkat kaybı, telafisi mümkün olmayan sonuçlara yol açabilir.

Bu proje aşağıdaki hedeflere odaklanır:

- **Gerçek Zamanlı Tespit:**  
  Sürücünün uyuklama ve dikkat kaybı durumlarını milisaniyeler içinde belirlemek

- **Kural İhlali Kontrolü:**  
  Sürüş esnasında yasaklı nesne (cep telefonu vb.) kullanımını otomatik olarak tespit etmek

- **Anlık Müdahale:**  
  Sesli ve görsel uyarılar ile sürücüyü bilgilendirerek kazaları önlemek

- **Veriye Dayalı Analiz:**  
  Tüm ihlalleri zaman damgalı olarak raporlayarak operasyonel iyileştirmelere veri sağlamak

---

## 🚀 Sistem Özellikleri

### 👁️ 1. Yorgunluk Tespiti (Drowsiness Detection)
- **Teknoloji:** Google MediaPipe Face Mesh  
- **Yöntem:** EAR (Eye Aspect Ratio) algoritması  
- **İşleyiş:**  
  Göz kapakları arasındaki mesafe sürekli ölçülür. Gözler belirlenen eşik değerin altında belirli bir süre kapalı kalırsa sistem **“Yorgunluk” alarmı** üretir.

---

### 📱 2. Dikkat Dağınıklığı ve Nesne Tespiti
- **Teknoloji:** Ultralytics YOLOv8  
- **Yöntem:** Derin öğrenme tabanlı nesne tespiti  
- **İşleyiş:**  
  Sürücünün elinde telefon gibi dikkat dağıtıcı bir nesne algılandığında sistem **“Kural İhlali” uyarısı** verir.

---

### 📊 3. Askeri HUD Arayüzü ve Canlı Grafik
- **HUD Tasarımı:**  
  Askeri operasyon hissi verecek şekilde özel olarak tasarlanmıştır.
- **Canlı EAR Grafiği:**  
  Sürücünün göz açıklık oranını gösteren kalp monitörü benzeri akan grafik
- **Durum Göstergeleri:**  
  - GÜVENLİ  
  - UYUYOR  
  - İHLAL  
- **Operasyon Saati ve Sistem Durumu**

---

### 📝 4. Otomatik Raporlama (Black Box)
- Sistem kapatıldığında görev süresince yaşanan tüm ihlaller otomatik olarak kaydedilir.
- **Dosya Formatı:**  
  `gorev_raporu_TARIH_SAAT.csv`
- **Kayıt İçeriği:**  
  - Tarih  
  - Saat  
  - İhlal Türü (Uyuma / Telefon)

---

## 🛠️ Kurulum ve Kullanım

### 🔧 Gereksinimler
- Python **3.10** veya **3.11**
- Web kamera
- Windows / Linux / macOS



### 📥 Adım 1: Repoyu Klonlayın
bash
git clone https://github.com/Gulnaz-Aydemir/Military-Driver-Monitoring-System.git
cd Military-Driver-Monitoring-System
### 🧪 Adım 2: Sanal Ortam Oluşturun 
bash
Kodu kopyala
python -m venv venv

# Windows
bash
venv\Scripts\activate

# macOS / Linux
bash
source venv/bin/activate

### 📦 Adım 3: Gerekli Kütüphaneleri Yükleyin
bash
Kodu kopyala
pip install opencv-python mediapipe ultralytics pandas numpy playsound matplotlib

### 📂 Adım 4: Gerekli Dosyaları Kontrol Edin

Aşağıdaki dosyaların proje klasöründe bulunduğundan emin olun:
bash
alarm.mp3 → Uyarı sesi
yolov8n.pt → YOLOv8 modeli (ilk çalıştırmada otomatik iner)

### ▶️ Adım 5: Sistemi Çalıştırın
bash
Kodu kopyala
python final_system_v3.py


Çıkış için q tuşuna basınız.
📷 Ekran Görüntüleri
Senaryo	Açıklama
Normal Sürüş	Güvenli sürüş durumu
Yorgunluk Tespiti	Gözler kapalı – alarm
Telefon Tespiti	Kural ihlali algılandı

🔬 Teknik Detaylar ve Kaynakça

EAR (Eye Aspect Ratio):
Soukupová & Čech (2016) – Gerçek zamanlı göz kırpma analizi

YOLOv8:
COCO veri seti ile eğitilmiş, gerçek zamanlı nesne tespiti modeli

📚 Veri Seti Referansları

State Farm Distracted Driver Detection Dataset (Kaggle)

UTA Real-Life Drowsiness Dataset (UTA-RLDD)

👨‍💻 Geliştirici

Gülnaz Aydemir
Endüstri Mühendisliği & Yapay Zeka Mühendisliği (Çift Anadal)

📄 Lisans

Bu proje eğitim ve araştırma amaçlı geliştirilmiştir.
Açık kaynaklı bir prototip çalışmadır.
