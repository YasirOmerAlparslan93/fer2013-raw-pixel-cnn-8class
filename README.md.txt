# Ham Piksel Verisinden Yüz İfadesi Tanıma için Derin Öğrenme Tabanlı CNN Modeli (FER2013 – 8 Sınıf)  
(A Deep Learning–Based CNN Model for Facial Expression Recognition from Raw Pixel Data)

**GitHub Repo Name:** `fer2013-raw-pixel-cnn-8class`  
**Course:** Derin Öğrenme ve Uygulamaları  
**Student:** <YOUR NAME> (<STUDENT ID>)  
**Instructor:** <INSTRUCTOR NAME>  
**Date:** Dec 2025 – Jan 2026

---

## 🇹🇷 Türkçe

### 1) Problem Tanımı
Bu proje, **FER2013** veri setindeki yüz görüntülerinden **ham piksel verisi** (48×48 gri seviye) kullanarak duygu sınıflandırması yapmayı hedefler.  
Model, **elle öznitelik çıkarımı (HOG/LBP/SIFT)** veya klasik ML yöntemleri olmadan, **uçtan uca (end-to-end)** bir **CNN** ile öğrenir.

**Sınıflar (8):** `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise`

---

### 2) Veri Seti ve Ön İşleme
- **Dataset:** FER2013 (48×48 grayscale facial images)
- **Normalization:** piksel değerleri `[0, 1]` aralığına ölçeklenir
- (Opsiyonel) **Augmentation:** küçük döndürme, yatay çevirme, parlaklık/kontrast değişimleri
- **Sınıf dengesizliği:** (opsiyonel) `class_weight` veya dengeli örnekleme

> Not: Bu repo, “ham piksel + derin öğrenme” şartına uygun olarak tasarlanmıştır.

---

### 3) Model Mimarisi ve Yaklaşımın Gerekçesi
Hafif ve güvenilir bir CNN mimarisi:
- Girdi: `48×48×1`
- Conv blokları: `Conv2D → BatchNorm → ReLU → MaxPool`
- Dense + Dropout
- Çıkış: Softmax (8 sınıf)

**Kayıp:** Categorical Cross-Entropy  
**Optimizer:** Adam

Amaç: düşük boyutlu ham görüntülerde yeterli genelleme sağlayan “küçük CNN” yaklaşımı.

---

### 4) Kurulum (Bağımlılıklar)
Python 3.10+ önerilir.

```bash
pip install -r requirements.txt


5) Çalıştırma Talimatları (Komutlar)

python src/train.py --config configs/config.yaml

Değerlendirme

python src/eval.py --weights models/best_model.keras


Tek bir görüntü üzerinde inference


python src/infer.py --image path/to/img.png --weights models/best_model.keras

Not (Önemli): Çalışma Notebook ile yapılmış olsa bile, bu repo öğretim üyesi değerlendirmesi için train/eval/infer script giriş noktaları sağlar. Script’ler aynı model/ön işleme kodunu çağıracak şekilde düzenlenmelidir.

6) Repo Yapısı

fer2013-raw-pixel-cnn-8class/
├─ src/
│  ├─ train.py
│  ├─ eval.py
│  ├─ infer.py
│  └─ utils.py                # (opsiyonel)
├─ configs/
│  └─ config.yaml
├─ models/
│  ├─ best_model.keras
│  └─ final_model.keras
├─ outputs/
│  ├─ figures/
│  │  ├─ accuracy_curve_8class.jpg
│  │  ├─ loss_curve_8class.jpg
│  │  ├─ confusion_matrix_counts_8class.jpg
│  │  ├─ confusion_matrix_normalized_8class.jpg
│  │  └─ sample_predictions_8class.jpg
│  └─ metrics/
│     └─ classification_report_8class.txt
├─ presentation/
│  └─ final_presentation.pdf
├─ requirements.txt
└─ README.md




7) Sonuçlar (Test)
Test Accuracy: 0.7181
Test Loss: 0.9193

Sınıflandırma Raporu (Özet):

Weighted F1: 0.6908

Macro F1: 0.4184

Özellikle Contempt / Disgust / Fear sınıflarında düşük performans gözlenmiştir (dengesizlik ve örnek sayısı azlığı etkili olabilir).

Eğitim Eğrileri ve Confusion Matrix:

Accuracy:

Loss:

Confusion Matrix (Counts):

Confusion Matrix (Normalized):

Örnek Tahminler (Test Inference):

8) Sunum
Nihai sunum dosyası:

presentation/final_presentation.pdf

Sunumda anlatılan deneyler repo içeriğiyle birebir örtüşmektedir (kod, metrikler, görseller).



1) Problem Definition
This project performs facial expression recognition on FER2013 using raw pixel data only (48×48 grayscale).
No hand-crafted feature extraction (HOG/LBP/SIFT) or classical ML models are used. The model learns representations end-to-end via a lightweight CNN.

Classes (8): Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise

2) Dataset & Preprocessing
Dataset: FER2013 (48×48 grayscale faces)

Normalization: scale pixels to [0, 1]

(Optional) Augmentation: small rotations, horizontal flips, brightness/contrast jitter

Class imbalance: (optional) class_weight or balanced sampling

3) Model Architecture & Rationale
A compact CNN:

Input: 48×48×1

Conv blocks: Conv2D → BatchNorm → ReLU → MaxPool

Dense + Dropout

Output: Softmax (8 classes)

Loss: Categorical Cross-Entropy
Optimizer: Adam

Goal: a lightweight yet reliable baseline CNN for raw low-resolution facial emotion classification.

4) Installation
Python 3.10+ recommended.

bash
Copy code
pip install -r requirements.txt
5) How to Run
Training
bash
Copy code
python src/train.py --config configs/config.yaml
Evaluation
bash
Copy code
python src/eval.py --weights models/best_model.keras
Inference on a single image
bash
Copy code
python src/infer.py --image path/to/img.png --weights models/best_model.keras
Note: Even if the main development was done in a Notebook, this repository provides script entrypoints (train/eval/infer) to make grading and reproducibility straightforward.

6) Repository Structure
(see the structure above in the Turkish section)

7) Test Results
Test Accuracy: 0.7181
Test Loss: 0.9193

Report highlights:

Weighted F1: 0.6908

Macro F1: 0.4184

Weak performance is observed for Contempt / Disgust / Fear (likely affected by class imbalance / low sample counts).

Figures:

Accuracy curve:

Loss curve:

Confusion matrices:


Sample predictions:

8) Presentation
Final slides:

presentation/final_presentation.pdf

Acknowledgements
FER2013 dataset (Kaggle mirror / public sources)

TensorFlow / Keras


