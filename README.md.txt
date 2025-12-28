Ham Piksel Verisinden Yüz İfadesi Tanıma için Derin Öğrenme Tabanlı CNN Modeli (FER2013 – 8 Sınıf)
(A Deep Learning–Based CNN Model for Facial Expression Recognition from Raw Pixel Data)
📌 Proje Özeti (Türkçe)

Bu projede, yüz görüntülerinden temel duyguların doğrudan ham piksel verisi kullanılarak otomatik olarak tanınmasını amaçlayan derin öğrenme tabanlı bir Evrişimsel Sinir Ağı (CNN) modeli geliştirilmiştir.

Model, FER2013 veri seti üzerinde eğitilmiş ve aşağıdaki 8 duygu sınıfını sınıflandırmaktadır:

Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise

Bu çalışmada:

HOG, LBP, SIFT gibi elle çıkarılmış öznitelikler

SVM, Random Forest gibi klasik makine öğrenmesi yöntemleri

❌ kullanılmamıştır.
Tüm öznitelik temsili, CNN tarafından uçtan uca (end-to-end) olarak öğrenilmiştir.

📌 Project Overview (English)

This project presents a deep learning–based facial expression recognition system trained directly on raw pixel data using a Convolutional Neural Network (CNN).

The model is trained and evaluated on the FER2013 dataset and classifies facial expressions into 8 emotion classes:

Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise

No handcrafted feature extraction (HOG, LBP, SIFT, etc.) or classical machine learning models are used.
All feature representations are learned end-to-end by the CNN.

🧠 Problem Tanımı / Problem Definition

Yüz ifadesi tanıma problemi, aşağıdaki zorluklar nedeniyle karmaşık bir bilgisayarlı görü problemidir:

Düşük çözünürlüklü (48×48) gri seviye görüntüler

Duygular arası yüksek benzerlik (ör. Anger – Sadness)

Ciddi sınıf dengesizliği (özellikle Contempt, Disgust, Fear)

Bu çalışmanın amacı, hafif ama etkili, genelleme kabiliyeti yüksek bir CNN modeli tasarlamaktır.

📂 Dataset

Veri Seti / Dataset: FER2013

Kaynak / Source: Kaggle

Görüntü Boyutu / Image Size: 48×48 (grayscale)

Bölünme / Split: Training / Validation / Test

Test örnek sayısı: 7,099

Ön İşleme ve Artırma / Preprocessing & Augmentation

Piksel normalizasyonu 
[
0
,
1
]
[0,1]

Veri artırma:

Küçük döndürmeler (rotation)

Yatay çevirme (horizontal flip)

Parlaklık / kontrast değişimleri

Sınıf dengesizliği için class_weight kullanımı

🏗️ Model Mimarisi / Model Architecture

Model, TensorFlow / Keras kullanılarak sıfırdan tasarlanmıştır:

Girdi / Input: 48×48×1

Conv Blok 1: Conv2D(32) → BatchNorm → ReLU → MaxPooling

Conv Blok 2: Conv2D(64) → BatchNorm → ReLU → MaxPooling

Conv Blok 3: Conv2D(128) → BatchNorm → ReLU → MaxPooling

Tam Bağlantılı Katman:

Dense(128, ReLU)

Dropout(0.3)

Çıkış Katmanı: Dense(8, Softmax)

Kayıp Fonksiyonu: Categorical Cross-Entropy
Optimizasyon: Adam (learning rate ≈ 1e-3)

⚙️ Kurulum / Installation
pip install -r requirements.txt

🚀 Çalıştırma Talimatları / How to Run
🔹 Eğitim / Training
python src/train.py --config configs/config.yaml

🔹 Değerlendirme (Test) / Evaluation
python src/eval.py --weights models/best_model.keras

🔹 Tek Görüntü Üzerinde Tahmin / Inference
python src/infer.py --image path/to/image.png --weights models/best_model.keras


ℹ️ Projenin ana geliştirme süreci notebook ortamında yapılmış olsa bile,
değerlendirme ve tekrar üretilebilirlik için eşdeğer Python scriptleri sağlanmıştır.

📊 Deneysel Sonuçlar / Experimental Results (8 Sınıf)

Test Accuracy: 71.81%

Test Loss: 0.9193

Sınıf Bazlı F1-Skorları
Sınıf	F1-score
Anger	0.58
Contempt	0.00
Disgust	0.00
Fear	0.00
Happiness	0.82
Neutral	0.77
Sadness	0.43
Surprise	0.75
Gözlemler

Happiness, Neutral ve Surprise sınıflarında yüksek başarı

Contempt, Disgust ve Fear sınıflarında düşük recall

Bunun temel nedeni: aşırı sınıf dengesizliği ve görsel benzerlik

📈 Görsel Çıktılar / Visual Outputs

outputs/ klasörü içinde:

Eğitim / doğrulama doğruluk eğrileri

Eğitim / doğrulama kayıp eğrileri

Confusion Matrix (Count & Normalized)

Test seti üzerinde örnek tahmin görselleri

📽️ Proje Sunumu / Project Presentation

Nihai sunum dosyası (PDF) aşağıdaki dizinde yer almaktadır:

presentation/final_presentation.pdf


Sunumda yer alan tüm deneyler ve sonuçlar,
bu GitHub deposundaki kod ve çıktılar ile birebir örtüşmektedir.

🔁 Tekrar Üretilebilirlik / Reproducibility

Sabit random seed kullanımı

Açık bağımlılık listesi (requirements.txt)

Ayrı eğitim / değerlendirme / inference scriptleri

📚 Kaynaklar / References

FER2013 Facial Expression Recognition Dataset (Kaggle)

Keras Sequential Model Guide

CNN tabanlı yüz ifadesi tanıma literatürü