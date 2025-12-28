
#götür dataseti dosyasını aynı notbook jupiterın dosyasınde sonra bu eğitim kodu çalıştır 

%pip install tensorflow

import tensorflow as tf
print(tf.__version__)


%pip install torch torchvision torchaudio

import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())


import tensorflow as tf
import torch

print("TensorFlow version:", tf.__version__)
print("PyTorch version:", torch.__version__)



%pip install "protobuf<6,>=3.20"




import tensorflow as tf
import torch
import google.protobuf

print("TensorFlow:", tf.__version__)
print("PyTorch:", torch.__version__)
print("Protobuf:", google.protobuf.__version__)

# ============================================================
# Cell 1 – Imports and general configuration
# Hücre 1 – Kütüphaneler ve genel ayarlar
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import itertools

print("TensorFlow version:", tf.__version__)

# Dataset directories
# Veri seti dizinleri
DATA_DIR = "./fer2013"          # Contains train/ and test/
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

# Optional external images directory
# Harici test görüntüleri (isteğe bağlı)
EXTERNAL_IMAGES_DIR = "./external_images"

# Results directory
# Sonuçların kaydedileceği klasör
RESULTS_DIR = "./training_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Global constants
# Genel sabitler
IMG_SIZE = (48, 48)
BATCH_SIZE = 64

# Solution B: 8 classes (including contempt)
# Çözüm B: 8 sınıf (contempt dahil)
NUM_CLASSES = 8
CLASS_NAMES = ["anger", "contempt", "disgust", "fear",
               "happiness", "neutral", "sadness", "surprise"]

EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]

# Number of epochs (minimum 15-60 as requested)
# Epoch sayısı (istenildiği gibi minimum 15-60)
EPOCHS = 15
#EPOCHS = 60
#EPOCHS = 360

# ============================================================
# Cell 2 – Load training, validation and test datasets
# Hücre 2 – Eğitim, doğrulama ve test verilerinin yüklenmesi
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

# Normalize images and convert labels to one-hot encoding
# Görüntüleri normalize et ve etiketleri one-hot yap
def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    labels_one_hot = tf.one_hot(labels, depth=NUM_CLASSES)
    return images, labels_one_hot

train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)
test_ds  = test_ds.map(preprocess)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# Cell 3 – Visualize sample images (optional)
# Hücre 3 – Örnek görüntüleri göster (isteğe bağlı)
# ============================================================

def show_sample_batch(dataset, n=9):
    images, labels = next(iter(dataset))
    images = images.numpy()
    labels = tf.argmax(labels, axis=1).numpy()

    plt.figure(figsize=(6, 6))
    for i in range(min(n, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(EMOTION_LABELS[labels[i]])
    plt.tight_layout()
    plt.show()

show_sample_batch(train_ds)


# ============================================================
# Cell 4 – Define CNN model from scratch
# Hücre 4 – CNN modelinin sıfırdan tanımlanması
# ============================================================

def make_cnn(input_shape=(48, 48, 1), num_classes=NUM_CLASSES):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="emotion_cnn_8class")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

model = make_cnn()
model.summary()


# ============================================================
# Cell 5 – Callbacks (checkpoint and LR scheduler)
# Hücre 5 – Callback’ler (model kaydı ve öğrenme oranı)
# ============================================================

checkpoint_path = os.path.join(RESULTS_DIR, "best_emotion_cnn_8class.keras")

cb_list = [
    callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
    # EarlyStopping is not used to complete all epochs
    # Tüm epoch’ları tamamlamak için EarlyStopping kullanılmadı
]


# ============================================================
# Cell 6 – Train the model
# Hücre 6 – Modelin eğitilmesi
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cb_list
)

final_model_path = os.path.join(RESULTS_DIR, "final_emotion_cnn_8class.keras")
model.save(final_model_path)


# ============================================================
# Cell 7 – Plot and save accuracy & loss curves
# Hücre 7 – Doğruluk ve kayıp grafiklerinin çizilmesi
# ============================================================

def plot_and_save_history(history, results_dir=RESULTS_DIR):
    # Accuracy curve
    # Doğruluk grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title("Training vs Validation Accuracy (8-class)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(results_dir, "accuracy_curve_8class.jpg")
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.show()

    # Loss curve
    # Kayıp grafiği
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training vs Validation Loss (8-class)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(results_dir, "loss_curve_8class.jpg")
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("Saved:", acc_path, "and", loss_path)

plot_and_save_history(history)


# ============================================================
# Cell 8 – Evaluate on test set and save classification report
# Hücre 8 – Test verisi üzerinde değerlendirme
# ============================================================

y_true = []
y_pred = []

for batch_images, batch_labels_one_hot in test_ds:
    preds = model.predict(batch_images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(batch_labels_one_hot.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

report = classification_report(
    y_true,
    y_pred,
    target_names=EMOTION_LABELS,
    digits=4
)

print(report)

report_path = os.path.join(RESULTS_DIR, "classification_report_8class.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

print("Saved report to:", report_path)


# ============================================================
# Cell 9 – Confusion matrix (raw and normalized)
# Hücre 9 – Karışıklık matrisi (ham ve normalize)
# ============================================================

cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix"):
    if normalize:
        cm_show = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        cm_show = cm
        fmt = "d"

    plt.figure(figsize=(8, 8))
    plt.imshow(cm_show, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_show.max() / 2.0
    for i, j in itertools.product(range(cm_show.shape[0]), range(cm_show.shape[1])):
        plt.text(
            j, i,
            format(cm_show[i, j], fmt),
            ha="center",
            color="white" if cm_show[i, j] > thresh else "black"
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

# Raw counts
# Ham sayılar
plot_confusion_matrix(cm, EMOTION_LABELS, normalize=False,
                      title="Confusion Matrix (Counts) - 8 class")
cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_counts_8class.jpg")
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.show()

# Normalized matrix
# Normalize edilmiş matris
plot_confusion_matrix(cm, EMOTION_LABELS, normalize=True,
                      title="Confusion Matrix (Normalized) - 8 class")
cm_norm_path = os.path.join(RESULTS_DIR, "confusion_matrix_normalized_8class.jpg")
plt.savefig(cm_norm_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved CM:", cm_path, "and", cm_norm_path)


# ============================================================
# Cell 10 – Show sample predictions from test set
# Hücre 10 – Test setinden örnek tahminler
# ============================================================

def plot_sample_predictions(dataset, n=16,
                            results_dir=RESULTS_DIR,
                            filename="sample_predictions_8class.jpg"):

    images, labels_one_hot = next(iter(dataset))
    images_np = images.numpy()
    true_labels = np.argmax(labels_one_hot.numpy(), axis=1)

    preds = model.predict(images, verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    n = min(n, len(images_np))
    plt.figure(figsize=(12, 12))

    for i in range(n):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images_np[i].squeeze(), cmap="gray")
        plt.axis("off")

        t = EMOTION_LABELS[true_labels[i]]
        p = EMOTION_LABELS[pred_labels[i]]
        color = "green" if true_labels[i] == pred_labels[i] else "red"

        plt.title(f"T:{t}\nP:{p}", color=color)

    plt.tight_layout()
    out_path = os.path.join(results_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("Saved:", out_path)

plot_sample_predictions(test_ds)


# ============================================================
# Cell 11 – Predict external images and save results
# Hücre 11 – Harici görüntüler için tahmin
# ============================================================

from tensorflow.keras.preprocessing import image

def predict_external_images(
    folder_path=EXTERNAL_IMAGES_DIR,
    model=model,
    img_size=IMG_SIZE,
    results_dir=RESULTS_DIR,
    output_file="external_predictions_8class.txt"
):
    if not os.path.isdir(folder_path):
        print("No external_images folder. Skipping.")
        return

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if not files:
        print("No images found in external_images.")
        return

    out_lines = []
    plt.figure(figsize=(12, 12))

    for i, fname in enumerate(files[:16]):
        img_path = os.path.join(folder_path, fname)

        img = image.load_img(
            img_path,
            color_mode="grayscale",
            target_size=img_size
        )

        img_arr = image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        pred = model.predict(img_arr, verbose=0)
        pred_label_int = int(np.argmax(pred, axis=1)[0])
        pred_label_name = EMOTION_LABELS[pred_label_int]

        out_lines.append(
            f"{fname}\t{pred_label_int}\t{pred_label_name}\t{pred[0]}"
        )

        plt.subplot(4, 4, i + 1)
        plt.imshow(img_arr[0].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title(pred_label_name)

    plt.tight_layout()
    ext_fig_path = os.path.join(
        results_dir,
        "external_images_predictions_8class.jpg"
    )
    plt.savefig(ext_fig_path, dpi=200, bbox_inches="tight")
    plt.show()

    out_path = os.path.join(results_dir, output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("filename\tpred_label_id\tpred_label_name\tprob_vector\n")
        for line in out_lines:
            f.write(line + "\n")

    print("Saved external predictions to:")
    print("Text:", out_path)
    print("Image:", ext_fig_path)

predict_external_images()


# ============================================================
# Cell 12 – Grad-CAM (TR/EN comments only)
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------
# TR: Son Conv2D katmanını otomatik bul
# EN: Auto-detect the last Conv2D layer
# ----------------------------
def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

# ----------------------------
# TR: Grad-CAM ısı haritası üret
# EN: Generate Grad-CAM heatmap
# ----------------------------
def make_gradcam_heatmap(
    img_tensor: tf.Tensor,
    model: tf.keras.Model,
    last_conv_layer_name: str = None,
    class_index: int = None
):
    # TR: Varsayılan olarak son conv katmanı
    # EN: Use last conv layer by default
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    # TR: Conv çıktısı + model çıktısı veren ara model
    # EN: Build a model that maps input -> (last_conv_output, predictions)
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)

        # TR: Hedef sınıf seçimi (yoksa modelin tahmini)
        # EN: Target class selection (if None, use predicted class)
        if class_index is None:
            class_index = tf.argmax(preds[0])

        # TR: Sınıf skoru (softmax çıkışı)
        # EN: Class score (softmax output)
        class_score = preds[:, class_index]

    # TR: Gradients = d(score)/d(conv_outputs)
    # EN: Gradients = d(score)/d(conv_outputs)
    grads = tape.gradient(class_score, conv_outputs)

    # TR: Kanal ağırlıkları (global average pooling)
    # EN: Channel weights (global average pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # TR: Ağırlıklı toplam
    # EN: Weighted sum
    conv_outputs = conv_outputs[0]  # shape: (H, W, C)
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # TR: ReLU ve normalize
    # EN: ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    denom = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / denom

    return heatmap.numpy(), int(class_index)

# ----------------------------
# TR: Heatmap'i görüntüye bindir (overlay)
# EN: Overlay heatmap on the original image
# ----------------------------
def overlay_gradcam(
    img_gray: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.35
):
    # TR: img_gray: (H,W) veya (H,W,1) [0,1]
    # EN: img_gray: (H,W) or (H,W,1) in [0,1]
    if img_gray.ndim == 3:
        img_gray = img_gray.squeeze()

    h, w = img_gray.shape
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy().squeeze()

    # TR: Renk haritası uygula
    # EN: Apply colormap
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap_resized)[..., :3]  # drop alpha

    # TR: Gri görüntüyü RGB'ye çevir
    # EN: Convert gray image to RGB
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)

    # TR: Karıştır (overlay)
    # EN: Blend (overlay)
    overlay = (1 - alpha) * img_rgb + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    return overlay

# ----------------------------
# TR: Test setinden örneklerle Grad-CAM görselleştir ve kaydet
# EN: Visualize & save Grad-CAM for sample test images
# ----------------------------
def save_gradcam_samples(
    dataset,
    model,
    emotion_labels,
    results_dir,
    n=16,
    filename="gradcam_samples_8class.jpg",
    target_mode="pred"  # "pred" or "true"
):
    os.makedirs(results_dir, exist_ok=True)

    images, labels_one_hot = next(iter(dataset))
    images_np = images.numpy()                  # (B,48,48,1)
    true_ids = np.argmax(labels_one_hot.numpy(), axis=1)

    preds = model.predict(images, verbose=0)
    pred_ids = np.argmax(preds, axis=1)

    last_conv_name = get_last_conv_layer_name(model)

    n = min(n, images_np.shape[0])
    plt.figure(figsize=(12, 12))

    for i in range(n):
        img = images_np[i]                      # (48,48,1) in [0,1]
        img_tensor = tf.convert_to_tensor(img[np.newaxis, ...], dtype=tf.float32)

        # TR: Hangi sınıfa göre açıklayacağımızı seç
        # EN: Choose which class to explain
        if target_mode == "true":
            class_id = int(true_ids[i])
        else:
            class_id = None  # explain predicted by default

        heatmap, used_class = make_gradcam_heatmap(
            img_tensor,
            model,
            last_conv_layer_name=last_conv_name,
            class_index=class_id
        )

        overlay = overlay_gradcam(img.squeeze(), heatmap, alpha=0.35)

        t = emotion_labels[int(true_ids[i])]
        p = emotion_labels[int(pred_ids[i])]
        used = emotion_labels[int(used_class)]
        ok = (true_ids[i] == pred_ids[i])

        plt.subplot(4, 4, i + 1)
        plt.imshow(overlay)
        plt.axis("off")
        # TR/EN: Title shows True / Pred / Explained-class
        plt.title(f"T:{t}\nP:{p}\nCAM:{used}", color=("green" if ok else "red"), fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(results_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", format="jpg")
    plt.show()
    print("Saved Grad-CAM samples:", out_path)

# Example usage:
# TR: En iyi modeli yüklemek istersen (opsiyonel)
# EN: If you want to load the best checkpoint (optional)
# best_model = tf.keras.models.load_model(checkpoint_path)

# TR: Mevcut 'model' ile çalış
# EN: Use the current 'model'
save_gradcam_samples(
    dataset=test_ds,
    model=model,
    emotion_labels=EMOTION_LABELS,
    results_dir=RESULTS_DIR,
    n=16,
    filename="gradcam_samples_8class.jpg",
    target_mode="pred"  # explain predicted class
)

# TR: Gerçek etiket üzerinden açıklama yapmak istersen:
# EN: If you want to explain based on the true class:
save_gradcam_samples(
    dataset=test_ds,
    model=model,
    emotion_labels=EMOTION_LABELS,
    results_dir=RESULTS_DIR,
    n=16,
    filename="gradcam_samples_TRUEclass_8class.jpg",
    target_mode="true"
)



