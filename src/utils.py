"""
TR: FER2013/FER2013Plus 8-sınıf duygu sınıflandırma yardımcıları
EN: Utilities for FER2013/FER2013Plus 8-class emotion classification

TR: Bu paket, notebook'taki (Cell 1-12) akışı script yapısına taşır:
    - Veri yükleme + normalize + one-hot
    - CNN modeli (scratch)
    - Class-weight / Focal Loss (opsiyonel)
    - Overfitting azaltma: EarlyStopping, Dropout, ReduceLROnPlateau
    - Confusion matrix + classification report
    - Grad-CAM

EN: This converts the notebook (Cell 1-12) flow into a script-friendly structure:
    - Dataset loading + normalize + one-hot
    - CNN model (from scratch)
    - Class-weight / Focal Loss (optional)
    - Reduce overfitting: EarlyStopping, Dropout, ReduceLROnPlateau
    - Confusion matrix + classification report
    - Grad-CAM
"""

from __future__ import annotations

import os
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, callbacks


# ----------------------------
# TR: Sınıf isimleri (klasör adlarıyla aynı olmalı)
# EN: Class names (must match folder names)
# ----------------------------
CLASS_NAMES = ["anger", "contempt", "disgust", "fear",
               "happiness", "neutral", "sadness", "surprise"]

EMOTION_LABELS = ["Anger", "Contempt", "Disgust", "Fear",
                  "Happiness", "Neutral", "Sadness", "Surprise"]


# ----------------------------
# TR: Config dataclass
# EN: Config dataclass
# ----------------------------
@dataclass
class DataConfig:
    data_dir: str                 # expects train/ and test/
    img_size: Tuple[int, int] = (48, 48)
    batch_size: int = 64
    num_classes: int = 8
    seed: int = 123
    val_split: float = 0.2


# ----------------------------
# TR: Dataset yükleme + preprocess
# EN: Dataset loading + preprocess
# ----------------------------
def build_datasets(cfg: DataConfig):
    train_dir = os.path.join(cfg.data_dir, "train")
    test_dir  = os.path.join(cfg.data_dir, "test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        batch_size=cfg.batch_size,
        image_size=cfg.img_size,
        shuffle=True,
        seed=cfg.seed,
        validation_split=cfg.val_split,
        subset="training"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        batch_size=cfg.batch_size,
        image_size=cfg.img_size,
        shuffle=True,
        seed=cfg.seed,
        validation_split=cfg.val_split,
        subset="validation"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="grayscale",
        batch_size=cfg.batch_size,
        image_size=cfg.img_size,
        shuffle=False
    )

    def preprocess(images, labels):
        # TR: [0..255] -> [0..1]
        # EN: [0..255] -> [0..1]
        images = tf.cast(images, tf.float32) / 255.0
        labels_oh = tf.one_hot(labels, depth=cfg.num_classes)
        return images, labels_oh

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(preprocess, num_parallel_calls=autotune).cache().shuffle(1000).prefetch(autotune)
    val_ds   = val_ds.map(preprocess,   num_parallel_calls=autotune).cache().prefetch(autotune)
    test_ds  = test_ds.map(preprocess,  num_parallel_calls=autotune).prefetch(autotune)

    return train_ds, val_ds, test_ds


# ----------------------------
# TR: Augmentation (opsiyonel)
# EN: Augmentation (optional)
# ----------------------------
def make_augmentation(cfg: Dict) -> tf.keras.Sequential:
    if not cfg.get("enable", False):
        return tf.keras.Sequential(name="aug_disabled")

    rot = float(cfg.get("rotation", 0.10))
    zoom = float(cfg.get("zoom", 0.10))
    con = float(cfg.get("contrast", 0.10))
    trans = float(cfg.get("translation", 0.08))

    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(rot),
        layers.RandomZoom(zoom),
        layers.RandomTranslation(trans, trans),
        layers.RandomContrast(con),
    ], name="augmentation")


# ----------------------------
# TR: CNN modeli (scratch)
# EN: CNN model (scratch)
# ----------------------------
def make_cnn(
    input_shape=(48, 48, 1),
    num_classes: int = 8,
    dropout: float = 0.30,
    learning_rate: float = 1e-3,
    augmentation_cfg: Optional[Dict] = None,
):
    aug = make_augmentation(augmentation_cfg or {"enable": False})

    inputs = layers.Input(shape=input_shape)
    x = aug(inputs)

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
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
    x = layers.Dropout(float(dropout))(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="emotion_cnn_8class")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(learning_rate)),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ----------------------------
# TR: Class-weight hesapla
# EN: Compute class weights
# ----------------------------
def compute_class_weights_from_dataset(train_ds) -> Dict[int, float]:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    for _, y in train_ds:
        y_ids = tf.argmax(y, axis=1).numpy()
        for k in y_ids:
            counts[int(k)] += 1

    total = counts.sum()
    K = len(CLASS_NAMES)

    weights = {}
    for k in range(K):
        weights[k] = float(total / (K * counts[k])) if counts[k] > 0 else 1.0
    return weights


# ----------------------------
# TR: Focal Loss (opsiyonel)
# EN: Focal Loss (optional)
# ----------------------------
def focal_loss(gamma: float = 2.0, alpha: Optional[List[float]] = None):
    alpha_t = None
    if alpha is not None:
        alpha_t = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        modulating = tf.pow(1.0 - pt, gamma)

        fl = modulating * ce
        if alpha_t is not None:
            alpha_factor = tf.reduce_sum(y_true * alpha_t, axis=-1)
            fl = alpha_factor * fl

        return tf.reduce_mean(fl)

    return loss_fn


def class_weights_to_alpha(class_weights: Dict[int, float], num_classes: int = 8) -> List[float]:
    alpha = np.ones(num_classes, dtype=np.float32)
    for k, v in class_weights.items():
        alpha[int(k)] = float(v)
    alpha = alpha / (alpha.sum() + 1e-8)
    return alpha.tolist()


def compile_with_loss(
    model: tf.keras.Model,
    learning_rate: float,
    loss_name: str,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[List[float]] = None,
):
    if str(loss_name).lower() == "focal":
        loss_fn = focal_loss(gamma=float(focal_gamma), alpha=focal_alpha)
    else:
        loss_fn = "categorical_crossentropy"

    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(learning_rate)),
        loss=loss_fn,
        metrics=["accuracy"]
    )
    return model


# ----------------------------
# TR: Callbacks (EarlyStopping dahil)
# EN: Callbacks (with EarlyStopping)
# ----------------------------
def build_callbacks(
    ckpt_path: str,
    use_early_stopping: bool = True,
    es_patience: int = 8,
    reduce_lr_patience: int = 5,
    min_lr: float = 1e-5,
):
    cb = [
        callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=int(reduce_lr_patience),
            min_lr=float(min_lr),
            verbose=1
        ),
    ]
    if use_early_stopping:
        cb.append(
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(es_patience),
                restore_best_weights=True,
                verbose=1
            )
        )
    return cb


# ----------------------------
# TR: History plot (accuracy/loss)
# EN: History plot (accuracy/loss)
# ----------------------------
def plot_and_save_history(history, results_dir: str, prefix: str = "8class"):
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title(f"Training vs Validation Accuracy ({prefix})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(results_dir, f"accuracy_curve_{prefix}.jpg")
    plt.savefig(acc_path, dpi=200, bbox_inches="tight", format="jpg")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title(f"Training vs Validation Loss ({prefix})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(results_dir, f"loss_curve_{prefix}.jpg")
    plt.savefig(loss_path, dpi=200, bbox_inches="tight", format="jpg")
    plt.close()

    return acc_path, loss_path


# ----------------------------
# TR: Confusion matrix çizimi
# EN: Confusion matrix plotting
# ----------------------------
def plot_confusion_matrix(cm: np.ndarray, classes: List[str], normalize: bool = False, title: str = "Confusion Matrix"):
    if normalize:
        cm_show = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
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
        plt.text(j, i, format(cm_show[i, j], fmt),
                 ha="center",
                 color="white" if cm_show[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


# ============================================================
# Cell 12 – Grad-CAM (TR/EN comments only)
# ============================================================

def get_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")


def make_gradcam_heatmap(
    img_tensor: tf.Tensor,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = None,
    class_index: Optional[int] = None
):
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer_name(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_tensor, training=False)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(class_index)


def overlay_gradcam(img_gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35):
    if img_gray.ndim == 3:
        img_gray = img_gray.squeeze()

    h, w = img_gray.shape
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy().squeeze()

    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap_resized)[..., :3]

    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
    overlay = (1 - alpha) * img_rgb + alpha * heatmap_rgb
    return np.clip(overlay, 0, 1)


def save_gradcam_samples(
    dataset,
    model: tf.keras.Model,
    emotion_labels: List[str],
    results_dir: str,
    n: int = 16,
    filename: str = "gradcam_samples_8class.jpg",
    target_mode: str = "pred"
):
    os.makedirs(results_dir, exist_ok=True)

    images, labels_one_hot = next(iter(dataset))
    images_np = images.numpy()
    true_ids = np.argmax(labels_one_hot.numpy(), axis=1)

    preds = model.predict(images, verbose=0)
    pred_ids = np.argmax(preds, axis=1)

    last_conv_name = get_last_conv_layer_name(model)

    n = min(n, images_np.shape[0])
    plt.figure(figsize=(12, 12))

    for i in range(n):
        img = images_np[i]
        img_tensor = tf.convert_to_tensor(img[np.newaxis, ...], dtype=tf.float32)

        if target_mode == "true":
            class_id = int(true_ids[i])
        else:
            class_id = None

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
        plt.title(f"T:{t}\nP:{p}\nCAM:{used}", color=("green" if ok else "red"), fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(results_dir, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", format="jpg")
    plt.close()
    return out_path


# ----------------------------
# TR: Tek görüntü preprocess
# EN: Single-image preprocess
# ----------------------------
def load_and_preprocess_image(image_path: str, img_size=(48, 48)) -> np.ndarray:
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=img_size)
    arr = tf.keras.preprocessing.image.img_to_array(img).astype("float32") / 255.0
    return arr
