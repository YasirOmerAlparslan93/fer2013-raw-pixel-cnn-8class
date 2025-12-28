# -*- coding: utf-8 -*-
"""
Evaluation / Değerlendirme
python src/eval.py --weights models/best_emotion_model_8class.keras --data_dir ./fer2013 --results_dir ./outputs
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    DataConfig, build_datasets,
    EMOTION_LABELS,
    plot_confusion_matrix,
    save_gradcam_samples
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./fer2013")
    p.add_argument("--results_dir", type=str, default="./outputs")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    dcfg = DataConfig(data_dir=args.data_dir, batch_size=int(args.batch_size))
    _, _, test_ds = build_datasets(dcfg)

    model = tf.keras.models.load_model(args.weights, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    y_true, y_pred = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(p, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=EMOTION_LABELS, digits=4)
    print(report)

    report_path = os.path.join(args.results_dir, "classification_report_8class.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(cm, EMOTION_LABELS, normalize=False, title="Confusion Matrix (Counts) - 8 class")
    cm_path = os.path.join(args.results_dir, "confusion_matrix_counts_8class.jpg")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight", format="jpg")
    plt.close()

    plot_confusion_matrix(cm, EMOTION_LABELS, normalize=True, title="Confusion Matrix (Normalized) - 8 class")
    cm_norm_path = os.path.join(args.results_dir, "confusion_matrix_normalized_8class.jpg")
    plt.savefig(cm_norm_path, dpi=200, bbox_inches="tight", format="jpg")
    plt.close()

    cam_pred = save_gradcam_samples(
        dataset=test_ds,
        model=model,
        emotion_labels=EMOTION_LABELS,
        results_dir=args.results_dir,
        n=16,
        filename="gradcam_samples_PREDclass_8class.jpg",
        target_mode="pred"
    )
    cam_true = save_gradcam_samples(
        dataset=test_ds,
        model=model,
        emotion_labels=EMOTION_LABELS,
        results_dir=args.results_dir,
        n=16,
        filename="gradcam_samples_TRUEclass_8class.jpg",
        target_mode="true"
    )

    print("Saved:")
    print("  Report:", report_path)
    print("  CM:", cm_path)
    print("  CM Norm:", cm_norm_path)
    print("  Grad-CAM pred:", cam_pred)
    print("  Grad-CAM true:", cam_true)


if __name__ == "__main__":
    main()
