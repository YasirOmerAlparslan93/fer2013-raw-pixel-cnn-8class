# -*- coding: utf-8 -*-
"""
Inference on a Single Image / Tek Görüntü Üzerinde Çalıştırma
python src/infer.py --image path/to/img.png --weights models/best_emotion_model_8class.keras --save_cam
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import EMOTION_LABELS, load_and_preprocess_image, make_gradcam_heatmap, overlay_gradcam


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./outputs")
    p.add_argument("--save_cam", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = tf.keras.models.load_model(args.weights, compile=False)

    img = load_and_preprocess_image(args.image, img_size=(48, 48))
    x = tf.convert_to_tensor(img[np.newaxis, ...], dtype=tf.float32)

    probs = model.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    pred_name = EMOTION_LABELS[pred_id]

    print("Prediction:", pred_id, pred_name)
    print("Probabilities:", probs)

    if args.save_cam:
        heatmap, _ = make_gradcam_heatmap(x, model, class_index=None)
        overlay = overlay_gradcam(img.squeeze(), heatmap, alpha=0.35)

        out_path = os.path.join(args.out_dir, "gradcam_single_image.jpg")
        plt.figure(figsize=(5, 5))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"P:{pred_name}", fontsize=12)
        plt.savefig(out_path, dpi=200, bbox_inches="tight", format="jpg")
        plt.close()
        print("Saved Grad-CAM:", out_path)


if __name__ == "__main__":
    main()
