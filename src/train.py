# -*- coding: utf-8 -*-
"""
Training / Eğitim
python src/train.py --config configs/config.yaml
"""

from __future__ import annotations

import os
import argparse
import yaml

from utils import (
    DataConfig, build_datasets,
    make_cnn, compute_class_weights_from_dataset,
    class_weights_to_alpha, compile_with_loss,
    build_callbacks, plot_and_save_history
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["data"]["data_dir"]
    results_dir = cfg["paths"]["results_dir"]
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    dcfg = DataConfig(
        data_dir=data_dir,
        img_size=tuple(cfg["data"].get("img_size", [48, 48])),
        batch_size=int(cfg["data"].get("batch_size", 64)),
        num_classes=int(cfg["model"].get("num_classes", 8)),
        seed=int(cfg["data"].get("seed", 123)),
        val_split=float(cfg["data"].get("val_split", 0.2)),
    )
    train_ds, val_ds, test_ds = build_datasets(dcfg)

    model = make_cnn(
        input_shape=tuple(cfg["model"].get("input_shape", [48, 48, 1])),
        num_classes=int(cfg["model"].get("num_classes", 8)),
        dropout=float(cfg["model"].get("dropout", 0.40)),
        learning_rate=float(cfg["training"].get("learning_rate", 1e-3)),
        augmentation_cfg=cfg.get("augmentation", {"enable": False})
    )

    # TR: Sınıf dengesizliği için class_weight
    # EN: class_weight for imbalance
    class_weights = None
    if bool(cfg["training"].get("use_class_weight", True)):
        class_weights = compute_class_weights_from_dataset(train_ds)

    # TR: Focal loss seçildiyse alpha üret
    # EN: If focal loss is selected, build alpha
    loss_name = str(cfg["training"].get("loss", "categorical_crossentropy")).lower()
    focal_alpha = None
    if loss_name == "focal":
        if class_weights is None:
            class_weights = compute_class_weights_from_dataset(train_ds)
        focal_alpha = class_weights_to_alpha(class_weights, num_classes=int(cfg["model"].get("num_classes", 8)))

    model = compile_with_loss(
        model,
        learning_rate=float(cfg["training"].get("learning_rate", 1e-3)),
        loss_name=loss_name,
        focal_gamma=float(cfg["training"].get("focal_gamma", 2.0)),
        focal_alpha=focal_alpha
    )

    best_path = os.path.join(models_dir, cfg["paths"].get("best_model_name", "best_emotion_model_8class.keras"))
    final_path = os.path.join(models_dir, cfg["paths"].get("final_model_name", "final_emotion_model_8class.keras"))

    cb_list = build_callbacks(
        ckpt_path=best_path,
        use_early_stopping=bool(cfg["training"].get("use_early_stopping", True)),
        es_patience=int(cfg["training"].get("early_stopping_patience", 8)),
        reduce_lr_patience=int(cfg["training"].get("reduce_lr_patience", 5)),
        min_lr=float(cfg["training"].get("min_lr", 1e-5)),
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg["training"].get("epochs", 30)),
        callbacks=cb_list,
        class_weight=class_weights if bool(cfg["training"].get("use_class_weight", True)) else None
    )

    model.save(final_path)
    plot_and_save_history(history, results_dir=results_dir, prefix="8class")

    if bool(cfg["training"].get("quick_test_eval", True)):
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"[Quick Test] loss={test_loss:.4f} acc={test_acc:.4f}")

    print("Saved:")
    print("  Best:", best_path)
    print("  Final:", final_path)
    print("  Results:", results_dir)


if __name__ == "__main__":
    main()
