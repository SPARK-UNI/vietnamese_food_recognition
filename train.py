# -*- coding: utf-8 -*-
"""
Train a simple CNN classifier (flowers/foods/anything in folders)
- Input: data_dir/<class>/<image>
- Split: train/val via validation_split
- Exports to out_dir:
    - food_cnn.h5 (best val_accuracy)
    - labels.txt (exact class order)
    - protocol.json (IMG_SIZE, color space, normalization, class_names, etc.)
"""

import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    print(f"[INFO] GPUs detected: {len(gpus)}" if gpus else "[INFO] No GPU detected; running on CPU.")


def build_cnn(input_shape, num_classes):
    """Plain CNN with in-model normalization to lock protocol."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Keep normalization INSIDE the model so inference doesn't need to guess
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"your_dataset",
                        help="Dataset folder with structure: data_dir/class/images")
    parser.add_argument("--out_dir", type=str, default="./artifacts",
                        help="Where to save model + labels + protocol")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    enable_gpu_memory_growth()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.val_split,
        subset="training",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=True,
        color_mode='rgb',
        label_mode='int'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=args.val_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
        color_mode='rgb',
        label_mode='int'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("[INFO] Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    # Build + compile model
    model = build_cnn((args.img_size, args.img_size, 3), num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Save best model
    model_path = out_dir / "food_cnn.h5"
    checkpoint_cb = ModelCheckpoint(
        str(model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    # Train
    print("[INFO] Training…")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb]
    )

    # Evaluate final
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"[RESULT] Final val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")

    # Export labels.txt
    labels_path = out_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"[INFO] Saved labels to: {labels_path}")

    # Export protocol.json
    protocol = {
        "version": 1,
        "framework": "tf.keras",
        "model_file": model_path.name,
        "img_size": [args.img_size, args.img_size],
        "channels": 3,
        "color_space": "RGB",
        "rescaling_in_model": True,  
        "class_names": class_names,
        "val_split": args.val_split,
        "seed": args.seed
    }
    protocol_path = out_dir / "protocol.json"
    with protocol_path.open("w", encoding="utf-8") as f:
        json.dump(protocol, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved protocol to: {protocol_path}")

    print(f"[DONE] Best model: {model_path}")


if __name__ == "__main__":
    main()
