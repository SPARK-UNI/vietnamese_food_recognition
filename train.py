import os, math, numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow_addons as tfa

# ====== Cấu hình ======
DATA_DIR = r"your_data_path"  # Thư mục data với cấu trúc: data_dir/class/images
CLASSES = ['bun bo', 'com tam', 'hu tieu', 'my quang', 'pho']
IMG_SIZE = 384                  
BATCH_SIZE = 32                     
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 12
EPOCHS_STAGE3 = 10
SEED = 1337
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = len(CLASSES)

# ====== Dataset loader với split trong-directory ======
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    class_names=CLASSES,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset='training'
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    labels='inferred',
    label_mode='categororical' if False else 'categorical',
    class_names=CLASSES,
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
    seed=SEED,
    validation_split=0.2,
    subset='validation'
)

# ====== Augmentations (mạnh để chống lệ thuộc "cái bát") ======
data_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1, fill_mode="nearest"),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
    layers.RandomContrast(0.25),
    layers.GaussianNoise(0.05),
    # Brightness/Saturation ngẫu nhiên giúp chống phụ thuộc ánh sáng/màu
    layers.Lambda(lambda x: tf.image.random_brightness(x, 0.1)),
    layers.Lambda(lambda x: tf.image.random_saturation(x, 0.7, 1.3)),
], name="strong_aug")

# ====== MixUp (giảm overfitting, tăng tính "khó") ======
def sample_beta_distribution(size, alpha):
    gamma_1 = tf.random.gamma(shape=[size], alpha=alpha)
    gamma_2 = tf.random.gamma(shape=[size], alpha=alpha)
    return gamma_1 / (gamma_1 + gamma_2)

def mixup_batch(images, labels, alpha=0.2):
    # images: (B,H,W,3), labels: (B,C)
    batch_size = tf.shape(images)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    shuffled_images = tf.gather(images, index)
    shuffled_labels = tf.gather(labels, index)

    lam = sample_beta_distribution(batch_size, alpha)  # (B,)
    lam_x = tf.reshape(lam, (batch_size, 1, 1, 1))
    lam_y = tf.reshape(lam, (batch_size, 1))

    mixed_images = images * lam_x + shuffled_images * (1.0 - lam_x)
    mixed_labels = labels * lam_y + shuffled_labels * (1.0 - lam_y)
    return mixed_images, mixed_labels

# ====== Preprocess theo EfficientNetV2 ======
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

def train_map(images, labels):
    images = tf.cast(images, tf.float32)  # 0..255
    images = data_aug(images)
    # 50% batch dùng MixUp (có thể tăng lên 0.7 nếu vẫn overfit)
    images, labels = tf.cond(
        tf.random.uniform(()) < 0.5,
        lambda: mixup_batch(images, labels, alpha=0.2),
        lambda: (images, labels)
    )
    images = preprocess_input(images)     # [-1,1]
    return images, labels

def val_map(images, labels):
    images = preprocess_input(tf.cast(images, tf.float32))
    return images, labels

train_ds = train_ds.map(train_map, num_parallel_calls=AUTO).prefetch(AUTO)
val_ds   = val_ds.map(val_map,   num_parallel_calls=AUTO).prefetch(AUTO)

# ====== Model: EfficientNetV2S + head được regularize ======
base = tf.keras.applications.EfficientNetV2S(
    include_top=False, weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg'
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='swish',
                 kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# ====== Optimizers, losses, callbacks ======
loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
opt1 = tfa.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "food5_best.h5", monitor="val_loss", save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
    ),
]

model.compile(optimizer=opt1, loss=loss, metrics=["accuracy"])
print(model.summary())

# ====== Stage 1: train head ======
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1, callbacks=callbacks)

# ====== Stage 2: unfreeze ~40% cuối của backbone ======
base.trainable = True
n = len(base.layers)
for i, layer in enumerate(base.layers):
    layer.trainable = (i >= int(n*0.60))  # freeze 60% đầu

opt2 = tfa.optimizers.AdamW(learning_rate=5e-5, weight_decay=5e-5)
model.compile(optimizer=opt2, loss=loss, metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2, callbacks=callbacks)

# ====== Stage 3: unfreeze toàn bộ (LR rất thấp) ======
for layer in base.layers:
    layer.trainable = True

opt3 = tfa.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-5)
model.compile(optimizer=opt3, loss=loss, metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE3, callbacks=callbacks)

# ====== Đánh giá chi tiết trên validation ======
# (sử dụng val_ds gốc trước preprocess để lấy y_true chính xác theo thứ tự)
# Ở đây chúng ta dùng val_ds đã map preprocess nên cần thu y_true/y_pred sau predict.
y_true = []
for _, y in val_ds:
    y_true.append(y.numpy())
y_true = np.concatenate(y_true, axis=0).argmax(axis=1)

y_pred = model.predict(val_ds).argmax(axis=1)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))
