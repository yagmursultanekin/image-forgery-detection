import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
import tensorflow as tf

DATASET_DIR = "dataset"
MODEL_SAVE_DIR = "models"
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 50
SEQ_LENGTH = 16  # Her görüntü 16 parçaya bölünür (satır satır sequence)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def build_lstm_model():
    """
    CNN feature extractor + LSTM classifier.
    Görüntüyü satırlara bölerek sequence olarak işler.
    """
    model = models.Sequential([
        # Giriş: (128, 128, 3) → satırları sequence olarak işle
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        # Her satırı flatten et → (128, 128*3) = (128, 384)
        layers.Reshape((IMG_SIZE[0], IMG_SIZE[1] * 3)),

        # LSTM katmanları
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        # Sınıflandırma başlığı
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),  # 0=Au (gerçek), 1=Tp (sahte)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_lstm():
    print("Veriler okunuyor...")

    # Dataset klasör isimleri: Au=0 (gerçek), Tp=1 (sahte)
    # Keras alfabetik sıraya göre label atar: Au→0, Tp→1 ✓
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
    )

    # Normalizasyon [0,255] → [0,1]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Performans için prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Callback'ler
    callbacks = [
        # En iyi modeli kaydet
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "lstm_model_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Eğitim duruyorsa early stop
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate düşür
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    print("\n--- LSTM Model Eğitimi Başlıyor ---")
    print(f"Label mapping: Au=0 (gerçek), Tp=1 (sahte)\n")

    model = build_lstm_model()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Son modeli de kaydet
    final_path = os.path.join(MODEL_SAVE_DIR, "lstm_model.keras")
    model.save(final_path)
    print(f"\nİşlem Tamam! Model kaydedildi: {final_path}")
    print(f"En iyi model: {MODEL_SAVE_DIR}/lstm_model_best.keras")


if __name__ == "__main__":
    train_lstm()