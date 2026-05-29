import os
import tensorflow as tf
from src.ai_models.ai_detector import build_cnn_model

DATASET_DIR = "dataset" 
MODEL_SAVE_DIR = "models"
BATCH_SIZE = 32
IMG_SIZE = (256, 256) # 256 idealdir
EPOCHS = 25

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def train_cnn():
    print("Veriler okunuyor, bu işlem biraz sürebilir...")
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    # İŞTE HAYAT KURTARAN DÜZELTME (Normalizasyon)
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Eğitimi hızlandırmak için RAM'e alma (Cache)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("\n--- CNN Model Eğitimi Başlıyor ---")
    model = build_cnn_model()
    
    # Sadece en iyi modeli kaydetmesi ve gelişmiyorsa durması için (Early Stopping)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_DIR, "cnn_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    print(f"\nİşlem Tamam! Model kaydedildi.")

if __name__ == "__main__":
    train_cnn()