import os
import tensorflow as tf
from src.ai_models.ai_detector import build_cnn_model

DATASET_DIR = "dataset" 
MODEL_SAVE_DIR = "models"
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
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
        batch_size=BATCH_SIZE
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    print("\n--- Model Eğitimi Başlıyor ---")
    model = build_cnn_model()
    
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    model_path = os.path.join(MODEL_SAVE_DIR, "cnn_model.keras")
    model.save(model_path)
    print(f"\nİşlem Tamam! Model kaydedildi: {model_path}")

if __name__ == "__main__":
    train_cnn()