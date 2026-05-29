import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Klasör ayarları
DATASET_DIR = "dataset"
MODELS_DIR = "models"
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

def evaluate_model(model_name, model_file):
    model_path = os.path.join(MODELS_DIR, model_file)
    if not os.path.exists(model_path):
        print(f"❌ {model_name} modeli bulunamadı: {model_path}")
        return

    print(f"\n{'='*40}")
    print(f"[{model_name}] Modeli Değerlendiriliyor...")
    print(f"{'='*40}")
    
    # Modeli yükle
    model = tf.keras.models.load_model(model_path)

    # Eğitimdeki seed(123) ve split(0.2) oranını aynı tutuyoruz ki 
    # modelin daha önce hiç görmediği doğrulama (validation) verilerini test edelim.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )
    
    class_names = val_ds.class_names # Genelde ['Au', 'Tp'] olur

    print(f"[{model_name}] Tahminler yapılıyor, lütfen bekleyin...")
    y_true = []
    y_pred_probs = []

    # Verileri batch (grup) halinde işleyip gerçek etiketleri ve tahminleri topluyoruz
    for x_batch, y_batch in val_ds:
        y_true.extend(y_batch.numpy().flatten())
        
        # LSTM eğitiminde verileri /255 ile normalize etmiştin, değerlendirirken de etmeliyiz
        if model_name == "LSTM":
            x_batch = x_batch / 255.0
            
        preds = model.predict(x_batch, verbose=0)
        y_pred_probs.extend(preds.flatten())

    # Değerleri numpy dizisine çevir (Olasılığı 0.5'ten büyük olanlara 1 de)
    y_true = np.array(y_true, dtype=int)
    y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

    # 1. Sınıflandırma Raporu (Precision, Recall, F1-Score)
    print(f"\n--- {model_name} Detaylı Rapor ---")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 2. Doğruluk Matrisi (Confusion Matrix) Hesaplama ve Çizimi
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    
    plt.title(f"{model_name} Doğruluk Matrisi")
    plt.xlabel("Yapay Zekanın Tahmini")
    plt.ylabel("Gerçekte Olan")
    
    # Grafiği kaydet
    save_path = f"{model_name.lower()}_confusion_matrix.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Görsel Kaydedildi: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Yapay Zeka Performans Analizi Başlıyor...\n")
    # CNN ve LSTM modellerini sırayla test et
    evaluate_model("CNN", "cnn_model.keras")
    evaluate_model("LSTM", "lstm_model_best.keras")
    print("\nTüm analizler tamamlandı!")