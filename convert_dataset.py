import os
from PIL import Image

DATASET_DIR = "dataset"
converted_count = 0
deleted_count = 0

print("Dönüştürme işlemi başlıyor. Bilgisayarının hızına göre 1-2 dakika sürebilir...")

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        ext = os.path.splitext(file)[1].lower()
        
        # 1. TIF dosyalarını bul ve JPG'ye çevir
        if ext in ['.tif', '.tiff']:
            try:
                # Görüntüyü aç
                with Image.open(file_path) as img:
                    # JPG formatı saydamlık (alpha) desteklemediği için RGB'ye çevirmek şart
                    rgb_im = img.convert('RGB')
                    
                    # Yeni JPG dosyasının adını oluştur
                    new_file_path = os.path.splitext(file_path)[0] + '.jpg'
                    
                    # Kaliteyi bozmadan JPG olarak kaydet
                    rgb_im.save(new_file_path, 'JPEG', quality=100)
                
                # Eski TIF dosyasını sil (Alan kaplamasın)
                os.remove(file_path)
                converted_count += 1
                
                # Her 500 dosyada bir ekrana bilgi ver
                if converted_count % 500 == 0:
                    print(f"{converted_count} adet .tif başarıyla .jpg yapıldı...")
                    
            except Exception as e:
                print(f"Hata ({file}): {str(e)}")
        
        # 2. Gereksiz .db dosyalarını temizle
        elif ext == '.db':
            try:
                os.remove(file_path)
                deleted_count += 1
            except:
                pass

print(f"\n🎉 İşlem Tamamlandı!")
print(f"✅ Dönüştürülen TIF sayısı: {converted_count}")
print(f"🗑️ Silinen gereksiz dosya sayısı: {deleted_count}")
print("\nArtık eğitim kodlarını çalıştırdığında TensorFlow 10.000'den fazla dosyayı okuyacak!")