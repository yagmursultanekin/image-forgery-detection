import os
from collections import Counter

DATASET_DIR = "dataset"
uzantilar = []

print("Klasör taranıyor...")
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        # Dosyanın uzantısını alıp küçük harfe çeviriyoruz
        uzanti = os.path.splitext(file)[1].lower()
        uzantilar.append(uzanti)

print("\n📊 Veri setindeki dosya türleri ve sayıları:")
for uzanti, sayi in Counter(uzantilar).items():
    print(f"{uzanti if uzanti else 'Uzantısız Dosya'}: {sayi} adet")