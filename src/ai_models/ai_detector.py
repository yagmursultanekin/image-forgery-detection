"""
@file ai_detector.py
@brief Görüntü Sahteciliği Tespiti - Yapay Zeka ve Geleneksel Modeller
@details CNN, LSTM ve Geleneksel (SIFT, AKAZE, vb.) algoritmalar ile sahtecilik tespiti
@author Yağmur Sultan Ekin
@version 1.1
"""

import numpy as np
import cv2
import os
import keras
from keras import layers, models

# ==========================================
# 1. YAPAY ZEKA MODELLERİ (CNN & LSTM)
# ==========================================

def build_cnn_model():
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model():
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Reshape((128, 128 * 3)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image_cnn(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_image_lstm(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def ai_detect(image_path, algorithm='cnn'):
    try:
        model = None
        if algorithm == 'cnn':
            model_path = os.path.join('models', 'cnn_model.keras')
            if os.path.exists(model_path): model = keras.models.load_model(model_path)
            else: return {'error': 'Eğitilmiş CNN modeli bulunamadı!'}
            image = preprocess_image_cnn(image_path)
            algorithm_name = 'CNN'
            
        elif algorithm == 'lstm':
            model_path = os.path.join('models', 'lstm_model_best.keras')
            if os.path.exists(model_path): model = keras.models.load_model(model_path)
            else: return {'error': 'Eğitilmiş LSTM modeli bulunamadı!'}
            image = preprocess_image_lstm(image_path)
            algorithm_name = 'LSTM'
        else:
            return {'error': 'Geçersiz AI algoritması'}

        if image is None: return {'error': 'Görüntü yüklenemedi.'}

        prediction = model.predict(image, verbose=0)
        confidence = float(prediction[0][0]) * 100
        forged = confidence > 50.0
        display_confidence = confidence if forged else (100.0 - confidence)

        return {
            'algorithm': algorithm_name,
            'forged': forged,
            'confidence': round(display_confidence, 2),
            'message': 'Sahtecilik tespit edildi!' if forged else 'Görüntü orijinal görünüyor',
            'note': f'Eğitilmiş {algorithm_name} modeli sonucu'
        }
    except Exception as e:
        return {'error': f'AI analiz hatası: {str(e)}'}

# ==========================================
# 2. GELENEKSEL ALGORİTMALAR (KOPYALA-YAPIŞTIR TESPİTİ)
# ==========================================

def feature_based_detect(image_path, algorithm='akaze'):
    """
    @brief SIFT, SURF, AKAZE veya ORB ile Copy-Move (Kopyala-Yapıştır) sahteciliği tespiti.
    @details Görüntüyü kendi kendisiyle eşleştirerek klonlanmış bölgeleri arar.
    """
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return {'error': 'Görüntü yüklenemedi.'}

        # Algoritma seçimi
        algorithm = algorithm.lower()
        if algorithm == 'sift':
            detector = cv2.SIFT_create()
            norm_type = cv2.NORM_L2
        elif algorithm == 'surf':
            try:
                # SURF patentli olduğu için OpenCV'nin bazı sürümlerinde çalışmayabilir
                detector = cv2.xfeatures2d.SURF_create()
                norm_type = cv2.NORM_L2
            except AttributeError:
                return {'error': 'Kullandığınız OpenCV sürümünde SURF patent nedeniyle kapalı. SIFT veya AKAZE deneyin.'}
        elif algorithm == 'akaze':
            detector = cv2.AKAZE_create()
            norm_type = cv2.NORM_HAMMING
        elif algorithm == 'orb':
            detector = cv2.ORB_create()
            norm_type = cv2.NORM_HAMMING
        else:
            return {'error': 'Geçersiz algoritma (sift, surf, akaze, orb desteklenir).'}

        # Özellik Çıkarımı
        keypoints, descriptors = detector.detectAndCompute(image, None)

        if descriptors is None or len(descriptors) < 20:
            return {
                'algorithm': algorithm.upper(),
                'forged': False,
                'confidence': 100.0,
                'message': 'Görüntüde yeterli özellik noktası bulunamadı, orijinal varsayıldı.',
                'matches': 0
            }

        # Kendi Kendisiyle Eşleştirme (k=3 alıyoruz ki 1. noktanın kendisi olmasını ekarte edelim)
        bf = cv2.BFMatcher(norm_type)
        matches = bf.knnMatch(descriptors, descriptors, k=3)

        good_matches = []
        for match_tuple in matches:
            # En az 3 eşleşme dönmeli (kendisi, en iyi eşleşme, ikinci en iyi eşleşme)
            if len(match_tuple) >= 3:
                m, n, k = match_tuple
                
                # m, noktanın kendisidir (mesafesi 0'dır). Biz n ve k'yi kıyaslayacağız.
                if n.distance < 0.75 * k.distance: # Lowe's Ratio Test
                    pt1 = keypoints[n.queryIdx].pt
                    pt2 = keypoints[n.trainIdx].pt
                    
                    # Fiziksel Mesafe Kontrolü (Yan yana duran piksellerin eşleşmesini engelleriz)
                    # Aralarında en az 50 piksel mesafe olan benzer yapılar arıyoruz (Kopyala-yapıştır)
                    dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
                    if dist > 50: 
                        good_matches.append(n)

        match_count = len(good_matches)
        
        # Karar Mekanizması (Eğer 10'dan fazla uzak ve benzer nokta varsa sahtedir)
        forged = match_count > 10 

        # Eşleşme sayısına göre bir güven skoru oluşturma
        if forged:
            # 10 eşleşme %60, 30 eşleşme %100 güven gibi
            confidence = min(100.0, 50.0 + (match_count * 2))
            message = f'Copy-Move (Kopyala-Yapıştır) Sahteciliği Tespit Edildi! ({match_count} kopya eşleşme)'
        else:
            # Çok az kopya eşleşme varsa güvende
            confidence = max(50.0, 100.0 - (match_count * 5))
            message = 'Görüntü orijinal görünüyor (Klonlanmış bölge bulunamadı).'

        return {
            'algorithm': algorithm.upper(),
            'forged': forged,
            'confidence': round(confidence, 2),
            'message': message,
            'note': f'Bulunan anahtar nokta: {len(keypoints)} | Eşleşme: {match_count}'
        }

    except Exception as e:
        return {'error': f'Geleneksel algoritma hatası: {str(e)}'}