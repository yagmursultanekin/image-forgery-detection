"""
@file ai_detector.py
@brief Görüntü Sahteciliği Tespiti - Yapay Zeka Modelleri
@details CNN ve LSTM derin öğrenme algoritmaları ile sahtecilik tespiti
@author Yağmur Sultan Ekin
@version 1.0
"""

import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import os


def build_cnn_model():
    """
    @brief CNN (Evrişimli Sinir Ağı) modeli oluşturur.
    @details Görüntü sahteciliği tespiti için 3 katmanlı CNN mimarisi.
             Conv2D, MaxPooling, Dense ve Dropout katmanları içerir.
    
    @return keras.Sequential Derlenmiş CNN modeli
    """
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
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def build_lstm_model():
    """
    @brief LSTM (Uzun Kısa Süreli Bellek) modeli oluşturur.
    @details Görüntü sahteciliği tespiti için 2 katmanlı LSTM mimarisi.
             Görüntü satırları sıralı veri olarak işlenir.
    
    @return keras.Sequential Derlenmiş LSTM modeli
    """
    model = keras.Sequential([
        layers.Input(shape=(128, 128)),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def preprocess_image_cnn(image_path):
    """
    @brief CNN modeli için görüntü ön işleme yapar.
    @details Görüntüyü 128x128 boyutuna getirir ve normalize eder.
    
    @param image_path Görüntü dosyasının yolu (string)
    @return numpy.ndarray İşlenmiş görüntü dizisi veya None
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image_lstm(image_path):
    """
    @brief LSTM modeli için görüntü ön işleme yapar.
    @details Görüntüyü gri tona çevirir, 128x128 boyutuna getirir ve normalize eder.
    
    @param image_path Görüntü dosyasının yolu (string)
    @return numpy.ndarray İşlenmiş görüntü dizisi veya None
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def ai_detect(image_path, algorithm='cnn'):
    """
    @brief Yapay zeka modeli ile sahtecilik tespiti yapar.
    @details CNN veya LSTM modeli kullanarak görüntüdeki sahteciliği tespit eder.
             Model eğitilmemiş olsa dahi mimari olarak çalışır.
    
    @param image_path Görüntü dosyasının yolu (string)
    @param algorithm Kullanılacak model: 'cnn' veya 'lstm'
    
    @return dict Tespit sonuçlarını içeren sözlük:
            - algorithm: Kullanılan model adı
            - forged: Sahtecilik tespit edildi mi (bool)
            - confidence: Güven oranı yüzde olarak (float)
            - message: Sonuç mesajı (string)
            - note: Ek bilgi (string)
    """
    try:
        if algorithm == 'cnn':
            model = build_cnn_model()
            image = preprocess_image_cnn(image_path)
            algorithm_name = 'CNN'
        elif algorithm == 'lstm':
            model = build_lstm_model()
            image = preprocess_image_lstm(image_path)
            algorithm_name = 'LSTM'
        else:
            return {'error': 'Geçersiz AI algoritması'}

        if image is None:
            return {'error': 'Görüntü yüklenemedi'}

        prediction = model.predict(image, verbose=0)
        confidence = float(prediction[0][0]) * 100
        forged = confidence > 50.0

        return {
            'algorithm': algorithm_name,
            'forged': forged,
            'confidence': round(confidence, 2),
            'message': 'Sahtecilik tespit edildi!' if forged else 'Görüntü orijinal görünüyor',
            'note': 'Model eğitilmemiş - demo sonuçları gösteriliyor'
        }

    except Exception as e:
        return {'error': f'AI analiz hatası: {str(e)}'}