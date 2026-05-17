"""
Görüntü Sahteciliği Tespiti - AI Modeller
CNN ve LSTM algoritmaları ile tespit
"""

import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import os


def build_cnn_model():
    """CNN modeli oluşturur."""
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
    """LSTM modeli oluşturur."""
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
    """CNN için görüntüyü hazırlar."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image_lstm(image_path):
    """LSTM için görüntüyü hazırlar."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def ai_detect(image_path, algorithm='cnn'):
    """
    AI modeli ile sahtecilik tespiti yapar.

    Args:
        image_path: Görüntü dosyasının yolu
        algorithm: Kullanılacak model (cnn veya lstm)

    Returns:
        dict: Tespit sonuçları
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

        # Tahmin yap
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