"""
@file detector.py
@brief Görüntü Sahteciliği Tespiti - Klasik Algoritmalar
@details SIFT, SURF, AKAZE, ORB algoritmaları ile Copy-Move sahtecilik tespiti
@author Yağmur Sultan Ekin
@version 1.1
"""

import cv2
import numpy as np


def detect_forgery(image_path, algorithm='sift'):
    """
    @brief Verilen görüntüde sahtecilik tespiti yapar.
    @details Bu fonksiyon SIFT, SURF, AKAZE veya ORB algoritmalarından birini
             kullanarak görüntüdeki Copy-Move sahteciliğini tespit eder.
             Anahtar nokta eşleştirme yöntemi kullanılır.

    @param image_path Görüntü dosyasının yolu (string)
    @param algorithm Kullanılacak algoritma: 'sift', 'surf', 'akaze', 'orb'

    @return dict Tespit sonuçlarını içeren sözlük:
            - algorithm: Kullanılan algoritma adı
            - forged: Sahtecilik tespit edildi mi (bool)
            - confidence: Güven oranı (float)
            - keypoints_found: Bulunan anahtar nokta sayısı (int)
            - matches_found: Eşleşme sayısı (int)
            - message: Sonuç mesajı (string)

    @note SURF algoritması patentli olduğundan bazı sistemlerde çalışmayabilir.
    """
    image = cv2.imread(image_path)
    if image is None:
        return {'error': 'Görüntü yüklenemedi'}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if algorithm == 'sift':
        detector = cv2.SIFT_create()
        algorithm_name = 'SIFT'
    elif algorithm == 'surf':
        try:
            detector = cv2.xfeatures2d.SURF_create(400)
            algorithm_name = 'SURF'
        except:
            return {'error': 'SURF algoritması bu sistemde desteklenmiyor'}
    elif algorithm == 'akaze':
        detector = cv2.AKAZE_create()
        algorithm_name = 'AKAZE'
    elif algorithm == 'orb':
        detector = cv2.ORB_create(nfeatures=1000)
        algorithm_name = 'ORB'
    else:
        return {'error': 'Geçersiz algoritma'}

    keypoints, descriptors = detector.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 10:
        return {
            'algorithm': algorithm_name,
            'forged': False,
            'confidence': 0.0,
            'keypoints_found': len(keypoints) if keypoints else 0,
            'matches_found': 0,
            'message': 'Yeterli anahtar nokta bulunamadı'
        }

    if algorithm == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    matches = bf.knnMatch(descriptors, descriptors, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            # queryIdx != trainIdx: kendisiyle eşleşmeyi engelle
            if m.distance < 0.75 * n.distance and m.queryIdx != m.trainIdx:
                good_matches.append(m)

    num_keypoints = len(keypoints)
    num_matches = len(good_matches)
    match_ratio = num_matches / num_keypoints  # 0.0 - 1.0 arası

    # Sahtecilik kararı: hem mutlak eşleşme sayısı hem oran yüksek olmalı
    forged = num_matches > 10 and match_ratio > 0.05

    if forged:
        # Sahte: eşleşme oranı arttıkça güven artar, max %95
        # match_ratio 0.05 → ~%30, 0.5 → ~%95
        raw = match_ratio * 180
        confidence = min(round(raw, 2), 95.0)
        confidence = max(confidence, 30.0)
    else:
        # Orijinal: az eşleşme = daha güvenilir
        # match_ratio 0.00 → %90, 0.04 → ~%60, 0.05 eşiğine yaklaştıkça düşer
        raw = 90.0 - (match_ratio * 600)
        confidence = min(round(raw, 2), 90.0)
        confidence = max(confidence, 30.0)

        print(f"DEBUG - keypoints: {num_keypoints}, matches: {num_matches}, ratio: {match_ratio:.4f}, forged: {forged}, confidence: {confidence}")

    return {
        'algorithm': algorithm_name,
        'forged': forged,
        'confidence': confidence,
        'keypoints_found': num_keypoints,
        'matches_found': num_matches,
        'message': 'Sahtecilik tespit edildi!' if forged else 'Görüntü orijinal görünüyor'
    }