"""
Image Forgery Detection - Ana Flask Uygulaması
Görüntü sahteciliği tespiti için web arayüzü
"""

from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Yüklenen dosyaların kaydedileceği klasör
UPLOAD_FOLDER = 'static/uploads'
# İzin verilen dosya formatları (US-1)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum 16MB


def allowed_file(filename):
    """Dosya formatının izin verilenler arasında olup olmadığını kontrol eder."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Görüntü yükleme endpoint'i"""
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'Dosya başarıyla yüklendi'
        })
    else:
        return jsonify({'error': 'İzin verilmeyen dosya formatı'}), 400


@app.route('/analyze', methods=['POST'])
def analyze():
    """Görüntü analiz endpoint'i - algoritma seçimine göre tespit yapar"""
    data = request.get_json()
    filename = data.get('filename')
    algorithm = data.get('algorithm', 'sift')

    if not filename:
        return jsonify({'error': 'Dosya adı belirtilmedi'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Dosya bulunamadı'}), 404

    if algorithm in ['sift', 'surf', 'akaze', 'orb']:
        from src.algorithms.detector import detect_forgery
        result = detect_forgery(filepath, algorithm)
    elif algorithm in ['cnn', 'lstm']:
        from src.ai_models.ai_detector import ai_detect
        result = ai_detect(filepath, algorithm)
    else:
        return jsonify({'error': 'Geçersiz algoritma'}), 400

    return jsonify(result)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)