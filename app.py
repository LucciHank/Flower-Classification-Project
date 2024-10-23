from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

# Tải mô hình
model = tf.keras.models.load_model('VGG16_model.keras')

img_height, img_width = 224, 224

# Tên các loài hoa và thông tin chi tiết
class_names = ['Cúc', 'Bồ Công Anh', 'Hoa Hồng', 'Hướng Dương', 'Tulip']

flower_info = {
    'Cúc': {'Nguồn gốc': 'Cúc có nguồn gốc từ châu Âu và châu Á.', 'Đặc điểm nhận dạng': 'Hoa nhỏ, màu trắng hoặc vàng, có tâm màu vàng.', 'Kích thước': 'Chiều cao khoảng 20-30 cm.', 'Thông tin thêm': 'Cúc biểu tượng cho sự trong sáng và thuần khiết.'},
    'Bồ Công Anh': {'Nguồn gốc': 'Bồ Công Anh phân bố rộng khắp vùng ôn đới châu Âu và Bắc Mỹ.', 'Đặc điểm nhận dạng': 'Hoa nhỏ, màu vàng, nở thành cụm tròn.', 'Kích thước': 'Chiều cao khoảng 10-30 cm.', 'Thông tin thêm': 'Hạt bồ công anh được phát tán nhờ gió, biểu tượng cho tự do.'},
    'Hoa Hồng': {'Nguồn gốc': 'Hoa hồng có nguồn gốc từ châu Á.', 'Đặc điểm nhận dạng': 'Cánh hoa dày, màu sắc đa dạng, thường có gai trên thân.', 'Kích thước': 'Chiều cao từ 50-200 cm.', 'Thông tin thêm': 'Hoa hồng là biểu tượng của tình yêu và sự lãng mạn.'},
    'Hướng Dương': {'Nguồn gốc': 'Hướng dương có nguồn gốc từ Bắc Mỹ.', 'Đặc điểm nhận dạng': 'Hoa lớn, màu vàng tươi, có tâm màu nâu.', 'Kích thước': 'Chiều cao từ 1-3 mét.', 'Thông tin thêm': 'Hoa hướng dương biểu tượng cho sự lạc quan và hướng tới ánh sáng.'},
    'Tulip': {'Nguồn gốc': 'Tulip có nguồn gốc từ Trung Đông và châu Âu.', 'Đặc điểm nhận dạng': 'Cánh hoa hình tròn hoặc oval, nhiều màu sắc rực rỡ.', 'Kích thước': 'Chiều cao từ 10-70 cm.', 'Thông tin thêm': 'Tulip là biểu tượng của sự thịnh vượng và tình yêu viên mãn.'}
}

app = Flask(__name__)

def predict_image(image):
    # Xử lí hình ảnh
    img = cv2.resize(image, (img_height, img_width))
    img = np.array(img).reshape(-1, img_height, img_width, 3)
    img = img / 255.0

    # Dự đoán
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    # Lấy thông tin chi tiết về loài hoa
    flower_details = flower_info.get(predicted_class_name, "Không có thông tin")

    # Trả về tên hoa và thông tin chi tiết
    result = {
        "name": predicted_class_name,
        "details": flower_details
    }
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Đọc ảnh và chuyển về dạng NumPy array
    image = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Dự đoán
    result = predict_image(image)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
