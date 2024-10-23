import gdown

# Đường dẫn Google Drive chia sẻ mô hình
url = 'https://drive.google.com/file/d/1U7FBGMfSOpocxQCQ4-JJYGtsBoN-AmKg/view?usp=sharing'
output = 'VGG16_model.keras'

# Tải mô hình xuống máy cục bộ
gdown.download(url, output, quiet=False)
