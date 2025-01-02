import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

# Đường dẫn đến mô hình đã lưu và thư mục chứa các hình ảnh cần dự đoán
model_path = 'vgg16_fruit_classifier.h5'
predict_dir = 'data/Predict'

# Tải mô hình đã huấn luyện
model = load_model(model_path)

# Từ điển ánh xạ từ chỉ số đến tên nhãn
label_to_index = {'apple': 0, 'avocado': 1, 'banana': 2, 'cherry': 3, 'kiwi': 4,
                  'mango': 5, 'orange': 6, 'pinenapple': 7, 'strawberries': 8, 'watermelon': 9}
index_to_label = {v: k for k, v in label_to_index.items()}


# Hàm để dự đoán tên quả từ hình ảnh
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# Lấy 10 ảnh đầu tiên trong thư mục Predict
image_files = [f for f in os.listdir(predict_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]

# Tạo một lưới 2x5 để hiển thị 10 ảnh
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, filename in enumerate(image_files):
    file_path = os.path.join(predict_dir, filename)
    predicted_label = predict_image(file_path)

    # Hiển thị hình ảnh và nhãn dự đoán
    img = load_img(file_path, target_size=(224, 224))
    ax = axes[i // 5, i % 5]  # Tính toán vị trí của ảnh trên lưới 2x5
    ax.imshow(img)
    ax.set_title(f"Dự đoán: {predicted_label}")
    ax.axis('off')  # Tắt hiển thị trục

plt.tight_layout()
plt.show()
