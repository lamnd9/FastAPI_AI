# 🐶🐱 Cats vs Dogs Classifier API

Dự án API nhận diện Chó và Mèo sử dụng **FastAPI** kết hợp với mô hình học máy **MobileNetV2 (Transfer Learning)** bằng **TensorFlow/Keras**. 

Đây là dự án thực hành hoàn chỉnh từ việc chuẩn bị dataset, huấn luyện mô hình (Training) cho đến đóng gói thành một dịch vụ Web API chuyên nghiệp phục vụ Production.

---

## 🛠️ Công nghệ sử dụng
- **Backend & API**: FastAPI, Uvicorn, Pydantic v2 (Settings)
- **Deep Learning**: TensorFlow 2.15+ (tương thích Apple Silicon GPU Acceleration qua `tensorflow-metal`)
- **Xử lý ảnh**: Pillow, NumPy
- **Logging**: Loguru (Console + Rotating File Logs)
- **Quản lý môi trường**: Conda / Pip

---

## 📁 Cấu trúc dự án
```
fast_api_ai/
├── config/                  # ⚙️ Cấu hình hệ thống (Settings từ .env)
│   ├── __init__.py
│   └── settings.py          
├── logs/                    # 📝 Nhật ký hoạt động (Auto-rotate hàng ngày)
├── middleware/               # 🔗 Các bộ lọc (Logging request/response time)
│   ├── __init__.py
│   └── logging_middleware.py 
├── models/                  # 🧠 Nơi lưu trữ trọng số mô hình đã huấn luyện
│   └── cats_dogs_model.h5   # File model sau khi train
├── routes/                  # 🛣️ Định nghĩa API Endpoints
│   ├── __init__.py
│   ├── health.py            # Kiểm tra sức khỏe hệ thống
│   └── prediction.py        # API chính tiếp nhận và dự đoán ảnh
├── schemas/                 # 📋 Kiểm chuẩn dữ liệu đầu vào/đầu ra (Pydantic)
│   ├── __init__.py
│   └── prediction.py        
├── utils/                   # 🔧 Tiện ích bổ sung
│   ├── __init__.py
│   ├── image_processing.py  # Tiền xử lý ảnh (Resize 160x160, Normalize)
│   ├── logger.py            # Ghi log tập trung
│   └── model_loader.py      # Tải và lưu cache model trên RAM
├── dataset/                 # 📊 Bộ dữ liệu Chó & Mèo để huấn luyện
│   ├── training_set/        # Thư mục cats/ và dogs/ chứa ảnh train
│   └── test_set/            # Thư mục cats/ và dogs/ chứa ảnh test
├── app.py                   # 🚀 Khởi tạo FastAPI App
├── server.py                # 🖥️ Điểm khởi chạy uvicorn server
├── train.py                 # 🎓 Script huấn luyện model tự động
├── requirements.txt         # 📦 Danh sách thư viện cần thiết
├── .env.example             # 🔐 File cấu hình môi trường mẫu
└── README.md                # 📖 Tài liệu hướng dẫn (File này)
```

---

## 🚀 Hướng dẫn cài đặt và chạy dự án

### 1. Chuẩn bị môi trường huấn luyện
Kích hoạt môi trường ảo Conda chứa TensorFlow phù hợp với hệ máy của bạn (ở đây sử dụng môi trường `FoodRecognition_env` có hỗ trợ Apple Silicon):

```bash
conda activate FoodRecognition_env
```

Cài đặt các gói thư viện FastAPI bổ sung:
```bash
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình (Training)
Đảm bảo bộ dữ liệu đã được đặt đúng cấu trúc trong thư mục `dataset/`. Chạy script huấn luyện:

```bash
python train.py
```
- Quá trình train sẽ chạy qua **5 epochs** sử dụng mạng nền tảng MobileNetV2 (đã được đóng băng các lớp trích xuất đặc trưng chính và chỉ train lớp phân loại Chó/Mèo ở đầu ra).
- Kết quả huấn luyện (file model `cats_dogs_model.h5`) sẽ được lưu vào thư mục `models/`.
- File nhãn `cats_dogs_classes.txt` và biểu đồ lịch sử huấn luyện `logs/training_history.png` cũng sẽ tự động được tạo ra.

### 3. Cấu hình môi trường chạy API
Copy file cấu hình mẫu `.env.example` thành `.env` để sử dụng:

```bash
cp .env.example .env
```

Kiểm tra nội dung file `.env` để đảm bảo trỏ đúng model Chó & Mèo:
```env
APP_NAME=Cats vs Dogs Classifier API
MODEL_FILE=cats_dogs_model.h5
CLASSES_PATH=cats_dogs_classes.txt
IMAGE_SIZE=160
```

### 4. Khởi chạy Web API Service
Khởi động Uvicorn server:

```bash
python server.py
```

API sẽ khởi chạy tại địa chỉ: `http://localhost:8000`

---

## 🔍 Kiểm tra & Trải nghiệm API (Swagger UI)
Sau khi server chạy, truy cập đường dẫn sau trên trình duyệt để tương tác trực quan:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

1. Chọn API `POST /api/v1/predict` (Dự đoán Chó hoặc Mèo từ ảnh gửi lên).
2. Bấm **"Try it out"**.
3. Upload 1 ảnh bất kỳ của chú chó hoặc chú mèo của bạn.
4. Bấm **"Execute"** để nhận kết quả JSON trả về.

**Ví dụ định dạng phản hồi từ API:**
```json
{
  "prediction": "dogs",
  "confidence": 0.9854,
  "top_5": [
    {
      "class_name": "dogs",
      "probability": 0.9854
    },
    {
      "class_name": "cats",
      "probability": 0.0146
    }
  ]
}
```

---
*Chúc chú có những trải nghiệm tuyệt vời khi học lập trình AI và Web Service!* 🐾
