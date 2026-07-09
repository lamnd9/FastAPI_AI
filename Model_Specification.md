# Tài liệu Mô tả & Hướng dẫn Tích hợp Model EfficientNetV2

Tài liệu này đặc tả chi tiết đầu vào (Input), đầu ra (Output), cách tiền xử lý dữ liệu và cung cấp khung mã nguồn hoàn chỉnh để chú xây dựng một dịch vụ API bằng **FastAPI** phục vụ cho việc phát triển ứng dụng nhận diện món ăn.

---

## 1. Thông số Kỹ thuật của Model (`efficientnetv2_product_model.hdf5`)

| Thông số | Đặc tả chi tiết |
| :--- | :--- |
| **Cấu trúc mạng (Architecture)** | EfficientNetV2 (Tối ưu hóa về tốc độ và dung lượng hơn bản V1) |
| **Mức độ chi tiết (Granularity)** | **Product** (Nhận diện món ăn cụ thể) |
| **Số lượng phân loại (Classes)** | **840** lớp món ăn khác nhau (danh sách chi tiết nằm ở `src/product_classes.txt`) |
| **Kích thước file weights** | ~248 MB |
| **Định dạng file** | Keras HDF5 (`.hdf5`) |

---

## 2. Đặc tả Input (Đầu vào)

Mô hình yêu cầu ảnh đầu vào phải được chuyển đổi về dạng ma trận (tensor) 4 chiều trước khi đưa vào dự đoán.

* **Kích thước ảnh yêu cầu (Resolution)**: `224x224` pixel.
* **Số kênh màu (Channels)**: `3` kênh màu (RGB).
* **Kiểu dữ liệu (Data Type)**: `float32`.
* **Chuẩn hóa (Normalization)**: Giá trị pixel phải được chuẩn hóa về khoảng **`[0.0, 1.0]`** bằng cách chia giá trị màu gốc (`0-255`) cho `255.0`.
* **Hình dáng Tensor đầu vào (Input Shape)**: `(1, 224, 224, 3)`.
  * `1`: Batch size (số lượng ảnh xử lý cùng lúc - ở đây là 1 ảnh).
  * `224, 224`: Chiều cao và chiều rộng của ảnh.
  * `3`: Ba kênh màu Đỏ, Xanh lá, Xanh dương (RGB).

---

## 3. Đặc tả Output (Đầu ra)

Mô hình trả về một mảng số thực tương ứng với các lớp (classes).

* **Hình dáng Tensor đầu ra (Output Shape)**: `(1, 840)`.
* **Bản chất của mảng trả về**: Mảng chứa các giá trị **Logits (điểm thô)**, có thể âm hoặc dương và không có giới hạn khoảng giá trị.
* **Cách chuyển đổi thành Phần trăm xác suất (Probability %)**:
  * Phải áp dụng hàm **Softmax** lên mảng Logits này để chuyển đổi toàn bộ giá trị thành dạng phần trăm xác suất dương chạy từ `0%` đến `100%`, tổng cả 840 phần tử bằng đúng `100%`.
* **Cách lấy tên món ăn cụ thể**:
  * Tìm index của phần tử có giá trị lớn nhất sau khi chạy Softmax (hoặc argmax trên logits).
  * Đối chiếu index này với danh sách nhãn món ăn đọc từ file `src/product_classes.txt` (dòng thứ `index` tương ứng với tên món ăn, tính từ 0).

---

## 4. Hướng dẫn và Code mẫu xây dựng API với FastAPI

Dưới đây là mã nguồn hoàn chỉnh cho file `app.py` để dựng một Web Service bằng FastAPI. Dịch vụ này sẽ nhận ảnh tải lên từ Client, tự động xử lý và trả về kết quả dự đoán (gồm món ăn dự đoán cao nhất và danh sách Top 5 món có khả năng nhất).

### Khởi tạo File `app.py`
Chú lưu code này vào một file tên là `app.py` nằm ở thư mục gốc của dự án:

```python
import os
import numpy as np
from PIL import Image
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import TensorFlow
import tensorflow as tf
from tensorflow.keras import models

app = FastAPI(
    title="AI4Food Nutrition Recognition API",
    description="API nhận diện món ăn chi tiết (840 món) sử dụng mạng EfficientNetV2",
    version="1.0"
)

# Cấu hình CORS để frontend hoặc app di động gọi được API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gọi API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn đến Model và File nhãn lớp
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnetv2_product_model.hdf5")
CLASSES_PATH = os.path.join(BASE_DIR, "src", "product_classes.txt")

# Biến toàn cục để lưu Model và danh sách nhãn
model = None
lst_classes = []

# Tải model và nhãn lúc khởi động Server
@app.on_event("startup")
def startup_event():
    global model, lst_classes
    
    # 1. Tải nhãn lớp
    if not os.path.exists(CLASSES_PATH):
        raise RuntimeError(f"Không tìm thấy file nhãn tại {CLASSES_PATH}")
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        lst_classes = [line.strip() for line in f.readlines()]
        
    # 2. Tải model học máy
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Không tìm thấy file model tại {MODEL_PATH}. Vui lòng tải model trước!")
    
    print("Đang nạp Model vào bộ nhớ...")
    model = models.load_model(MODEL_PATH)
    print("Nạp Model thành công!")

# Định nghĩa cấu trúc dữ liệu trả về cho API
class PredictionResult(BaseModel):
    class_name: str
    probability: float

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    top_5: List[PredictionResult]

# Hàm tính toán Softmax để đổi Logits sang Xác suất phần trăm
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Trừ max để tránh lỗi tràn số (overflow)
    return exp_x / np.sum(exp_x)

@app.post("/predict", response_model=PredictResponse, summary="Dự đoán món ăn từ ảnh gửi lên")
async def predict_food(file: UploadFile = File(...)):
    # 1. Kiểm tra định dạng file gửi lên
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ ảnh định dạng JPG, JPEG, PNG hoặc WEBP.")
    
    try:
        # 2. Đọc ảnh từ bộ nhớ (memory)
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # 3. Tiền xử lý ảnh (Resize về 224x224)
        img_resized = img.resize((224, 224))
        
        # 4. Chuyển thành ma trận numpy và chuẩn hóa [0.0, 1.0]
        x = np.array(img_resized, dtype=np.float32)
        x = x / 255.0
        
        # 5. Thêm chiều batch: (224, 224, 3) -> (1, 224, 224, 3)
        x = np.expand_dims(x, axis=0)
        
        # 6. Chạy mô hình dự đoán (Inference)
        preds = model.predict(x)
        raw_logits = preds[0]
        
        # 7. Tính xác suất bằng Softmax
        probabilities = softmax(raw_logits)
        
        # 8. Tìm món ăn có xác suất cao nhất
        predicted_idx = np.argmax(probabilities)
        final_class = lst_classes[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # 9. Lấy danh sách Top 5 món có khả năng nhất
        top_5_idx = np.argsort(probabilities)[-5:][::-1]
        top_5_results = []
        for idx in top_5_idx:
            top_5_results.append(
                PredictionResult(
                    class_name=lst_classes[idx],
                    probability=round(float(probabilities[idx]), 4)
                )
            )
            
        return PredictResponse(
            prediction=final_class,
            confidence=round(confidence, 4),
            top_5=top_5_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi trong quá trình xử lý ảnh: {str(e)}")

@app.get("/health", summary="Kiểm tra trạng thái hệ thống")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}
```

---

## 5. Hướng dẫn Chạy Service FastAPI

Chú chạy các bước sau để khởi động Web Service:

### Bước 1: Cài đặt thư viện FastAPI và Uvicorn
Chú mở Terminal, kích hoạt môi trường ảo Conda trước đó và chạy lệnh cài đặt:
```bash
conda activate FoodRecognition_env
pip install fastapi uvicorn python-multipart pillow
```

### Bước 2: Khởi động Server API
Đứng tại thư mục gốc của dự án, chú chạy lệnh khởi động server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
* Giải thích tham số:
  * `app:app`: Tìm file `app.py` và nạp ứng dụng `app = FastAPI()`.
  * `--reload`: Tự động tải lại server khi chú thay đổi code (rất tiện khi viết code).
  * `--port 8000`: Mở cổng mạng `8000`.

### Bước 3: Kiểm tra và Giao diện tương tác Swagger UI
FastAPI có một tính năng cực kỳ tuyệt vời là tự động sinh giao diện thử nghiệm API trực quan. Chú mở trình duyệt web bất kỳ và truy cập đường dẫn:
```text
http://127.0.0.1:8000/docs
```
Chú sẽ thấy giao diện **Swagger UI**. Tại đây chú có thể:
1. Chọn API `/predict`.
2. Bấm nút **"Try it out"**.
3. Bấm **"Choose File"** để tải một ảnh món ăn lên.
4. Bấm **"Execute"** để xem kết quả dự đoán trả về dạng JSON cực kỳ chuyên nghiệp!
