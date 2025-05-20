# Ứng dụng nhận diện biển số xe

Ứng dụng web này sử dụng các model đã huấn luyện để nhận diện biển số xe từ camera hoặc ảnh tải lên.

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- Các thư viện Python trong file requirements.txt

### Các bước cài đặt

1. Clone repository hoặc tải mã nguồn về máy
2. Di chuyển vào thư mục web:
   ```
   cd web
   ```
3. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```

## Chạy ứng dụng

1. Từ thư mục web, chạy lệnh sau để khởi động server:
   ```
   python server.py
   ```
2. Mở trình duyệt web và truy cập địa chỉ:
   ```
   http://localhost:5000
   ```

## Cách sử dụng

### Sử dụng camera
1. Chọn tab "Camera"
2. Nhấn nút "Bật camera" để kích hoạt camera
3. Bạn có thể:
   - Nhấn nút "Lật camera" để chuyển đổi giữa camera trước và sau
   - Nhấn nút "Chụp ảnh" để chụp một khung hình và phân tích
   - Nhấn nút "Nhận diện liên tục" để tự động nhận diện biển số theo thời gian thực (cứ mỗi 2 giây)
4. Hệ thống sẽ xử lý ảnh và hiển thị kết quả nhận diện

### Sử dụng ảnh tải lên
1. Chọn tab "Tải ảnh lên"
2. Nhấn nút "Chọn hình ảnh" và chọn ảnh biển số từ máy tính
3. Nhấn nút "Phân tích ảnh" để xử lý
4. Hệ thống sẽ hiển thị kết quả nhận diện

## Cấu trúc thư mục

- `index.html`: Giao diện người dùng
- `styles.css`: File CSS định dạng giao diện
- `script.js`: Mã JavaScript xử lý camera và gửi dữ liệu
- `server.py`: Server Flask xử lý các request API và nhận diện biển số
- `requirements.txt`: Danh sách các thư viện Python cần thiết

## Thuật toán nhận diện

Ứng dụng sử dụng quy trình sau để nhận diện biển số:

1. **Tiền xử lý ảnh**: 
   - Chuyển đổi không gian màu
   - Tăng cường tương phản
   - Làm mờ ảnh và áp dụng ngưỡng thích ứng

2. **Phát hiện biển số**:
   - Phát hiện cạnh và dãn ảnh
   - Tìm các contour
   - Lọc các contour để tìm biển số dựa vào hình dạng và diện tích
   - Sử dụng SVM đã huấn luyện để phân loại các vùng ảnh

3. **Phân đoạn và nhận dạng ký tự**:
   - Phân đoạn ảnh để tách từng ký tự
   - Lọc các contour ký tự dựa vào kích thước
   - Chuẩn hóa ký tự về kích thước cố định
   - Nhận dạng ký tự bằng KNN đã huấn luyện
   - Kết hợp các ký tự để tạo thành biển số hoàn chỉnh

## Model đã huấn luyện

- **KNN model (`knn_chars_model.xml`)**: Dùng để nhận dạng ký tự trên biển số
- **SVM model (`plate_detector.pkl`)**: Dùng để phân loại vùng ảnh có chứa biển số 