from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
import joblib
import base64
import re

app = Flask(__name__, static_folder='.')

# Đường dẫn đến các model đã huấn luyện
KNN_MODEL_PATH = 'knn_chars_model.xml'
SVM_MODEL_PATH = 'plate_detector.pkl'

# Tham số cấu hình từ notebook
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
MIN_CONTOUR_AREA = 40
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
Min_char = 0.01
Max_char = 0.09

# Load các model
def load_models():
    try:
        print("Bắt đầu load các model...")
        
        # Kiểm tra sự tồn tại của các file model
        if not os.path.exists(KNN_MODEL_PATH):
            print(f"Lỗi: Không tìm thấy file KNN model tại {KNN_MODEL_PATH}")
            return None, None
        
        if not os.path.exists(SVM_MODEL_PATH):
            print(f"Lỗi: Không tìm thấy file SVM model tại {SVM_MODEL_PATH}")
            return None, None
            
        print(f"Tìm thấy các file model: {KNN_MODEL_PATH}, {SVM_MODEL_PATH}")
        
        # Load KNN model cho nhận dạng ký tự
        print(f"Đang load KNN model từ {KNN_MODEL_PATH}...")
        knn = cv2.ml.KNearest_load(KNN_MODEL_PATH)
        print("Load KNN model thành công")
        
        # Load SVM model cho phân loại biển số
        print(f"Đang load SVM model từ {SVM_MODEL_PATH}...")
        clf = joblib.load(SVM_MODEL_PATH)
        print("Load SVM model thành công")
        
        return knn, clf
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Các hàm xử lý ảnh từ notebook
def extractValue(imgOriginal):
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    _, _, imgValue = cv2.split(imgHSV)
    return imgValue

def maximizeContrast(imgGrayscale):
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    return cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(
        imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return imgGrayscale, imgThresh

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return [
        img.shape[1] / img.shape[0],
        np.mean(gray),
        np.std(gray),
        np.sum(edges) / (img.shape[0] * img.shape[1])
    ]

# Nhận dạng các ký tự trên biển số
def recognize_characters(roi, roi_thresh, knn_model):
    # Xử lý ảnh cho phân đoạn ký tự
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(roi_thresh, cv2.MORPH_DILATE, kerel3)
    
    # Tìm contour các ký tự
    cont, _ = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    char_x_ind = {}
    char_x = []
    height, width, _ = roi.shape
    roiarea = height * width
    
    # Tạo bản sao của ROI để vẽ các khung ký tự lên đó
    roi_with_chars = roi.copy()
    
    # Lọc và lưu thông tin các contour ký tự
    for ind, cnt in enumerate(cont):
        (x_char, y_char, w_char, h_char) = cv2.boundingRect(cnt)
        ratiochar = w_char / h_char
        char_area = w_char * h_char
        if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
            if x_char in char_x:
                x_char += 1
            char_x.append(x_char)
            char_x_ind[x_char] = ind
            
            # Vẽ khung xung quanh ký tự được nhận diện
            cv2.rectangle(roi_with_chars, (x_char, y_char), 
                         (x_char + w_char, y_char + h_char), (0, 255, 0), 2)
    
    # Sắp xếp các ký tự từ trái sang phải
    char_x = sorted(char_x)
    first_line = ""
    second_line = ""
    detected_chars = []
    
    # Nhận dạng từng ký tự
    for i in char_x:
        (x_char, y_char, w_char, h_char) = cv2.boundingRect(cont[char_x_ind[i]])
        imgROI = thre_mor[y_char:y_char + h_char, x_char:x_char + w_char]
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        roi_flatten = imgROIResized.reshape(1, -1).astype(np.float32)
        _, result, _, _ = knn_model.findNearest(roi_flatten, k=3)
        strCurrentChar = chr(int(result[0][0]))
        
        # Lưu thông tin ký tự đã nhận diện
        char_info = {
            'char': strCurrentChar,
            'x': x_char,
            'y': y_char,
            'w': w_char,
            'h': h_char,
            'line': 1 if y_char < height / 3 else 2
        }
        detected_chars.append(char_info)
        
        # Vẽ ký tự được nhận diện lên ảnh
        cv2.putText(roi_with_chars, strCurrentChar, 
                   (x_char, y_char - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 255), 2)
        
        if y_char < height / 3:
            first_line += strCurrentChar
        else:
            second_line += strCurrentChar
    
    # Tổng hợp biển số
    plate_text = f"{first_line}-{second_line}" if second_line else first_line
    
    return plate_text, roi_with_chars, detected_chars

# Hàm nhận diện biển số với cơ chế thử lại và điều chỉnh ngưỡng
def recognize_license_plate(img, knn_model, clf_model):
    # Danh sách các ngưỡng Canny để thử nếu không tìm thấy biển số
    canny_thresholds = [(250, 255), (200, 255), (150, 255), (100, 200)]
    
    # Các tỷ lệ khung hình để thử với các ảnh khác nhau
    aspect_ratio_ranges = [(1.0, 6.0), (0.8, 7.0)]
    
    # Các ngưỡng phóng đại kích thước cho OCR
    scaling_factors = [3, 2, 4]
    
    # Resize để có kích thước phù hợp
    img = cv2.resize(img, (1920, 1080))
    img_result = img.copy()
    
    imgGray, imgThreshplate = preprocess(img)
    
    # Mảng lưu thông tin các biển số được tìm thấy
    detected_plates = []
    
    # Thử các tham số khác nhau cho đến khi tìm thấy ít nhất một biển số
    for canny_threshold in canny_thresholds:
        for aspect_ratio_range in aspect_ratio_ranges:
            print(f"Thử với ngưỡng Canny {canny_threshold} và tỷ lệ {aspect_ratio_range}")
            
            # Áp dụng Canny và dãn ảnh
            canny_image = cv2.Canny(imgThreshplate, canny_threshold[0], canny_threshold[1])
            dilated_image = cv2.dilate(canny_image, np.ones((3, 3), np.uint8), iterations=1)

            # Tìm contour
            contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]  # Tăng số lượng contour xem xét
            
            # Xác định các vùng tiềm năng chứa biển số
            potential_plates = []
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.06 * peri, True)
                
                # Kiểm tra hình dạng gần giống hình chữ nhật (4 đỉnh)
                if len(approx) < 4 or len(approx) > 5:  # Nới lỏng điều kiện, cho phép từ 4-5 đỉnh
                    continue
                    
                x, y, w, h = cv2.boundingRect(approx)
                
                # Bỏ qua những vùng quá nhỏ
                if w < 60 or h < 20:
                    continue
                    
                # Kiểm tra tỷ lệ khung hình của biển số
                aspect_ratio = float(w) / h
                if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                    continue
                
                # Lưu thông tin vùng tiềm năng
                potential_plates.append({
                    'contour': approx,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h
                })
            
            # Nếu không tìm thấy vùng tiềm năng, thử tiếp tham số khác
            if not potential_plates:
                continue
                
            # Phân tích từng vùng tiềm năng
            for plate_info in potential_plates:
                approx = plate_info['contour']
                x, y, w, h = plate_info['x'], plate_info['y'], plate_info['w'], plate_info['h']
                
                # Cắt vùng ảnh chứa biển số tiềm năng
                roi = img[y:y+h, x:x+w]
                roi_thresh = imgThreshplate[y:y+h, x:x+w]

                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                # Phân loại vùng ảnh với SVM
                try:
                    roi_resized = cv2.resize(roi, (160, 80))
                    feat = extract_features(roi_resized)
                    is_plate = clf_model.predict([feat])[0]
                    
                    # Nếu không phải biển số, bỏ qua
                    if is_plate != 1:
                        continue
                except Exception as e:
                    print(f"Lỗi khi dự đoán biển số: {e}")
                    continue

                # Vẽ khung biển số với độ dày và màu sắc rõ ràng hơn
                cv2.drawContours(img_result, [approx], -1, (0, 255, 0), 3)
                cv2.rectangle(img_result, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Thử các hệ số phóng đại khác nhau cho OCR
                for scale_factor in scaling_factors:
                    # Resize lại ảnh cho OCR
                    roi_ocr = cv2.resize(roi, (roi.shape[1] * scale_factor, roi.shape[0] * scale_factor))
                    roi_thresh_ocr = cv2.resize(roi_thresh, (roi_ocr.shape[1], roi_ocr.shape[0]))
                    
                    try:
                        # Nhận dạng các ký tự trên biển số
                        plate_text, roi_with_chars, detected_chars = recognize_characters(roi_ocr, roi_thresh_ocr, knn_model)
                        
                        # Kiểm tra kết quả nhận dạng
                        if len(plate_text) < 3:  # Nếu nhận dạng quá ít ký tự, thử scale_factor tiếp theo
                            print(f"Nhận dạng được quá ít ký tự ({plate_text}) với scale_factor={scale_factor}, thử lại")
                            continue
                            
                        # Thêm thông tin biển số vào danh sách
                        plate_info = {
                            'text': plate_text,
                            'coords': {'x': x, 'y': y, 'width': w, 'height': h},
                            'roi_with_chars': roi_with_chars,
                            'approx': approx
                        }
                        
                        # Kiểm tra xem biển số này đã được thêm vào chưa (tránh trùng lặp)
                        is_duplicate = False
                        for existing_plate in detected_plates:
                            # So sánh tọa độ để xác định nếu là cùng một biển số
                            ex = existing_plate['coords']['x']
                            ey = existing_plate['coords']['y']
                            ew = existing_plate['coords']['width']
                            eh = existing_plate['coords']['height']
                            
                            # Nếu hai biển số chồng lấn nhiều hơn 50%, coi là trùng nhau
                            overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                            overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                            overlap_area = overlap_x * overlap_y
                            area = w * h
                            if overlap_area / area > 0.5:
                                is_duplicate = True
                                # Nếu biển số mới có nhiều ký tự hơn, thay thế biển số cũ
                                if len(plate_text) > len(existing_plate['text']):
                                    detected_plates.remove(existing_plate)
                                    detected_plates.append(plate_info)
                                break
                                
                        if not is_duplicate:
                            detected_plates.append(plate_info)
                            
                            # Hiển thị văn bản biển số với kiểu dáng nổi bật
                            label_bg_y = y - 40 if y > 40 else y + h + 10
                            # Vẽ nền cho văn bản
                            cv2.rectangle(img_result, (x, label_bg_y - 30), (x + w, label_bg_y + 10), (0, 0, 0), -1)
                            # Vẽ văn bản
                            cv2.putText(img_result, f"Biển số: {plate_text}", (x, label_bg_y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                            
                            # Đã nhận dạng thành công với scale_factor này, không cần thử tiếp
                            break
                    except Exception as e:
                        print(f"Lỗi khi nhận dạng ký tự với scale_factor={scale_factor}: {e}")
                        # Tiếp tục thử scale_factor tiếp theo
            
            # Nếu đã tìm thấy ít nhất một biển số, không cần thử thêm tham số khác
            if detected_plates:
                break
        
        # Nếu đã tìm thấy ít nhất một biển số, không cần thử thêm tham số khác
        if detected_plates:
            break
    
    # Hiển thị các vùng ảnh chi tiết ở phía dưới
    if detected_plates:
        # Tìm vị trí thấp nhất của tất cả biển số
        max_y = 0
        for plate in detected_plates:
            coords = plate['coords']
            max_y = max(max_y, coords['y'] + coords['height'])
        
        # Thêm khoảng cách để đặt các ROI bên dưới
        vertical_offset = max_y + 50
        
        # Đảm bảo không vượt quá kích thước ảnh
        if vertical_offset >= img_result.shape[0] - 150:
            vertical_offset = max_y + 20
        
        # Hiển thị ROI của từng biển số theo hàng ngang
        horizontal_offset = 20
        max_height_in_row = 0
        
        for i, plate in enumerate(detected_plates):
            roi_with_chars = plate['roi_with_chars']
            roi_height, roi_width, _ = roi_with_chars.shape
            
            # Giới hạn kích thước ROI nếu quá lớn
            max_display_width = 400  # Kích thước tối đa cho mỗi ROI
            if roi_width > max_display_width:
                scale_factor = max_display_width / roi_width
                new_width = int(roi_width * scale_factor)
                new_height = int(roi_height * scale_factor)
                roi_with_chars = cv2.resize(roi_with_chars, (new_width, new_height))
                roi_height, roi_width, _ = roi_with_chars.shape
            
            # Nếu ROI không vừa hàng hiện tại, xuống hàng mới
            if horizontal_offset + roi_width > img_result.shape[1] - 20:
                horizontal_offset = 20
                vertical_offset += max_height_in_row + 40  # Thêm khoảng cách giữa các hàng
                max_height_in_row = 0
            
            # Lưu lại chiều cao lớn nhất trong hàng
            max_height_in_row = max(max_height_in_row, roi_height)
            
            # Kiểm tra nếu vượt quá chiều cao của ảnh
            if vertical_offset + roi_height >= img_result.shape[0]:
                # Thay vì bỏ qua, mở rộng ảnh kết quả
                padding = vertical_offset + roi_height + 50 - img_result.shape[0]
                if padding > 0:
                    padding_img = np.ones((padding, img_result.shape[1], 3), dtype=np.uint8) * 255
                    img_result = np.vstack((img_result, padding_img))
            
            # Vẽ ROI lên ảnh kết quả
            try:
                img_result[vertical_offset:vertical_offset + roi_height, 
                          horizontal_offset:horizontal_offset + roi_width] = roi_with_chars
                
                # Thêm chú thích cho ROI
                label_y = vertical_offset - 10
                cv2.putText(img_result, f"Biển số {i+1}: {plate['text']}", 
                           (horizontal_offset, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Cập nhật vị trí ngang cho ROI tiếp theo
                horizontal_offset += roi_width + 30
            except Exception as e:
                print(f"Lỗi khi vẽ ROI: {e}")
    else:
        print("Không tìm thấy biển số nào sau khi thử tất cả các tham số")
    
    # Encode ảnh kết quả về base64 để gửi lại client
    _, buffer = cv2.imencode('.jpg', img_result)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return detected_plates, processed_image_base64

# Xử lý ảnh từ base64
def process_base64_image(base64_string):
    try:
        # Kiểm tra chuỗi base64
        if not base64_string:
            print("Chuỗi base64 rỗng hoặc None")
            return None
        
        # In ra một phần của chuỗi base64 để debug
        print(f"Chuỗi base64 bắt đầu với: {base64_string[:50]}...")
        
        # Xóa phần header nếu có
        base64_pattern = r'^data:image/[a-zA-Z]+;base64,'
        if re.search(base64_pattern, base64_string):
            print("Tìm thấy header image/base64, đang xóa...")
            base64_string = re.sub(base64_pattern, '', base64_string)
        
        # Loại bỏ khoảng trắng hoặc ký tự xuống dòng nếu có
        base64_string = base64_string.strip()
        
        # Kiểm tra độ dài của chuỗi base64 sau khi xử lý
        if len(base64_string) < 100:  # Giá trị tối thiểu hợp lý
            print(f"Chuỗi base64 quá ngắn sau khi xử lý: {len(base64_string)} ký tự")
            return None
            
        # Giải mã base64 thành bytes
        try:
            image_bytes = base64.b64decode(base64_string)
            print(f"Đã giải mã base64 thành {len(image_bytes)} bytes")
        except Exception as e:
            print(f"Lỗi khi giải mã base64: {e}")
            return None
        
        # Chuyển đổi bytes thành mảng numpy
        nparr = np.frombuffer(image_bytes, np.uint8)
        print(f"Đã chuyển đổi thành mảng numpy với {len(nparr)} phần tử")
        
        # Đọc ảnh từ buffer
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            print("OpenCV không thể đọc được ảnh từ dữ liệu")
            return None
            
        print(f"Đã đọc thành công ảnh với kích thước: {img.shape}")
        return img
    except Exception as e:
        print(f"Lỗi trong process_base64_image: {e}")
        import traceback
        traceback.print_exc()
        return None

# Route cho trang chủ
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# API nhận dạng biển số
@app.route('/api/recognize', methods=['POST'])
def recognize_plate_api():
    try:
        print("=============================================")
        print("Nhận được yêu cầu nhận diện biển số")
        data = request.json
        
        if 'image' not in data:
            print("Lỗi: Không tìm thấy ảnh trong dữ liệu gửi lên")
            return jsonify({'success': False, 'error': 'Không tìm thấy ảnh'}), 400
        
        print(f"Kích thước dữ liệu ảnh: {len(str(data['image']))} ký tự")
        
        # Load models
        print("Đang load models...")
        knn_model, clf_model = load_models()
        if knn_model is None or clf_model is None:
            print("Lỗi: Không thể load models")
            return jsonify({'success': False, 'error': 'Không thể load models'}), 500
        
        # Xử lý ảnh
        print("Đang xử lý ảnh...")
        try:
            img = process_base64_image(data['image'])
            if img is None or img.size == 0:
                print("Lỗi: Không thể xử lý ảnh từ dữ liệu base64")
                return jsonify({'success': False, 'error': 'Không thể xử lý ảnh'}), 400
            print(f"Kích thước ảnh đã xử lý: {img.shape}")
        except Exception as img_error:
            print(f"Lỗi khi xử lý ảnh: {img_error}")
            return jsonify({'success': False, 'error': f'Lỗi xử lý ảnh: {str(img_error)}'}), 400
        
        # Nhận diện biển số
        print("Đang nhận diện biển số...")
        try:
            detected_plates, processed_image_base64 = recognize_license_plate(img, knn_model, clf_model)
            print(f"Kết quả nhận diện: {len(detected_plates)} biển số")
        except Exception as recog_error:
            print(f"Lỗi khi nhận diện biển số: {recog_error}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': f'Lỗi nhận diện: {str(recog_error)}'}), 500
        
        if not detected_plates:
            print("Không tìm thấy biển số trong ảnh")
            return jsonify({'success': False, 'error': 'Không tìm thấy biển số'}), 404
        
        # Chuẩn bị dữ liệu trả về
        plates_data = []
        for i, plate in enumerate(detected_plates):
            print(f"Biển số {i+1}: {plate['text']}")
            plates_data.append({
                'text': plate['text'],
                'coords': plate['coords']
            })
        
        print(f"Trả về kết quả thành công với {len(plates_data)} biển số")
        
        # Đảm bảo tất cả biển số được trả về trong 'plates'
        response_data = {
            'success': True, 
            'plates': plates_data,
            'processed_image': processed_image_base64,
            'num_plates': len(detected_plates)
        }
        
        # Thêm trường plate và coordinates cho tương thích ngược (lấy biển số đầu tiên)
        if len(detected_plates) > 0:
            response_data['plate'] = detected_plates[0]['text']
            response_data['coordinates'] = detected_plates[0]['coords']
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Lỗi không xử lý được: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Phục vụ các file tĩnh
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 