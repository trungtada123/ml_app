document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const startBtn = document.getElementById('startBtn');
    const flipCamBtn = document.getElementById('flipCamBtn');
    const captureBtn = document.getElementById('captureBtn');
    const realTimeBtn = document.getElementById('realTimeBtn');
    const realtimeStatus = document.getElementById('realtimeStatus');
    const plateResult = document.getElementById('plateResult');
    const plateImage = document.getElementById('plateImage');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const processBtn = document.getElementById('processBtn');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Debug elements
    const debugContainer = document.querySelector('.debug-container');
    const debugData = document.getElementById('debugData');
    const toggleDebugBtn = document.getElementById('toggleDebug');
    
    // Xử lý toggle debug
    toggleDebugBtn.addEventListener('click', () => {
        if (debugContainer.style.display === 'none') {
            debugContainer.style.display = 'block';
            toggleDebugBtn.textContent = 'Hide Debug Data';
        } else {
            debugContainer.style.display = 'none';
            toggleDebugBtn.textContent = 'Show Debug Data';
        }
    });
    
    let stream = null;
    let isRealTimeDetection = false;
    let realtimeDetectionInterval = null;
    let currentFacingMode = 'environment'; // 'environment' là camera sau, 'user' là camera trước
    let isFlipped = false; // Biến để theo dõi trạng thái đảo ngược của camera
    let isProcessing = false; // Biến theo dõi trạng thái đang xử lý ảnh
    let consecutiveErrors = 0; // Đếm số lỗi liên tiếp
    let realTimeDelayMs = 3000; // Thời gian mặc định giữa các lần nhận diện (ms)
    
    // Thiết lập cấu hình camera
    const cameraOptions = {
        defaultWidth: 1280,
        defaultHeight: 720,
        lowResWidth: 640,
        lowResHeight: 480
    };
    
    // Thêm HTML cho menu điều chỉnh cấu hình
    const createConfigMenu = () => {
        const configMenu = document.createElement('div');
        configMenu.className = 'config-menu';
        configMenu.innerHTML = `
            <div class="config-item">
                <label for="detectionDelay">Thời gian giữa các lần nhận diện (ms):</label>
                <select id="detectionDelay">
                    <option value="1000">1 giây</option>
                    <option value="2000">2 giây</option>
                    <option value="3000" selected>3 giây</option>
                    <option value="5000">5 giây</option>
                </select>
            </div>
            <div class="config-item">
                <label for="cameraResolution">Độ phân giải camera:</label>
                <select id="cameraResolution">
                    <option value="high">Cao (1280x720)</option>
                    <option value="low">Thấp (640x480)</option>
                </select>
            </div>
        `;
        
        // Thêm menu vào trước các nút điều khiển
        document.querySelector('.camera-container .button-container').before(configMenu);
        
        // Xử lý sự kiện thay đổi thời gian giữa các lần nhận diện
        document.getElementById('detectionDelay').addEventListener('change', (e) => {
            realTimeDelayMs = parseInt(e.target.value);
            if (isRealTimeDetection) {
                // Nếu đang trong chế độ nhận diện liên tục, cập nhật lại
                stopRealTimeDetection();
                startRealTimeDetection();
            }
        });
        
        // Xử lý sự kiện thay đổi độ phân giải
        document.getElementById('cameraResolution').addEventListener('change', (e) => {
            if (stream) {
                // Nếu đang có camera, khởi động lại camera với độ phân giải mới
                const tracks = stream.getTracks();
                tracks.forEach(track => track.stop());
                stream = null;
                
                // Khởi động lại camera
                startCamera();
            }
        });
    };
    
    // Gọi hàm tạo menu cấu hình
    createConfigMenu();
    
    // Xử lý chuyển tab
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Dừng nhận diện thời gian thực nếu đang bật
            stopRealTimeDetection();
            
            // Dừng camera nếu đang bật
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                startBtn.disabled = false;
                flipCamBtn.disabled = true;
                captureBtn.disabled = true;
                realTimeBtn.disabled = true;
            }
            
            // Bỏ active từ tất cả các tab
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.style.display = 'none');
            
            // Kích hoạt tab được chọn
            button.classList.add('active');
            const tabId = button.getAttribute('data-tab');
            document.getElementById(`${tabId}-tab`).style.display = 'block';
            
            // Đặt lại kết quả
            plateResult.textContent = 'Chưa có kết quả';
            plateImage.style.display = 'none';
        });
    });

    // Bắt đầu truy cập camera
    startBtn.addEventListener('click', async () => {
        try {
            await startCamera();
        } catch (error) {
            console.error('Lỗi truy cập camera:', error);
            plateResult.textContent = `Không thể truy cập camera: ${error.message}`;
        }
    });
    
    // Lật camera (đảo chiều hình ảnh)
    flipCamBtn.addEventListener('click', () => {
        isFlipped = !isFlipped; // Đảo trạng thái lật
        applyVideoTransform();
        plateResult.textContent = `Đã ${isFlipped ? 'lật' : 'khôi phục'} chiều camera`;
    });
    
    // Áp dụng biến đổi cho video dựa trên trạng thái lật
    function applyVideoTransform() {
        if (isFlipped) {
            video.style.transform = 'scaleX(-1)'; // Lật ngang video
        } else {
            video.style.transform = 'scaleX(1)'; // Khôi phục trạng thái bình thường
        }
    }
    
    // Hàm lấy cấu hình camera
    function getCameraConstraints() {
        const resolution = document.getElementById('cameraResolution').value;
        const constraints = {
            video: {
                width: { ideal: resolution === 'high' ? cameraOptions.defaultWidth : cameraOptions.lowResWidth },
                height: { ideal: resolution === 'high' ? cameraOptions.defaultHeight : cameraOptions.lowResHeight },
                frameRate: { ideal: 30 },
                facingMode: currentFacingMode
            },
            audio: false
        };
        return constraints;
    }
    
    // Hàm khởi động camera với chế độ hiện tại
    async function startCamera() {
        try {
            // Yêu cầu quyền truy cập camera với cấu hình đã chọn
            stream = await navigator.mediaDevices.getUserMedia(getCameraConstraints());
            
            // Hiển thị video từ camera
            video.srcObject = stream;
            
            // Áp dụng trạng thái lật camera nếu có
            applyVideoTransform();
            
            // Kích hoạt các nút
            flipCamBtn.disabled = false;
            captureBtn.disabled = false;
            realTimeBtn.disabled = false;
            startBtn.disabled = true;
            
            // Hiển thị thông tin về track video
            const videoTrack = stream.getVideoTracks()[0];
            const settings = videoTrack.getSettings();
            console.log('Cài đặt camera:', settings);
            
            plateResult.textContent = 'Camera đã sẵn sàng. Hãy chụp ảnh biển số.';
        } catch (error) {
            console.error('Lỗi truy cập camera:', error);
            throw error;
        }
    }

    // Chụp ảnh từ camera
    captureBtn.addEventListener('click', () => {
        if (!stream) return;
        
        // Dừng nhận diện thời gian thực nếu đang bật
        stopRealTimeDetection();
        
        captureAndProcess();
    });
    
    // Chức năng nhận diện thời gian thực
    realTimeBtn.addEventListener('click', () => {
        if (!stream) return;
        
        if (!isRealTimeDetection) {
            // Bắt đầu nhận diện thời gian thực
            startRealTimeDetection();
            realTimeBtn.textContent = 'Dừng nhận diện';
            realtimeStatus.style.display = 'flex';
            
            // Hiển thị thông tin về tần suất nhận diện
            const delayText = (realTimeDelayMs / 1000).toFixed(1);
            document.querySelector('#realtimeStatus span').textContent = 
                `Đang nhận diện trực tiếp (${delayText}s/lần)`;
        } else {
            // Dừng nhận diện thời gian thực
            stopRealTimeDetection();
            realTimeBtn.textContent = 'Nhận diện liên tục';
            realtimeStatus.style.display = 'none';
        }
    });
    
    // Bắt đầu nhận diện thời gian thực
    function startRealTimeDetection() {
        isRealTimeDetection = true;
        consecutiveErrors = 0; // Đặt lại bộ đếm lỗi
        
        // Thực hiện nhận diện ngay lập tức
        captureAndProcess();
        
        // Sau đó, thiết lập interval để nhận diện tiếp
        realtimeDetectionInterval = setInterval(() => {
            // Chỉ thực hiện nếu không có yêu cầu nào đang xử lý
            if (!isProcessing) {
                captureAndProcess();
            } else {
                console.log('Bỏ qua chu kỳ này vì yêu cầu trước đó vẫn đang xử lý');
            }
        }, realTimeDelayMs);
    }
    
    // Dừng nhận diện thời gian thực
    function stopRealTimeDetection() {
        if (isRealTimeDetection) {
            clearInterval(realtimeDetectionInterval);
            isRealTimeDetection = false;
            realTimeBtn.textContent = 'Nhận diện liên tục';
            realtimeStatus.style.display = 'none';
        }
    }
    
    // Hàm chụp và xử lý ảnh
    function captureAndProcess() {
        if (!video.videoWidth) {
            console.error('Video không có kích thước, camera có thể chưa được khởi tạo đúng');
            return;
        }
        
        // Thiết lập kích thước canvas
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Vẽ khung hình hiện tại vào canvas với xử lý lật ảnh nếu cần
        if (isFlipped) {
            context.translate(canvas.width, 0);
            context.scale(-1, 1);
        }
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Khôi phục trạng thái của context nếu đã lật
        if (isFlipped) {
            context.setTransform(1, 0, 0, 1, 0, 0); // Đặt lại ma trận biến đổi
        }
        
        // Chuyển đổi canvas thành dữ liệu ảnh
        const imageData = canvas.toDataURL('image/jpeg');
        
        // Gửi ảnh đến server để xử lý
        sendImageToServer(imageData);
    }
    
    // Xử lý chọn file
    imageInput.addEventListener('change', (event) => {
        if (event.target.files.length > 0) {
            const file = event.target.files[0];
            const reader = new FileReader();
            
            reader.onload = (e) => {
                // Hiển thị ảnh đã chọn
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                
                // Kích hoạt nút phân tích
                processBtn.disabled = false;
            };
            
            reader.readAsDataURL(file);
        }
    });
    
    // Xử lý phân tích ảnh đã tải lên
    processBtn.addEventListener('click', () => {
        if (imagePreview.src) {
            plateResult.textContent = 'Đang phân tích biển số...';
            sendImageToServer(imagePreview.src);
        }
    });

    // Gửi ảnh đến server
    function sendImageToServer(imageData) {
        // Đánh dấu đang xử lý
        isProcessing = true;
        plateResult.textContent = 'Đang phân tích biển số...';
        
        // Hiển thị trạng thái xử lý
        updateRealtimeStatusIndicator(true);
        
        fetch('/api/recognize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Đánh dấu đã xử lý xong
            isProcessing = false;
            updateRealtimeStatusIndicator(false);
            
            // Đặt lại số lỗi liên tiếp
            consecutiveErrors = 0;
            
            // Debug - Log dữ liệu trả về từ server
            console.log("Dữ liệu trả về từ server:", data);
            if (data.plates) {
                console.log(`Số lượng biển số phát hiện được: ${data.plates.length}`);
                console.log("Chi tiết các biển số:", data.plates);
            }
            
            // Hiển thị dữ liệu debug
            debugData.textContent = JSON.stringify(data, null, 2);
            
            if (data.success) {
                // Tạo danh sách biển số đã phát hiện được
                let resultHTML = '';
                
                // Hiển thị tất cả các biển số đã phát hiện
                if (data.plates && data.plates.length > 0) {
                    if (data.plates.length === 1) {
                        resultHTML = `<div>Đã nhận diện biển số: ${data.plates[0].text}</div>`;
                    } else {
                        // Hiển thị nhiều biển số
                        resultHTML = `<div>Đã nhận diện ${data.plates.length} biển số:</div>`;
                        data.plates.forEach((plate, index) => {
                            resultHTML += `<div style="margin-top: 8px; padding: 8px; background-color: #f0f9ff; border-left: 3px solid #3498db; border-radius: 4px;">
                                <span style="font-weight: bold;">Biển số ${index + 1}:</span> ${plate.text}
                            </div>`;
                        });
                    }
                    plateResult.innerHTML = resultHTML;
                } else if (data.plate) {
                    // Để tương thích với phiên bản cũ
                    plateResult.textContent = `Biển số xe: ${data.plate}`;
                } else {
                    plateResult.textContent = 'Không tìm thấy thông tin biển số';
                }
                
                // Hiển thị ảnh đã xử lý từ server
                if (data.processed_image) {
                    plateImage.src = 'data:image/jpeg;base64,' + data.processed_image;
                    plateImage.style.display = 'block';
                }
            } else {
                handleError(data.error || 'Không thể nhận diện biển số. Vui lòng thử lại.');
            }
        })
        .catch(error => {
            // Đánh dấu đã xử lý xong
            isProcessing = false;
            updateRealtimeStatusIndicator(false);
            
            console.error('Lỗi khi gửi ảnh:', error);
            handleError('Lỗi kết nối đến server. Vui lòng thử lại.');
        });
    }
    
    // Cập nhật trạng thái xử lý trong chỉ báo nhận diện liên tục
    function updateRealtimeStatusIndicator(isActive) {
        if (isRealTimeDetection) {
            const statusDot = document.querySelector('#realtimeStatus .status-dot');
            if (statusDot) {
                if (isActive) {
                    statusDot.style.backgroundColor = '#e74c3c'; // Đỏ khi đang xử lý
                } else {
                    statusDot.style.backgroundColor = '#2ecc71'; // Xanh khi đã xong
                }
            }
        }
    }
    
    // Xử lý lỗi trong chế độ nhận diện liên tục
    function handleError(errorMessage) {
        plateResult.textContent = `Lỗi: ${errorMessage}`;
        
        if (isRealTimeDetection) {
            consecutiveErrors++;
            console.log(`Lỗi lần thứ ${consecutiveErrors}`);
            
            // Nếu có quá nhiều lỗi liên tiếp, tự động dừng chế độ nhận diện liên tục
            if (consecutiveErrors >= 5) {
                plateResult.textContent = `Đã dừng nhận diện liên tục do có quá nhiều lỗi: ${errorMessage}`;
                stopRealTimeDetection();
            }
        }
    }
}); 