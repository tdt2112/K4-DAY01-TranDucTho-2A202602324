# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python: 3.13.15
PyTorch: 2.11.0+cpu
Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): sample_id: trafic,class_id: 468, class_name: cab, rank: 1, score: 0.510915, taxonomy_name: ImageNet-1K.
- Record này mô tả toàn ảnh như thế nào? Record này dự đoán lớp cab (xe taxi) là đối tượng chủ đạo xuất hiện trong toàn bộ khung hình traffic với độ tin cậy mô hình là khoảng 51.1%.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Đội ngũ xây dựng tập dữ liệu ImageNet-1K.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? class_id giúp hệ thống máy tính xử lý và truy vấn dữ liệu hiệu quả; class_name giúp con người đọc hiểu nhanh chóng; còn taxonomy_name xác định rõ phạm vi bộ nhãn chuẩn đang sử dụng (ImageNet-1K).
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần nêu rõ quy tắc ưu tiên là nên gán nhãn cho chủ thể chiếm diện tích lớn nhất, chủ thể trung tâm, hoặc cho phép gán nhiều nhãn khi có nhiều chủ thể cùng xuất hiện.
- Vì sao model score không phải ground truth? Model score chỉ phản ánh kết quả tính toán của checkpoint. Ground truth phải do con người xác nhận dựa trên taxonomy và guideline của dự án.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): class_name: person, score: 0.912625, bbox_xyxy: [385.33, 69.24, 498.92, 348.92], bbox_width: 113.58, bbox_height: 279.68.
- Diễn giải vị trí box bằng lời: Hộp giới hạn bao quanh đối tượng người (person) trong ảnh kitchen bắt đầu từ góc trên bên trái có tọa độ $(385.33, 69.24)$ đến góc dưới bên phải $(498.92, 348.92)$, với chiều rộng hộp là $113.58$ pixel và chiều cao là $279.68$ pixel tính từ gốc tọa độ góc trên bên trái ảnh.
- So sánh số prediction ở hai threshold: Khi hạ ngưỡng (threshold thấp hơn như 0.20), số lượng vật thể phát hiện tăng lên do giữ lại cả các dự đoán xác suất thấp; khi tăng ngưỡng (như 0.60), số lượng giảm, chỉ giữ lại các vật thể có độ tin cậy cao.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Ngưỡng thấp giúp tăng độ bao phủ (ít bỏ sót vật thể thực tế) nhưng gây nhiễu, làm tăng nặng khối lượng kiểm duyệt cho con người; ngưỡng cao giảm bớt nhiễu nhưng dễ làm sót đối tượng.
- Đề xuất một quy tắc box chặt: Hộp giới hạn phải ôm sát mép ngoài cùng của vật thể, không để dư ra khoảng trống nền lớn và không để thiếu các bộ phận của vật thể.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định rõ tỷ lệ hiển thị tối thiểu để quyết định có vẽ box hay không; các trường hợp phức tạp cần escalation xử lý.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): instance_id: kitchen-001, class_name: person, score: 0.899318, số điểm polygon: 348, polygon_xy: [[385.45, 66.44], [386.0, 67.0], ..., [498.02, 348.58]]
- Polygon bổ sung chi tiết gì so với box? Polygon mô tả chính xác đường viền hình học từng pixel theo hình dáng thực tế của đối tượng (ví dụ: đường cong cơ thể người, nếp áo), thay vì chỉ bao quanh bằng một khung hình chữ nhật thô như bounding box.
- `instance_id` dùng để làm gì và không phải loại ID nào? instance_id dùng để phân biệt các cá thể riêng biệt trong cùng một lớp vật thể không phải là mã định danh lớp (class_id) hay mã theo dõi (tracking ID).
- Đề xuất một quy tắc biên mask: Đường biên mask phải bám sát ranh giới pixel thực tế của đối tượng, không lấn sang nền hoặc vật thể xung quanh.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Cần quy định cách xử lý các vùng ranh giới mờ, nhòe hoặc khi hai cá thể dính sát vào nhau; trường hợp khó phân tách cần có quy tắc escalation thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cấp ảnh (Image-level label) | Ảnh chứa nhiều đối tượng nhưng mô hình chỉ gán 1 nhãn nổi bật nhất (cab) | Lựa chọn nhãn chính xác nhất theo đúng hệ thống phân cấp taxonomy trong guideline | Kiểm tra xem nhãn được gán có đúng với chủ thể chính trong bức ảnh hay không |
| Phát hiện vật thể | Bounding box [x_min, y_min, x_max, y_max] kèm class_id | Bỏ sót các vật thể nhỏ/mờ (như lọ hoa potted plant, thìa spoon khi nâng threshold) | Vẽ khung chữ nhật ôm sát từng đối tượng được yêu cầu gán nhãn | Kiểm tra xem box có bị thừa nền, cắt phạm vật thể hoặc gán sai nhãn lớp không |
| Instance segmentation | Đa giác polygon_xy kèm instance_id | Ranh giới giữa các vật thể chồng lấp lên nhau khó xác định (như chồng bát bowl) | Vẽ đường đa giác khép kín bao quanh chính xác ranh giới từng cá thể | Kiểm tra độ mịn đường biên polygon, đảm bảo không gán trùng instance_id |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Không bao giờ tải lên hệ thống hoặc đưa vào mã nguồn các dữ liệu cá nhân, hình ảnh khuôn mặt, biển số xe, thông tin nhạy cảm hoặc dữ liệu nội bộ độc quyền.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Lab Coach / Giảng viên phụ trách học phần.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
