# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?  
**Trả lời:** record này nói về toàn bộ ảnh. Lí do: dựa vào task=image_classification => mô hình chỉ đánh giá mức độ tương đồng của cả bức ảnh với các danh mục(class). Ngoài ra, nếu record mô tả 1 vật cụ thể trong ảnh, cần phải có thông tin cụ thể của tọa độ vị trí như bbox:[x,y,w,h]

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?  
**Trả lời:** đi kèm taxonomy/checkpoint
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?  
**Trả lời:** nếu dùng mỗi class_name sẽ không tránh được vấn đề 1 từ có đa nghĩa nên cần thêm class_id để mang tính unique. Cần taxonomy vì tránh xung đột id giữa các bộ dữ liệu khác nhau, ở đây là ImageNet-1K. Nhưng trong bộ dữ liệu khác, class_id này có thể không phải là class_name=cab
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?  
**Trả lời:** guidline cần chỉ rõ trước khi gán nhãn, cần xác định vật thể chiếm diện tích lớn nhất, độ nét cao nhất, ở giữa nhất
6. Vì sao không được lấy `score` của model làm ground truth?  
**Trả lời:** Vì score là phán đoán chủ quan của mô hình. Mô hình có thể tự tin hoàn toàn vào 1 kết quả sai (overconfident

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?  
**Trả lời:** góc trên-trái: x=385.33, y=69.24

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?  
**Trả lời:** Chọn ngưỡng 0.20: 17 prediction. Chọn ngưỡng 0.60: 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?  
**Trả lời:.** xóa các hộp rác hay hộp nhận diện sai, xử lí các hộp có hiện tượng bị trùng lặp hoặc bị lấp, chỉnh lại kích thước hộp
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).  
**Trả lời:** khung bao phải chạm sát vào 4 điểm cực biên (trên, dưới, trái, phải). Khung bao không được bao gồm phần đổ bóng hay cắt lẹm vào đối tượng
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?  
**Trả lời:** vật bị che khuất: bowl. Viết thành quy tắc trong guideline. Vì mắt người vẫn có thể nhận biết được đấy là bowl, nhưng cần prompt lại để box không cắt vào vật khác

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-004 ; `class_name`=potted plant ; `score`=0.632235 ; số điểm polygon=`polygon_point_count`=100 ; 3–5 điểm đầu của `polygon_xy`=[1.0,2.0],[1.0,148.0],[3.0,148.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?  
**Trả lời:** polygon bao khít từng đường biên của vật thể , tách biệt hoàn toàn điểm ảnh khỏi phông nền, thể hiện rõ góc nghiêng, hướng đổ của vật thể
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?  
**Trả lời:** instance_id định danh riêng cho từng đối tượng độc lập trong ảnh thay vì class_id dùng để định danh nhiều đối tượng có đặc điểm giống nhau. Không phải tracking_id vì tracking_id dùng để định danh đối tượng qua nhiều frame, trong khi instance_id chỉ có giá trị nội bộ trong 1 bức ảnh tĩnh
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.  
**Trả lời:** Xử lí khoảng trống hoặc cấu trúc rỗng: với các vật thể có khoảng trống ở giữa, phải dùng công cụ tạo lỗ rỗng để loại bỏ phần nền nhìn xuyên qua đó, không tô kín hoàn toàn thành 1 khối đặc
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?  
**Trả lời:.** chỗ bowl 0.55. Hỏi escalation vì bowl 0.55 nằm kẹp giữa bowl 0.74 và bowl 0.70. Cả 3 vật thể đều có cùng màu hồng / trắng ngà, ánh sáng gây lóa làm mất độ tương phản rõ rệt ở mép tiếp xúc khiến mắt thường có phân biệt được đâu là điểm kết thúc của vật thể này và điểm bắt đầu của vật thể kia. Có hiện tượng chồng đè phức tạp, thân của bowl 0.55 bị che khuất 1 phần bởi bowl 0.74. Cuối cùng là guidline chưa có quy định về xử lí tình huống đặc biệt như nên gộp chung thành 1 mảng bột lớn hay cố gắng tách ra thành từng bowl riêng lẻ

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho toàn bộ ảnh:  chỉ gồm class_id và class_name (ví dụ: 468:cab), không có tọa độ | Ảnh có nhiều đối tượng cùng lúc (vừa có taxi, vừa có xe buýt, xe cảnh sát, đường phố); mô hình phân vân với điểm số thấp (cab chỉ đạt ~51%, minibus 16%). | Quan sát chủ thể chính theo guideline (vật thể to nhất, ở trung tâm hoặc chiếm diện tích nổi bật nhất) để chọn đúng 1 nhãn duy nhất. | Kiểm tra xem nhãn được chọn có đúng quy tắc ưu tiên trong guideline không; đảm bảo không gán nhầm sang vật thể phụ ở hậu cảnh.|
| Phát hiện vật thể | Danh sách các khung hình chữ nhật: gồm tọa độ bounding box (bbox_xyxy hoặc [x, y, w, h]) kèm theo class_id/class_name cho từng đối tượng. | Hộp bị trùng lặp, bị đè hoặc vẽ sai kích thước; khi hạ ngưỡng (threshold), mô hình sinh ra nhiều hộp rác (False Positive) hoặc bắt nhầm chi tiết nền. | Vẽ khung chữ nhật bao khít (tight) sát mép 4 phía của từng vật thể; ước lượng phần bị che khuất nhẹ; gán đúng tên lớp cho từng hộp. | Rà soát và xóa các hộp rác; loại bỏ các hộp bị trùng lặp bọc cùng một vật; chỉnh lại ranh giới các hộp bị lệch, quá rộng hoặc quá chật. |
| Instance segmentation | Danh sách đa giác bám sát điểm ảnh: gồm instance_id, class_id, tọa độ polygon (polygon_xy gồm nhiều điểm $(x, y)$) hoặc mặt nạ pixel (mask). | Biên mờ, các vật thể cùng màu chạm/dính vào nhau (như cụm bowl 0.55, bowl 0.74, bowl 0.70) hoặc vật thể có nhiều lỗ rỗng, cành lá phức tạp (potted plant) | Dùng công cụ đa giác/bút vẽ để bám sát đường viền nhìn thấy; khoét lỗ rỗng nếu có; tách biệt ranh giới giữa hai vật chạm nhau; nếu quá mờ thì escalation (báo cáo hỏi ý kiến). | Kiểm tra độ khít của mask ở mức pixel (không lấn ra nền, không cắt lẹm vào vật); kiểm tra xem các vật chạm nhau đã được tách thành các instance_id riêng biệt chưa. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  
**Trả lời:** Làm mờ thông tin định danh cá nhân như khuôn mặt người trong ảnh kitchen và biển số xe trong ảnh traffic
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?  
**Trả lời:** báo cho lab coach

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
