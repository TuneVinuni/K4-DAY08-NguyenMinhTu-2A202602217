# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Minh Tú

Công cụ gán nhãn đã dùng: CVAT v2.76.0 chạy bằng Docker trên máy cá nhân. Nhập pre-label và xuất nhãn theo định dạng Ultralytics YOLO Detection 1.0.

Nguồn số liệu: `reports/rounds_table.md`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`,
`outputs/selection_round1.csv`, `outputs/selection_round2.csv`, `outputs/round1_diff.md`,
`outputs/compare_round0.jpg`, `outputs/compare_round1.jpg`. Nhãn test do một mô hình tạo và chưa được người
rà, nên mọi số đo dưới đây là **mức khớp với bộ tham chiếu này**, không phải độ đúng tuyệt đối.

## 1. Dữ liệu và cách chia tập

Camera đặt cố định, cảnh quay liên tục 160 giây, lấy mẫu 2.5 frame/giây. Hai frame liền nhau chỉ cách 0.4 giây
và gần như giống hệt nhau, còn mỗi chiếc xe ở trong khung hình vài giây (`data/DATA.md`). Nếu chia ngẫu nhiên,
cùng một chiếc xe ở cùng vị trí sẽ xuất hiện ở cả pool/train lẫn test. Khi đó model được chấm trên chính những
xe nó đã học. Đây là rò rỉ dữ liệu, và số đo sẽ **cao hơn thực tế** (lệch lạc quan), vì model chỉ cần "nhớ" cảnh
chứ không cần tổng quát hóa.

Chia theo trục thời gian loại bỏ rủi ro đó: test gồm 20 frame ở 4 đoạn có tâm tại giây 20, 60, 100 và 140. Mỗi
đoạn có vùng đệm 4 giây bị bỏ đi (112 frame), pool còn 268 frame. Frame pool gần test nhất vẫn cách 4.4 giây. Nhờ
vậy, test đo được khả năng nhận xe ở thời điểm model chưa thấy. Tuy vậy, nền ảnh vẫn là cùng một cảnh, nên kết
quả chưa nói gì về camera hoặc góc quay khác.

## 2. Mô hình khởi đầu lạnh (cold start)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

`metrics_round0.json`: trên 403 box tham chiếu (bỏ qua 14 box cao dưới 16 px), model có 197 TP, 16 FP và 206 FN.
Precision cao nhưng recall chưa tới một nửa, nghĩa là model ít vẽ sai mà chủ yếu **bỏ sót**.

Trong `compare_round0.jpg` (xanh lá = khớp, vàng = tham chiếu bị bỏ sót, đỏ = dự đoán thừa):

- **Xe nhỏ ở xa**, ở hàng xe dồn về phía chân cầu và làn đối diện chỉ còn đèn hậu đỏ, phần lớn là box vàng.
  Recall xe nhỏ chỉ 0.182 (66 box), thấp hơn nhiều so với xe trung bình (0.547) và xe lớn (0.561).
- **Xe gần camera nhưng bị lóa đèn pha** cũng bị bỏ sót: frame_0250 có xe lớn ở góc dưới bên trái, frame_0350
  có xe ở dưới bên trái và giữa bên phải. Như vậy recall xe lớn chỉ 0.561 không hẳn do xe nhỏ: ánh đèn pha chói
  làm mất đường viền thân xe.
- **Dự đoán thừa** thường là box gộp nhiều xe sát nhau: frame_0350 có một box đỏ lớn ôm cụm xe làn trái,
  frame_0150 có các box chồng nhau ở cụm đèn hậu bên phải.

Ca cần người rà lại nhãn tham chiếu trước khi kết luận model sai: **frame_0150, cụm đèn hậu đỏ ở bên phải**.
Tham chiếu vẽ ba box chồng lên nhau trên một vùng có vệt phản chiếu đỏ trên mặt đường. Model cold start bị tính
FP ở đó. Cần xem ảnh gốc xem thật sự có ba xe, hay tham chiếu đã tính cả vệt phản chiếu (theo guideline thì
không gán nhãn vệt phản chiếu). Tương tự, frame_0250 có một box tham chiếu rất hẹp sát mép phải; cần xác nhận
đó là xe bị cắt ở mép chứ không phải ánh đèn.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D` với trọng số 0.5 / 0.3 / 0.2 (`tools/al_select.py`):

- **U** (độ bất định): mỗi box có `u = 1 − |2·conf − 1|`, lớn nhất khi conf = 0.5. U là trung bình 5 box khó nhất
  của frame, tức frame nào có box mà model "phân vân nhất" thì U cao.
- **A** (mơ hồ): số box có 0.15 ≤ conf < 0.5, chia cho giá trị lớn nhất trong pool. A cao nghĩa là frame có nhiều
  box lưng chừng, cũng là nhiều chỗ con người cần rà.
- **D** (đa dạng): khoảng cách thời gian tới frame đã gán gần nhất, chặn ở 10 giây. Ở vòng 1 chưa có frame nào
  được gán nên D = 1 cho mọi frame.
- **MIN_GAP_S = 2.0**: khi chọn tham lam theo score, bỏ frame cách một frame đã chọn dưới 2 giây. Camera cố định
  nên hai frame gần nhau gần như trùng nhau. Gán cả hai tốn gấp đôi công mà model học thêm rất ít.

Dẫn chứng từ `reports/SELECTION.md` và `selection_round1.csv`:

- frame_0182 (hạng 1, U 0.918, A 1.0) và frame_0331 (hạng 5, A 1.0, 47 box) được chọn vì có nhiều box mơ hồ nhất.
  Khi rà, frame_0331 cần xóa 5 box và thêm 21 box.
- frame_0099 (hạng 8, U 0.946) được chọn và nằm ở đoạn đầu video. Đây là frame tôi quét độc lập.
- **Frame khác:** frame_0372 (hạng 6, score 0.9101) không được chọn vì chỉ cách frame_0369 1.2 giây. Đây là
  MIN_GAP_S đang tiết kiệm công gán nhãn cho một ảnh gần trùng.
- Công gán nhãn: 12 frame cần thêm 187 box, trung bình khoảng 15.6 box/frame (`round1_diff.md`). Frame càng đông
  thì U và A càng cao, nhưng cũng càng tốn công.

Điểm bất định **không chứng minh** ảnh đó sẽ cải thiện model. Nó chỉ đo sự phân vân của model hiện tại, mà độ tin
cậy của model COCO chưa được hiệu chỉnh cho cảnh đêm. Box mơ hồ có thể là ánh đèn phản chiếu hoặc xe dưới 16 px
không được chấm. Kết quả vòng 1 (AP50 giảm) cho thấy chọn đúng ảnh "khó" chưa đủ để model tốt lên.

## 4. Các vòng học chủ động (active learning)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 343 | 0.623 | -0.149 | 1.000 | 0.102 | 0.185 | 0.000 | 0.081 | 0.415 |

**Mức độ sửa nhãn gợi ý (`outputs/round1_diff.md`):** model đề xuất 169 box. Tôi giữ nguyên 152, chỉnh 4, xóa 13
(box giả của model) và thêm 187 (xe model bỏ sót), còn lại 343 box. Số xe bị bỏ sót nhiều gấp khoảng 14 lần số
box giả. Lỗi chính của pre-label là **thiếu xe**, không phải vẽ sai.

**Thay đổi số đo:** AP50 giảm 0.149 so với cold start (0.771 → 0.623). Đây là vòng đầu nên so với vòng trước
cũng là mức giảm này. P tăng lên 1.000 (FP từ 16 xuống 0), nhưng R giảm từ 0.489 xuống 0.102 (TP từ 197 xuống 41).
Theo kích thước, mọi nhóm đều xấu đi: xe nhỏ 0.182 → 0.000, xe trung bình 0.547 → 0.081, xe lớn 0.561 → 0.415.
Nhóm xe lớn giữ được nhiều nhất.

**Ca thay đổi sau fine-tune (`compare_round1.jpg`):**

- *Tốt hơn:* ở frame_0350, box đỏ lớn ôm cả cụm xe làn trái của cold start đã biến mất. Cả 4 frame so sánh đều
  có FP = 0 (cold start có 2 FP mỗi frame).
- *Xấu hơn:* ở frame_0050, TP giảm từ 11 xuống 4. Các xe trung bình ở giữa đường vốn được cold start phát hiện
  đều chuyển sang vàng (bỏ sót); chỉ còn các xe lớn gần camera.

**Lý do có thể kiểm:** model không hẳn "không thấy" xe mà cho **độ tin cậy quá thấp**.

- AP50 tính trên mọi dự đoán có conf ≥ 0.01 vẫn đạt 0.623, trong khi R tại conf 0.25 chỉ còn 0.102.
- `selection_round2.csv` (do model vòng 1 tính trên pool) chỉ có 1–10 box có conf ≥ 0.05 mỗi frame (trung bình 5.5), trong khi
  `selection_round1.csv` của cold start có 14–53 box (trung bình 27.9).
- Nguyên nhân hợp lý: `data.yaml` chỉ có 1 lớp nên đầu phân loại của yolov8n (80 lớp COCO) bị khởi tạo lại. Với
  12 ảnh và batch 16, mỗi epoch chỉ có 1 bước cập nhật, 50 epoch chỉ khoảng 50 bước. Chừng đó chưa đủ để đầu
  mới học cho điểm tin cậy cao.
- Kiểm tra nhất quán: `metrics_round1.json` ghi `n_train_boxes = 343`, đúng bằng số box sau khi sửa trong
  `round1_diff.md`. Model đã train đúng trên lô đã sửa.

**Phân biệt ba nguồn bằng chứng:**

- *Quan sát độc lập* (`BLIND_SCAN.md`): trước khi xem nhãn AI, tôi đếm được 26 xe trong frame_0099 và dự đoán hai
  chỗ dễ sai: xe bị cắt ở mép dưới phải, và các xe rất xa gần chân cầu.
- *Lỗi pre-label đã sửa* (`REVIEW_LOG.csv`, `round1_diff.md`): frame_0099 có 13 box gợi ý, tôi thêm 14 và xóa 1,
  còn 26 box, khớp số xe đã đếm. Xe bị cắt ở mép phải và xe nhòe ở mép dưới phải được giữ đúng như dự đoán trong
  bản quét. Ở frame_0331, tôi xóa một box gộp hai vật, một box nằm trên ánh đèn mặt đường và một box trùng.
- *Model sau train* (`metrics_round1.json`): dù nhãn train đã đầy đủ hơn, model vòng 1 vẫn bỏ sót nhiều hơn cold
  start. Nhãn tốt hơn không tự động cho ra model tốt hơn khi lô quá nhỏ và số bước huấn luyện quá ít.

**Ca khó theo guideline và cách xử lý thống nhất:**

- *Xe bị cắt ở mép ảnh:* frame_0331 có một xe tối ở góc dưới bên trái và một xe ở giữa mép dưới chỉ còn nóc và
  đèn. Tôi chỉ vẽ box phần nằm trong ảnh, tới sát mép.
- *Xe rất xa:* 46 trên 343 box (13%) cao dưới 16 px. Quy tắc tôi dùng cho mọi ảnh: vẫn gán box khi còn tách được
  một xe riêng (thấy hai đèn hoặc đường viền thân xe), không gán khi chỉ là một đốm sáng không rõ là xe. Các box
  này không được tính khi chấm, nhưng tôi giữ để model không học rằng xe ở xa là nền.

## 5. Kết luận và giới hạn

**So với cold start:** vòng 1 kém hơn theo AP50 (−0.149) và recall (0.489 → 0.102), chỉ tốt hơn ở precision
(0.925 → 1.000). Với lô 12 ảnh, fine-tune làm model **thận trọng quá mức** thay vì phát hiện thêm xe.

**Quyết định: dừng ở vòng 1 cho bài nộp.** Nếu làm tiếp vòng 2 với cùng cách huấn luyện và thêm 12 ảnh nữa, tổng
cũng chỉ khoảng 24 ảnh, tức khoảng 100 bước cập nhật. Khó kỳ vọng khắc phục được việc độ tin cậy bị sụp. Ngoài ra,
`selection_round2.csv` được tính từ model vòng 1 vốn chỉ đưa ra 1–10 box mỗi frame, nên điểm U và A của vòng 2 kém
tin cậy. Nếu có thêm thời gian, tôi sẽ làm vòng 2 để so xu hướng, không phải để kỳ vọng AP50 tăng.

**Hai ca còn yếu cho vòng sau:**

1. **Xe nhỏ ở xa**, ở hàng xe dồn về chân cầu và làn đối diện: recall bằng 0.000 sau vòng 1. Chi phí rà cao, vì
   mỗi frame đông có 15–20 xe nhỏ, nhiều xe sát ngưỡng 16 px và khó gán nhất quán. Nên chọn frame có nhiều xe
   nhỏ ở mức 16–30 px (được chấm) thay vì frame toàn đốm sáng.
2. **Xe gần camera bị lóa đèn pha**, như frame_0250 góc dưới trái: bị bỏ sót ở cả cold start lẫn vòng 1. Ít xe
   mỗi frame nên chi phí rà thấp, nhưng các frame có kiểu xe này nằm liền nhau trong pool. Nguy cơ ảnh gần trùng
   cao, cần giữ MIN_GAP_S để không chọn nhiều frame của cùng một chiếc xe.

Đoạn đầu video (0–39 giây) chưa có frame nào trong lô vòng 1. `selection_round2.csv` đã chọn frame_0000,
frame_0008 và frame_0015, phù hợp để bù phần thiếu này.

**Giới hạn ảnh hưởng tới kết luận:**

- *Tập test nhỏ* (20 ảnh, 403 box): một vài xe khớp hoặc trượt cũng làm số đo đổi đáng kể. Tuy vậy, mức giảm
  0.149 AP50 lớn hơn nhiều so với ngưỡng nhiễu khoảng 0.01 trong `DATA.md`, nên xu hướng giảm là thật.
- *Luật bỏ qua xe nhỏ dưới 16 px:* 14 box tham chiếu không được tính. Các box dưới 16 px tôi gán (13% nhãn train)
  không được thưởng khi chấm, và recall "xe nhỏ" chỉ đo nhóm từ 16 px trở lên.
- *Nhãn tham chiếu do model tạo:* một số box "vàng" (bỏ sót) hoặc "đỏ" (thừa) có thể do tham chiếu sai, như cụm
  đèn hậu ở frame_0150. Nếu model tham chiếu có cùng điểm yếu với cold start (cũng là model phát hiện), AP50 có
  thể đang ưu ái model giống nó hơn là model học theo nhãn người.

**Nếu AP50 giảm, cần kiểm tra trước khi train thêm:**

1. Nhãn train có được nạp đúng không: số ảnh và số box trong `metrics_round1.json` phải khớp `round1_diff.md`
   (12 ảnh, 343 box, đã khớp).
2. Phân bố độ tin cậy: tính lại P và R ở ngưỡng thấp hơn (ví dụ 0.05 hoặc 0.1). Nếu R phục hồi thì vấn đề là
   hiệu chỉnh độ tin cậy, không phải model không thấy xe.
3. Nhãn có nhất quán không, nhất là box xe xa và xe bị cắt ở mép, để không dạy model hai quy tắc trái nhau.
4. Chạy chiến lược `random` với cùng số ảnh để đối chứng, trước khi kết luận chọn theo độ bất định có ích hay
   không.
