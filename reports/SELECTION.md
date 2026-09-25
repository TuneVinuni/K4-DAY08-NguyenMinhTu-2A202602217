# Vì sao chọn lô này?

Nguồn: `outputs/selection_round1.csv` (268 frame pool, điểm do model cold start tính), `outputs/selection_round1.jpg`
(contact sheet 12 frame được chọn), `outputs/round1_diff.md` (số box tôi đã sửa). Công thức:
`score = 0.5·U + 0.3·A + 0.2·D`; vòng 1 chưa có frame nào đã gán nên `D = 1` cho mọi frame, thứ hạng
chỉ do `U` (độ bất định của 5 box khó nhất) và `A` (số box mơ hồ 0.15 ≤ conf < 0.5) quyết định.
`MIN_GAP_S = 2.0` giây.

## Top 5 nếu chỉ đủ công rà năm ảnh

Trong 50 dòng đầu, tôi không lấy nguyên 5 hạng cao nhất vì hạng 2–6 dồn vào đoạn 130–152 giây, là cùng một
dòng xe đông. Tôi ưu tiên điểm cao nhưng trải đều theo thời gian:

| Thứ tự | Frame | Hạng | Score | t (giây) | U | A | Lý do |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 0.9591 | 72.8 | 0.9182 | 1.0 | Điểm cao nhất, 18 box mơ hồ trên 28 box. Khi sửa, tôi phải thêm 13 box (model bỏ sót nhiều). |
| 2 | frame_0369.jpg | 2 | 0.9324 | 147.6 | 0.9315 | 0.8889 | Cảnh đông (43 box). Chọn thay cho frame_0372 (hạng 6, 148.8 s) và frame_0368 (hạng 9, 147.2 s) vì chỉ cách 1.2 s và 0.4 s, gần như trùng ảnh. Khi sửa phải thêm 23 box, nhiều nhất lô. |
| 3 | frame_0331.jpg | 5 | 0.9154 | 132.4 | 0.8308 | 1.0 | 47 box, 18 box mơ hồ. Chọn thay cho frame_0326 (hạng 4, 130.4 s) và frame_0330 (hạng 12, 132.0 s) vì cùng cụm xe nhưng frame_0331 có nhiều box mơ hồ hơn (18 so với 15 và 16). Khi sửa: xóa 5, thêm 21, vừa có box giả vừa có xe bị bỏ sót. |
| 4 | frame_0099.jpg | 8 | 0.9063 | 39.6 | 0.946 | 0.7778 | Đại diện đoạn đầu video: bốn frame trên đều nằm sau giây 72. U cao (0.946). Là frame tôi quét độc lập: đếm được 26 xe, pre-label chỉ có 13 box. |
| 5 | frame_0270.jpg | 13 | 0.8878 | 108.0 | 0.9089 | 0.7778 | Lấp khoảng trống 73–132 giây. Contact sheet cho thấy có xe tải lớn, một dạng xe khác trong lớp `car`. Tôi bỏ frame_0380 (hạng 3, 152.0 s) vì chỉ cách frame_0369 4.4 s và cùng cảnh đông. |

Về trường hợp model không dự đoán được box: cột `empty` bằng `False` ở cả 268 frame, nên không frame nào được
cộng `EMPTY_BONUS`. Camera này luôn có xe, vì vậy một frame trống sẽ đáng rà trước tiên. Ở vòng 1 không có
trường hợp đó.

## Ba frame thuộc lô 12 ảnh model chọn và bằng chứng

- **frame_0182.jpg** (hạng 1, score 0.9591): U = 0.9182, A = 1.0, 28 box, 18 box mơ hồ. Trên contact sheet đây là
  cảnh đông, nhiều cặp đèn pha sát nhau ở làn trái. `round1_diff.md`: giữ 13, thêm 13, không xóa box nào. Model
  không vẽ sai nhưng bỏ sót một nửa số xe.
- **frame_0331.jpg** (hạng 5, score 0.9154): A = 1.0 (18 box mơ hồ trên 47 box). `round1_diff.md`: 20 box gợi ý,
  giữ 15, xóa 5, thêm 21. `REVIEW_LOG.csv` ghi ba box bị xóa: một box gộp hai vật, một box trên ánh đèn mặt
  đường và một box trùng nằm trong box khác. Ba xe bị cắt ở mép dưới hoặc bật đèn pha được thêm vào.
- **frame_0099.jpg** (hạng 8, score 0.9063): U = 0.946, cao thứ tư trong 50 dòng đầu. `round1_diff.md`: 13 box gợi
  ý, thêm 14, xóa 1. Sau khi sửa còn 26 box, khớp với 26 xe tôi đếm trong `BLIND_SCAN.md` trước khi xem nhãn AI.

## Một frame có điểm cao nhưng không chọn, và một frame điểm thấp hơn vẫn nên xem

- **frame_0372.jpg** (hạng 6, score 0.9101, t = 148.8 s) có điểm cao hơn 7 frame trong lô, nhưng không được chọn
  vì chỉ cách frame_0369 1.2 s, dưới `MIN_GAP_S = 2.0`. Rà frame này tốn công sửa khoảng 40 box, trong khi phần
  lớn là cùng những chiếc xe đã có ở frame_0369, nên model gần như không học thêm được gì. Tôi đồng ý bỏ.
- **frame_0002.jpg** (hạng 18, score 0.8658, t = 0.8 s) nên được xem dù không vào lô: lô vòng 1 không có frame nào
  trước giây 39.6. Khi chạy lại, `selection_round2.csv` chọn frame_0000, frame_0008 và frame_0015. Điều đó cho thấy
  đoạn đầu video bị thiếu ở vòng 1.

## Điều phép chọn này chưa chứng minh về chất lượng mô hình

- U và A cao chỉ cho biết model cold start (học trên COCO) phân vân trên frame đó. Chúng không chứng minh gán nhãn
  frame đó sẽ làm model tốt hơn. Thực tế, sau khi fine-tune trên đúng 12 frame này, AP50 trên tập test giảm từ
  0.771 xuống 0.623 (`reports/rounds_table.md`).
- Độ tin cậy của model COCO chưa được hiệu chỉnh cho cảnh đêm này. Box mơ hồ có thể là xe thật ở xa, cũng có thể
  là ánh đèn phản chiếu. Nhiều box nhỏ dưới 16 px bị bỏ qua khi chấm, dù chúng vẫn làm tăng A.
- Tập test chỉ có 20 ảnh, nhãn tham chiếu do model tạo, và không có lô chọn ngẫu nhiên để đối chứng. Vì vậy chưa
  thể nói chọn theo độ bất định tốt hơn chọn ngẫu nhiên.
