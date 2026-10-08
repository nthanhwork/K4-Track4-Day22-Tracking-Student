# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Phronesis 


**Thành viên:** 
- Nguyễn Thái Anh - 2A202602810
- Nguyễn Văn Giáp - 2A202602903

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

## 1. Cấu hình đã chọn

Đã chạy 37 lượt so sánh: 9 cấu hình trên đủ 600 frame của `video_1`, và 7 cấu hình trên 150 frame đầu của mỗi video còn lại. Với mỗi video, so sánh một tracker chuyển động với một tracker có Re-ID tại cùng `conf=0.25`, `iou=0.5`; sau đó chỉ thay đổi một ngưỡng mỗi lượt. Năm cấu hình chọn cuối cùng đã được chạy lại **đủ frame**, bằng RTX 3060 với `--device cuda:0`.

Ngưỡng đã thử: `conf` 0.15 / 0.25 / 0.3 / 0.5 và `iou` 0.4 / 0.5 / 0.7. Các lượt thử không tạo thành một phép quét mọi tổ hợp. Chi tiết 37 lượt, cấu hình, thời gian và số liệu `video_1` được lưu trong [`bang_thu_nghiem.csv`](../runs/so_sanh/bang_thu_nghiem.csv).

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | strongsort | 0.15 | 0.4 | Người gần camera được theo dõi; nhiều người nhỏ ở xa còn thiếu hộp. Cấu hình này có HOTA cao nhất trong 9 lượt chấm. | bytetrack, conf=0.25, iou=0.5: HOTA 27.422, IDF1 26.647, thấp hơn cấu hình chọn. |
| video_2 (phố đêm, tĩnh, rất đông) | strongsort | 0.15 | 0.5 | Tại frame 150, hạ conf bắt thêm người trong nhóm gần đèn phía trên trái so với conf=0.5. Các frame 210–1050 vẫn có người trong nhóm xa chưa được gắn hộp. | strongsort, conf=0.5, iou=0.5: mất nhiều hộp trong nhóm đông ở frame 90 và 150. |
| video_3 (camera di động, ảnh nhỏ) | botsort | 0.25 | 0.5 | Người áo sọc và người áo khoác xanh giữ ID 1 và 2 tại các frame 30, 60, 90, 120, 150. Các đoạn sau có người sát camera bị cắt thân và hộp chồng nhau. | botsort, conf=0.5, iou=0.5: thiếu hộp của người nhỏ ở xa tại các frame 90 và 150 so với conf=0.25. |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.25 | 0.5 | Người áo đỏ và áo trắng giữ ID 2 và 6 tại các frame 30–150; cả hai vẫn có hộp ở các frame 180, 360, 540. Khi người đi ngược chiều tới gần, hộp lớn và che nhau. | ocsort, conf=0.25, iou=0.5: cũng giữ hai ID chính trong đoạn đầu; chọn botsort để dùng thêm ngoại hình và bù chuyển động camera. Chưa thấy bằng chứng vượt trội trên toàn video. |
| video_5 (trên xe bus, giao lộ đông) | botsort | 0.25 | 0.5 | Còn bỏ sót nhiều người nhỏ ở xa. Tại frame 450, nhóm ở góc đường có nhiều hộp hơn khi xe tới gần; sau khi xe rẽ, bối cảnh và các track thay đổi. | botsort, conf=0.5, iou=0.5: ở frame 90, nhóm vỉa hè bên trái mất nhiều hộp so với conf=0.25. |

Số frame đã xử lý trong bản nộp:

| File | Số frame nguồn và đã xử lý |
|---|---:|
| `video_1.txt` | 600 |
| `video_2.txt` | 1050 |
| `video_3.txt` | 837 |
| `video_4.txt` | 900 |
| `video_5.txt` | 750 |

Các file nằm trong `runs/nop_bai/`, kèm video có vẽ ID, log chạy và [`cau_hinh.json`](../runs/nop_bai/cau_hinh.json). Số dòng trong file tracking là số hộp, không phải số frame; một frame có thể chứa nhiều hộp hoặc không có hộp.

## 2. Số liệu video_1

Kết quả chấm **file bản nộp cuối cùng**, đủ 600 frame, bằng `scripts/evaluate_practice.py`:

```
Video       Tracker      conf    iou     HOTA      MOTA      IDF1
video_1     strongsort   0.15    0.4     29.214    20.295    32.678
```

Các chỉ số được trình bày theo thang 0–100 của TrackEval. Output đầy đủ được lưu trong [`video_1_evaluation.log`](../runs/nop_bai/video_1_evaluation.log) và [`video_1_metrics.txt`](../runs/nop_bai/video_1_metrics.txt).

So sánh các cấu hình trên **cùng toàn bộ video_1**:

| Tracker | conf | iou | HOTA | MOTA | IDF1 |
|---|---:|---:|---:|---:|---:|
| bytetrack | 0.25 | 0.5 | 27.422 | 17.582 | 26.647 |
| strongsort | 0.25 | 0.5 | 28.475 | 20.408 | 29.975 |
| strongsort | 0.15 | 0.5 | 29.189 | 19.918 | 32.574 |
| strongsort | 0.30 | 0.5 | 28.662 | 19.719 | 29.870 |
| strongsort | 0.50 | 0.5 | 25.109 | 15.989 | 22.623 |
| strongsort | 0.25 | 0.4 | 28.480 | 20.494 | 30.029 |
| strongsort | 0.25 | 0.7 | 27.968 | 20.101 | 29.564 |
| **strongsort, chọn nộp** | **0.15** | **0.4** | **29.214** | **20.295** | **32.678** |
| strongsort | 0.15 | 0.7 | 28.411 | 17.771 | 32.437 |

Cấu hình chọn có 13.372 hộp bỏ sót (FN), 1.346 hộp giả (FP), 92 lần đổi ID (IDSW), trên 18.581 hộp nhãn người được chấm. So với StrongSORT tại conf=0.25, iou=0.5, hạ conf và đổi iou giúp giảm FN từ 14.334 xuống 13.372, nhưng tăng FP từ 398 lên 1.346 và IDSW từ 57 lên 92. Vì vậy HOTA/IDF1 tăng nhưng MOTA giảm nhẹ; cấu hình chọn không tốt nhất trên mọi chỉ số.

Gói dữ liệu hiện tại thiếu `video_1/eval_config.json`. Đã tạo cấu hình **cục bộ** `benchmark=LAB22`, `split=train` để chỉ định thư mục staging; giữ cách tiền xử lý nhãn và các metric mặc định của script. Nhãn gốc không được sửa. Script được bổ sung bản vá alias NumPy trong tiến trình con để tương thích với TrackEval. Nếu giảng viên cung cấp cấu hình chấm khác, cần dùng cấu hình đó và chấm lại.

`video_2` đến `video_5` không có nhãn trong gói lab; không có HOTA/MOTA/IDF1 cho các video này.

## 3. Phân tích

**Video_1.** Ở cùng conf=0.25 và iou=0.5, StrongSORT đạt HOTA 28.475 và IDF1 29.975, cao hơn ByteTrack lần lượt 27.422 và 26.647. Cảnh có người đi ngang và che nhau nên ngoại hình là một nguồn thông tin bổ sung hợp lý khi liên kết hộp; số liệu hỗ trợ lựa chọn StrongSORT trên video này, dù IDSW không thấp hơn ByteTrack. Hạ conf xuống 0.15 giúp giữ thêm phát hiện yếu và giảm bỏ sót, nhưng làm tăng hộp giả. Chọn iou=0.4 vì HOTA cao nhất trong các lượt đã thử; chênh lệch với iou=0.5 chỉ 0.025 điểm, chưa đủ để kết luận lựa chọn này luôn tốt hơn ở cảnh khác.

**Video_2.** Tại frame 90 và 150, StrongSORT với conf=0.15 giữ thêm hộp trong nhóm đông gần đèn so với conf=0.5. Cảnh ban đêm và người che nhau khiến việc loại hết phát hiện confidence thấp dễ dẫn tới bỏ sót, nên chọn ngưỡng 0.15. Re-ID có thể hỗ trợ phân biệt người khi đường đi giao nhau, nhưng ảnh người nhỏ và chói sáng cũng hạn chế thông tin ngoại hình. Các frame kiểm tra trải tới frame 1050 vẫn có người xa không được gắn hộp, nên lựa chọn này chưa giải quyết hết lỗi và chưa được xác nhận bằng metric.

**Video_3.** Trong đoạn đầu, BoT-SORT giữ ID 1 cho người áo sọc và ID 2 cho người áo khoác xanh ở các frame 30–150; OC-SORT cũng giữ được hai ID này. Camera tiến gần làm kích thước hộp thay đổi rất mạnh, nên chọn BoT-SORT để dùng thêm bù chuyển động camera và ngoại hình, nhưng đoạn quan sát này chưa chứng minh nó vượt trội OC-SORT. Với conf=0.5, hai người chính vẫn có hộp, nhưng các người nhỏ ở xa có hộp tại conf=0.25 ở frame 90 và 150 lại bị bỏ sót, nên giữ conf=0.25. Các frame sau cho thấy nhiều người ở sát camera, cắt khỏi mép ảnh và che nhau; đây vẫn là giới hạn của cấu hình đã chọn.

**Video_4.** BoT-SORT giữ ID của hai người áo đỏ và áo trắng ở các frame 30–150 trong khi camera tiến tới; OC-SORT cũng làm được ở đoạn này. Chọn BoT-SORT vì cảnh có người đi ngược chiều, giao nhau và camera chuyển động, phù hợp với việc kết hợp ngoại hình và bù chuyển động. Giữ conf=0.25, iou=0.5 vì các ngưỡng đã thử chưa tạo ra cải thiện rõ ràng bằng mắt trên các người chính. Các frame 540–900 có nhiều người rất gần máy quay và bề mặt phản chiếu, cần kiểm tra kỹ hơn trước khi khẳng định khả năng tránh nhầm ID hoặc hộp giả.

**Video_5.** Khi xe chạy và rẽ, nền ảnh dịch chuyển mạnh, người ở vỉa hè thay đổi vị trí và kích thước nhanh. BoT-SORT được chọn để kết hợp bù chuyển động camera và Re-ID; conf=0.25 giữ nhiều hộp hơn conf=0.5 trong nhóm bên trái ở frame 90. Ở frame 450, người ở góc đường gần camera có hộp rõ hơn các nhóm nhỏ ở xa trong frame 150 và 300. Khi xe rẽ, nhiều người rời tầm nhìn và track mới xuất hiện; không xem việc ID mới xuất hiện là bằng chứng đổi danh tính nếu chưa đối chiếu cùng một người.

Phần quan sát dựa trên các frame 1, 30, 60, 90, 120, 150 của lượt thử và 6 frame trải đều toàn bộ mỗi bản nộp; không phải kiểm tra liên tục mọi frame. Ảnh bằng chứng nằm trong `runs/so_sanh/` và các file `runs/nop_bai/video_N_kiem_tra.jpg`. Các video đầy đủ có vẽ ID được lưu để xem lại các đoạn che khuất và giao nhau.

## 4. Nếu có thêm thời gian

Sẽ kiểm tra liên tục các đoạn giao nhau và che khuất, đồng thời so sánh các cấu hình trên nhiều đoạn hơn của video_2–video_5 thay vì chỉ 150 frame đầu. Với video_1, sẽ quét conf mịn quanh 0.15 và đối chiếu HOTA, IDF1, MOTA cùng FP/FN để cân bằng bỏ sót và hộp giả.
