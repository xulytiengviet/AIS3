# AIS3 Sen Việt — Lộ trình triển khai, huấn luyện và đảm bảo chất lượng

Ngày lập: 03/11/2026. Đây là **kế hoạch kỹ thuật đề xuất**, không phải tiến độ đã hoàn thành hoặc kết quả đo của AIS3.

Trang xem trực quan: [Lộ trình AIS3](./index.html#roadmap). Tài liệu chuyên môn: [TinyStories Việt](./VI_TRAINING.md) và [KWS, Unicode audit](./VI_KWS.md).

## 1. Quy ước thời gian

- **Chuẩn bị:** nhân sự, linh kiện, nguồn dữ liệu, cấu hình môi trường, giấy phép, thiết kế thử nghiệm.
- **Huấn luyện/build:** thời gian máy tính chạy training hoặc build/flash. Thời gian huấn luyện biến động mạnh theo GPU, dataset và checkpoint; không phải thời hạn giao sản phẩm.
- **Kiểm thử/đánh giá:** thời gian đo, sửa lỗi, kiểm toán, thu thập bằng chứng và nghiệm thu độc lập.
- **Đã có code ≠ KPI đã đạt:** chưa kiểm thử trên bo thật thì không đánh dấu kết quả đã chấp nhận.

### Khoảng thời gian cấp tuyến (ước tính có điều kiện)

| Tuyến | Chuẩn bị | Huấn luyện / build | Kiểm tra | Tổng thời gian tham khảo |
| --- | --- | --- | --- | --- |
| AIS3 gốc N16R8 | 0,5–1 ngày | Biên dịch/nạp 1–3 giờ, không train lại | 0,5–1 ngày | **1–2 ngày làm việc** |
| Super Mini 4MB | 2–4 giờ | Build/nạp 1–3 giờ, không train lại | 2–4 giờ | **0,5–1,5 ngày** |
| TinyStories tiếng Việt | 3–10 ngày làm corpus, 0,5–3 giờ tokenizer | PLE thử 8–72 giờ máy trên PC/GPU (rất phụ thuộc cấu hình) | 2–5 ngày | **2–5 tuần** nếu chưa có corpus đủ tốt |
| KWS tiếng Việt | 4–14 ngày thu giọng, gán nhãn | CNN host 0,5–6 giờ/lượt, tích hợp DSP/TFLM 5–15 ngày công | 3–7 ngày thực địa | **3–8 tuần** cho nguyên mẫu có số đo thiết bị |
| XiaoZhi DIY | 0,5–1 ngày | Build/flash 2–6 giờ, không huấn luyện LLM | 0,5–1,5 ngày | **1–3 ngày** khi bo/backend tương thích |

Các tuyến có thể làm song song và không cộng các khoảng làm cam kết. Chờ mua thiết bị, quyền dữ liệu, huấn luyện thử thất bại và thay đổi firmware có thể làm tiến độ kéo dài.

## 2. Chín giai đoạn theo giao diện

| Mã / Hạng mục | Chuẩn bị / train-build | Kết quả dự kiến | Kiểm tra, KPI nghiệm thu đề xuất |
| --- | --- | --- | --- |
| **R1 Phần cứng** | 0,5–1 ngày; không train | BOM, ảnh bo, USB, nguồn, Flash/PSRAM, mapping chân riêng từng firmware | 100% bo được đối chiếu Flash/PSRAM thực tế, nguồn và sơ đồ I/O an toàn, không dùng nhầm N16R8 với Super Mini 4MB |
| **R2 Cài AIS3** | 2–5 giờ chuẩn bị + 1–3 giờ build/flash | Log build, checksum, UART/OLED, snapshot baseline mô hình gốc | 10/10 boot không crash; model/partition đúng; hash/golden pass; ghi tốc độ và RAM thực tế |
| **R3 Super Mini** | 2–4 giờ chuẩn bị + 1–3 giờ build/flash | Firmware micro-model riêng và SoftAP/web UI trong LAN | 10/10 lần khởi động; 2 thiết bị truy cập trang 192.168.4.1; không nạp model AIS3 14,9MB vào Flash 4MB |
| **R4 Tiếng Việt** | Corpus 3–10 ngày; tokenizer 0,5–3 giờ; PLE thử 8–72 giờ máy | Corpus UTF-8, train/val, tokenizer + SHA, checkpoint .pt, INT4 và log đánh giá | Round-trip 100%, unknown=0 theo ByteLevel; PPL so với baseline **cùng test set**; ≥30 prompt mù chấm Grammar/Consistency; kích thước model vừa phân vùng |
| **R5 Audit** | 0,5–2 ngày ban đầu + mỗi lần build | Báo cáo JSON/CSV, hash, bộ test golden, chứng cứ CI, provenance | Không có lỗi nghiêm trọng chưa xử lý; có train/val/test tách biệt; kết quả PC–device parity có dung sai công bố trước |
| **R6 Giọng nói KWS** | 4–14 ngày thu giọng; CNN host 0,5–6 giờ/lượt; port DSP/TFLM 5–15 ngày công | WAV theo speaker, MFCC manifest, INT8 TFLite, firmware mẫu, confusion matrix & số đo on-device | Mục tiêu Macro-F1 ≥0,90, recall lớp quan trọng ≥0,90, false accept ≤1/giờ, p95 cuối lời nói→tác động ≤500ms; là **đích, chưa phải kết quả** |
| **R7 DIY XiaoZhi** | 0,5–1 ngày chọn bo; 2–6 giờ build/flash | Mic/loa, kết nối Wi-Fi, dịch vụ kích hoạt, log ASR→LLM→TTS | 20/20 yêu cầu thoại trong mạng test có phản hồi, 3/3 reconnect, đo response latency thực; không lộ khóa/tokens |
| **R8 Xử lý lỗi** | 0,5–1 ngày lập ma trận, 1–3 ngày khắc phục tùy sự cố | Issue register, log lỗi, bài thử hồi phục/rollback | 0 blocker, 3/3 rollback firmware thành công, 24 giờ soak không crash không giải thích (mục tiêu kiểm thử) |
| **R9 Nghiệm thu** | 0,5–1 ngày tổng hợp kết quả | Manifest release, bảng KPI có số đo, quyết định, hướng dẫn cập nhật & rollback | 100% KPI bắt buộc có nguồn/log/ngày/đơn vị đo/người kiểm; khuyết điểm có issue hoặc quyết định waiver |

## 3. Lịch theo tuần tham khảo (không phải tiến độ đã diễn ra)

| Giai đoạn | Ưu tiên | Cổng cần qua |
| --- | --- | --- |
| Tuần 1 | R1, R2, R3; chạy baseline trên bo đúng loại | **G0** phần cứng / tính khả thi |
| Tuần 2 | R4 chuẩn hóa corpus và tokenizer; R6 thu giọng KWS; R5 thiết kế audit | **G1** dữ liệu hợp lệ, train/val/test tách biệt |
| Tuần 3 | PLE thử trên PC/GPU; CNN host/INT8; R7 XiaoZhi nếu có bo/server | **G2** đánh giá host và lỗi dữ liệu |
| Tuần 4–5 | Nạp TinyStories Việt trên N16R8; port MFCC/TFLM; thử ồn/giọng vùng miền; R8 | **G3** parity PC–device, đo FPS/tok/s/latency/RAM thật |
| Tuần 6+ nếu cần | Điều chỉnh mô hình, đo thực địa, kiểm tra hồi phục, release | **G4** rollout, hồ sơ KPI và nghiệm thu |

Không ghép XiaoZhi, KWS và TinyStories INT4 vào cùng firmware khi chưa có thiết kế dung lượng, phân vùng, bộ nhớ, tác vụ và quyền sử dụng thành phần mã nguồn.

## 4. Cổng chất lượng và quy trình QA

**G0 — Feasibility:** xác nhận BOM, mã bo, nguồn, mapping GPIO, Flash/PSRAM, giấy phép dữ liệu và ngân sách công việc. **Dừng** nếu model lớn hơn phân vùng hoặc dùng sai bo.

**G1 — Reproducibility:** khóa dependencies, ghi hash dataset/tokenizer/model, random seed, toolchain, tập người nói/truyện tách biệt và đầu vào hợp lệ. Không tự đánh dấu PASS nếu không có log/manifest.

**G2 — Quality on host:** PLE cần audit NFC/round-trip, PPL và rubric ngữ pháp; KWS cần Macro-F1, recall, confusion matrix, negative samples và kiểm tra lượng tử hóa INT8, không dựa vào accuracy đơn lẻ.

**G3 — Device parity:** chạy trên ESP32-S3 thật; so output golden/feature MFCC và score giữa máy tính–MCU, log throughput/timing/heap/arena/Flash/PSRAM và độ ồn. Số 94%/368ms của nghiên cứu và ~51ms/frame của Voicute không phải kết quả AIS3.

**G4 — Release:** soak test, phiên bản firmware + checksum, quyền mã nguồn/model, rollback, kiểm soát bí mật và biên bản người đánh giá.

## 5. Công thức & phương pháp tính KPI

- **Macro-F1:** trung bình cộng F1 từng lớp trên tập test theo speaker; không lấy weighted-F1 thay khi các lớp lệch số mẫu.
- **Recall lệnh quan trọng:** TP/(TP+FN); báo riêng từng nhãn (bật/tắt có thể gây tác động vật lý).
- **False accepts/hour:** số lần kích hoạt nhầm khi không có từ khóa / tổng giờ âm tính ghi âm; tối thiểu 10 giờ là bài test ban đầu, **0 false trong 10 giờ chưa chứng minh tỷ lệ thực ≤1/h với độ tin cậy cao**.
- **P95 latency:** phân vị 95 của thời gian từ *kết thúc lời nói* đến tín hiệu điều khiển; phải định nghĩa đúng mốc đo. Đo tối thiểu 100 lần trên bo, ghi cả median và P95.
- **Tiếng Việt BPE:** đo chính xác decode(encode(x)) == x trên tập kiểm thử; unknown=0; token/100 âm tiết, byte/token, PPL trên cùng tokenizer và tập test. NFC **không** tự chuyển hóa↔hoá.
- **Điểm truyện kể:** ít nhất 30 prompt kiểm thử không có trong train, hai người chấm mù theo thang 1–10 về Grammar/Creativity/Consistency; mức ≥7/10 là **mục tiêu giả định cần phê duyệt**, không phải tiêu chuẩn quốc tế.
- **Độ bền:** ít nhất 24 giờ hoạt động, log restart, điện năng, bộ nhớ, 3 lần rollback, các tình huống USB/Wi-Fi/mất điện/khôi phục.

Không có dữ liệu thực → `PENDING`. Có đo nhưng không đạt → `FAIL` và ghi issue. `PASS` yêu cầu giá trị, đơn vị, thời gian, phiên bản, chứng cứ, người đánh giá. `WAIVED` phải có lý do và chữ ký quyết định.

## 6. Mẫu ghi nhận KPI

Tải file [roadmap-kpi-template.csv](./roadmap-kpi-template.csv) và điền các trường: phase, track, metric, target, unit, measured, evidence_path, date, owner, reviewer, decision, notes. Mỗi KPI bắt buộc được kiểm thử lại sau thay đổi model/firmware.

## 7. Rủi ro và phụ thuộc

- **P0/blocker:** Flash/PSRAM không đúng, ESP32 sai sơ đồ chân, model vượt phân vùng, thiếu quyền dữ liệu, checkpoint sai tokenizer, cấp nguồn không an toàn.
- **P1:** ít người nói/giọng vùng miền, đánh giá rò rỉ, MFCC host–MCU sai khác, unknown/noise không đầy đủ, backend XiaoZhi không ổn định, thiếu UART log.
- **P2:** giao diện OLED không có đủ glyph tiếng Việt, trình bày dashboard, tài liệu người dùng chưa bao quát.

## 8. Nguồn và trạng thái

- [AIS3 gốc và benchmark](https://github.com/xulytiengviet/AIS3/blob/main/RESULTS.md) — TinyStories, Barista, Fly.
- [Super Mini AI](https://github.com/Ngducok/esp32s3-Super-mini-AI) — micro-model độc lập cho ESP32-S3 4MB.
- [XiaoZhi Việt](https://github.com/Ngthhoang/xiaozhi-vn-ricky0611) — trợ lý giọng nói và phụ thuộc backend.
- [Bài báo KWS tiếng Việt PTIT 2025](https://tapchi.ptit.edu.vn/jstic-ptit/index.php/jstic/article/view/1668) — số liệu nghiên cứu, không phải benchmark AIS3.
- [Voicute onnx-wakeword](https://github.com/voicute/onnx-wakeword) — Causal TCN/Mel runtime, chưa có tiếng Việt chính thức trong README đã kiểm tra.

**Chưa được xác nhận:** checkpoint PLE tiếng Việt, kết quả KWS trên bo AIS3, các ngưỡng chấp nhận, và việc tích hợp AIS3+XiaoZhi chung một firmware. Mọi thời gian/KPI ở đây là kế hoạch để quản trị thử nghiệm, không phải cam kết đạt.
