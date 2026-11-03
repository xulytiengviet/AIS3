# AIS3 Sen Việt · Chuẩn hóa tiếng Việt và nhận diện lệnh KWS trên ESP32-S3

Ngày: 03/11/2026. Trạng thái: **module thử nghiệm, chưa xác nhận chạy KWS trên firmware AIS3**.

## 1. Ba kết quả khác nhau, không được trộn số liệu

| Dự án | Bản chất | Trạng thái Việt ngữ | Số liệu |
| --- | --- | --- | --- |
| [Bài báo PTIT/VJOL (2025)](https://tapchi.ptit.edu.vn/jstic-ptit/index.php/jstic/article/view/1668) | KWS DSP + MFCC + CNN trên ESP32-S3; 1 wake word + 8 lệnh | Tiếng Việt thực nghiệm | **129.098 tham số; accuracy trung bình 94%; 368 ms** cho chu trình cửa sổ 500–1.000 ms |
| [voicute/onnx-wakeword](https://github.com/voicute/onnx-wakeword) | Causal TCN, Mel, head tương thích; ONNX trên PC/web, INT8 TFLite ESP32 | 7 tiếng hỗ trợ, **chưa công bố tiếng Việt** | Demo ESP32 ~51 ms/khung theo README, **không** phải benchmark bài báo |
| [nguyenthaozzz/edge-ai-voice-control-esp32](https://github.com/nguyenthaozzz/edge-ai-voice-control-esp32) | Triển khai lệnh tiếng Việt độc lập, Edge Impulse/ESP32-S3 N16R8, I2S INMP441, MFCC, TFLite Micro | Có demo điều khiển tiếng Việt | Audio 16 kHz, cửa sổ 1 s, hop 200 ms; không tự động suy ra accuracy 94% |
| AIS3 `research/kws_vi` (bổ sung) | Bộ huấn luyện thử nghiệm MFCC/CNN TFLite INT8 trên PC | Người dùng tự cung cấp corpus giọng Việt | Chưa có benchmark hoặc firmware tích hợp; số liệu chỉ sinh ra sau khi train/test |

**Phân tích 368 ms:** Theo bảng đo trong bài nghiên cứu, tổng gồm endpoint detection ~35 ms, DSP + MFCC ~276 ms, neural inference ~56 ms và điều khiển ~1 ms. Không gọi cả 368 ms là 'CNN inference'.

KWS là **phân loại âm thanh**, không phải chuyển giọng nói thành văn bản (ASR), và **không dùng tokenizer BPE của TinyStories**. AIS3 PLE hiện dùng **INT4 tùy chỉnh** cho LLM; một số dự án KWS dùng TFLite Micro **INT8**. Không chuyển model TinyStories sang TFLite theo quy trình KWS rồi kỳ vọng chạy được.

## 2. Chuẩn bị thiết bị và dữ liệu giọng nói

Để thử hệ thống đã có firmware thực tế, xem [Edge AI Voice Control](https://github.com/nguyenthaozzz/edge-ai-voice-control-esp32). Bo **ESP32-S3 N16R8**, micro INMP441 I2S, nguồn 3.3 V, relay/LED tùy chọn. Giữ đúng mapping GPIO của **project được chọn**, không ghép sơ đồ chân OLED của AIS3. Với relay kết nối điện lưới, chỉ thử bằng LED/nguồn thấp khi mới phát triển; lắp đặt điện lưới cần người đủ chuyên môn.

### Bộ dữ liệu huấn luyện mới cho module AIS3 (không có sẵn)

Bố trí tệp WAV: **PCM 16-bit, mono, 16 kHz, đúng người nói**:

```text
data/kws_vi/
  bat_den/
    nguoi_01__lan_01.wav
    nguoi_01__lan_02.wav
    nguoi_02__lan_01.wav
  tat_den/
    nguoi_01__lan_01.wav
  unknown/
    nguoi_03__lan_01.wav
  silence/
    nguoi_04__lan_01.wav
```

Ví dụ trên chỉ mô tả cấu trúc, **không đủ người nói** để đảm bảo mỗi lớp xuất hiện trong train/val/test. Chương trình sẽ từ chối thiếu lớp ở mỗi phần và dùng **SHA-256 ID người nói** để chia **70%/15%/15% theo speaker**. Không cắt nhiều tệp một người nói rồi phân bổ rải rác ba tập. Thu nhiều người nói, giới tính/độ tuổi/giọng vùng miền và môi trường tiếng ồn với sự đồng ý của người tham gia; có tập 'unknown' và 'silence'. Cân bằng lệnh dễ nhầm như 'bật đèn' và 'tắt đèn'.

### Cài công cụ

```bash
git clone https://github.com/xulytiengviet/AIS3.git
cd AIS3
uv sync
uv pip install "tensorflow>=2.16,<3"
```

Trên PC Windows có thể dùng venv/WSL2; TensorFlow nên chạy trên CPU/GPU của máy tính, **không huấn luyện trên MCU**.

### Huấn luyện và xuất INT8

```bash
uv run --no-sync python -m research.kws_vi.train \
  --data data/kws_vi --epochs 20 --batch 32 --seed 42 \
  --out artifacts/kws_vi
```

Đầu ra: `model-int8.tflite`, `labels.json`, `manifest.json`. Báo cáo lưu accuracy, macro-F1, confusion matrix trên **test tách speaker**, accuracy bản host float và **bản INT8 sau chuyển đổi**, cấu hình MFCC và SHA-256 model. **Đừng thay con số của bài báo bằng số của mô hình mới trước khi đo.**

Thử một tệp WAV trên PC:

```bash
uv run --no-sync python -m research.kws_vi.infer \
  --model-dir artifacts/kws_vi \
  --wav data/kws_vi/bat_den/nguoi_01__lan_01.wav --threshold 0.80
```

Đây là thử một **clip**, chưa đo false accepts/hour khi luôn mở mic.

### Đưa model sang ESP32-S3

Module huấn luyện mới tạo tệp `model-int8.tflite`, **chưa có firmware AIS3 tương thích**. Các bước còn cần thực hiện trước khi nạp KWS của AIS3:

1. Xây dựng/port **MFCC DSP đúng TensorFlow.signal**: PCM 16 kHz, STFT Hann 25 ms, hop 10 ms, fft 512, power spectrum, 40 Mel bins (80–7600 Hz), log floor `1e-6`, lấy 13 MFCC từ 98 khung. Phải có host-vs-device feature parity trên cùng WAV.
2. Thêm TFLite Micro phiên bản tương thích vào một sketch/ESP-IDF project **độc lập**, bảo đảm các op Conv2D, MaxPool, Mean, FullyConnected, Softmax và RAM arena thực tế.
3. Quantize đầu vào MFCC đúng scale/zero-point từ `manifest.json`. Kiểm tra parity logits/probability bản INT8 trên PC và thiết bị.
4. Thêm VAD, ring buffer, ngưỡng confidence, nhiều khung liên tiếp, cooldown, lớp unknown/silence. Báo cáo FP/hour (đặt mic luôn bật nhiều giờ), recall theo người nói, độ ồn, bộ nhớ, latency tổng và năng lượng.
5. Khi đạt KPI mới chuyển tiếp sự kiện wake word sang XiaoZhi hoặc tác vụ AI riêng. Đừng gọi các ví dụ CLI trên là 'firmware nạp được ngay'.

Nếu cần một demo firmware tiếng Việt để nạp **ngay**, tham khảo [source Edge AI Voice Control](https://github.com/nguyenthaozzz/edge-ai-voice-control-esp32) và làm theo sơ đồ của chính dự án:

```bash
git clone https://github.com/nguyenthaozzz/edge-ai-voice-control-esp32.git
cd edge-ai-voice-control-esp32/Firmware/Firmware-KWS-vietnamese-command
# Đọc README và platformio.ini của thư mục firmware thực tế trước khi nạp:
pio run
pio run --target upload
pio device monitor
```

Đây là source **độc lập**, không phải firmware KWS được đóng gói trong AIS3. Những tính năng Firebase/cloud ở dự án tham chiếu là tùy chọn/ngoài luồng nhận diện offline.

## 3. Chuẩn hóa tiếng Việt cho TinyStories/Barista: một pipeline khác

Với **văn bản**, AIS3 đã có `research/tinystories/vi_text.py`, `prepare_vi.py`, `audit_vi.py`. Để không làm sai nghĩa, mặc định chỉ chuẩn hóa Unicode **NFC**, newline, whitespace, ký tự điều khiển; **giữ nguyên dấu thanh, chữ hoa, URL, email, số, emoji và từ viết tắt**. Các bước dễ làm sai được để **opt-in**.

- NFC **không** biến `hóa` thành `hoá`: hai cách đặt dấu thanh này khác nhau ở chính tả nhưng cả hai đã là Unicode NFC hợp lệ.
- Chỉ ánh xạ vị trí dấu hoặc từ viết tắt nếu chọn chính sách corpus và có **từ điển JSON** rõ ràng. `ko→không`, `dc→được`, `j→gì`, `r→rồi` có thể sai trong văn bản chuyên môn/tên riêng.
- Không thay `2` thành `hai` nếu không cần đọc số bằng giọng nói. Không xóa emoji/URL/email một cách máy móc.
- LLM hiện tại dùng **ByteLevel-BPE segmentation-free**, bao phủ raw UTF-8; có thể thử vocab `10.000` với **huấn luyện model mới**, nhưng đây là giả thuyết cần benchmark, không là quy chuẩn chắc chắn. SentencePiece Unigram + byte-fallback muốn thay vào firmware cần viết lại encoder/decoder và thử golden vector; không thể thay thẳng `tokenizer.json`.
- Tỷ lệ `<unk>`, round-trip lossless, bytes/token và token/100 âm tiết là chỉ số kiểm tra tokenizer văn bản, **không liên quan trực tiếp đến MFCC**.

### Chuẩn bị corpus (mặc định bảo toàn)

```bash
uv run python -m research.tinystories.prepare_vi \
  --input data/vi_stories.txt --vocab 32768
```

Tùy chọn mở rộng teencode/HTML/lowercase **chỉ khi corpus cần**:

```bash
uv run python -m research.tinystories.prepare_vi \
  --input data/vi_stories.txt --vocab 32768 \
  --expand-teencode --strip-html --lowercase \
  --output-root data/vi_tinystories_custom
```

Ví dụ một từ điển chuyển dấu tùy chính sách (không áp dụng mặc định):

```json
{"hoá": "hóa", "huỷ": "hủy"}
```

Lưu thành `tone_rules.json` và truyền `--replacements tone_rules.json`; script ghi SHA-256 của từ điển vào metadata.

**Phải dùng cùng chính sách trong train, eval và khi xử lý đầu vào runtime**. Việc làm sạch trước huấn luyện có thể làm mất dữ liệu gốc; giữ file nguồn chưa biến đổi.

### Audit tokenizer

```bash
uv run python -m research.tinystories.audit_vi \
  --input data/vi_stories.txt \
  --tokenizer data/vi_tinystories/vocab-32768/tokenizer.json \
  --output artifacts/vi_tokenizer_audit.json
```

Dòng cuối phải là `AUDIT PASS`. Một sai khác decode/tokenize, token unknown hay văn bản chưa NFC đều cần xử lý trước khi train. Giá trị bytes/token cao hơn **không** luôn đồng nghĩa chất lượng tiếng Việt tốt hơn; phải kiểm tra loss/perplexity, coherence, độ dài ngữ cảnh và tốc độ thực tế.

## 4. Mốc kiểm toán và KPI nghiên cứu

| Giai đoạn | Bằng chứng bắt buộc | Chưa được coi là hoàn thành khi |
| --- | --- | --- |
| Feasibility | Flash, SRAM, PSRAM, pipeline model/feature, benchmark baseline | Chỉ dùng thông số datasheet |
| Prototype | Hash corpus/model, người nói disjoint, manifest feature, mẫu calibration INT8 | DSP trên ESP32 khác PC |
| Validation | Held-out speakers, noise/OOD, macro-F1, precision, recall, confusion matrix, false accepts/hour | Chỉ báo 94% accuracy trích bài báo |
| Device | Test vector audio/feature/logits parity, độ trễ tổng, heap/arena và điện năng trên board | Mới test file trên PC |
| Rollout | Nguồn gốc release, firmware nạp lại, rollback, theo dõi lỗi định kỳ | Chưa có kịch bản quay lại firmware cũ |
| Text model | Tokenizer hash, NFC, round-trip, PPL và đánh giá Grammar/Creativity/Consistency | Lấy metric KWS để đánh giá TinyStories |

KPI tham khảo như **accuracy ≥90%** hoặc latency **<100 ms** là **mục tiêu thử nghiệm**, không phải chuẩn bắt buộc cho mọi tác vụ. Một chu trình KWS thực tế 368 ms vẫn có thể chấp nhận nếu quy trình tương tác cho phép. Kiểm tra ứng dụng luôn bật bằng **false accepts/hour** và mức tiêu thụ điện năng, không chỉ accuracy.

## 5. Tham khảo & ghi công

- Bài báo: [PTIT 2025 — TinyML-Based Vietnamese Keyword Recognition](https://tapchi.ptit.edu.vn/jstic-ptit/index.php/jstic/article/view/1668); [VJOL bản tiếng Việt](https://www.vjol.info.vn/3126-3445/article/view/184352/).
- Runtime tham khảo: [voicute/onnx-wakeword](https://github.com/voicute/onnx-wakeword) — Causal TCN, ESP32 INT8, xử lý false triggers, **chưa có pretrained tiếng Việt trong README**.
- Đối chứng DIY lệnh Việt: [nguyenthaozzz/edge-ai-voice-control-esp32](https://github.com/nguyenthaozzz/edge-ai-voice-control-esp32) — mô hình/firmware riêng, không mặc định tương thích với AIS3.
- Hướng dẫn kỹ thuật bổ sung: [VI_TRAINING.md](./VI_TRAINING.md).
