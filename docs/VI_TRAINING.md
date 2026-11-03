# AIS3 – Huấn luyện TinyStories tiếng Việt & DIY ESP32-S3

Đây là hướng dẫn đi kèm [trang AIS3 Lab](./index.html). Ngày cập nhật: 03/11/2026.

> **Ranh giới phần cứng:** AIS3 TinyStories dùng ESP32-S3 **N16R8** (16 MB Flash/8 MB PSRAM); [Super Mini AI](https://github.com/Ngducok/esp32s3-Super-mini-AI) dùng mô hình **khác** trên 4 MB Flash/0 MB PSRAM; [XiaoZhi Việt hóa](https://github.com/Ngthhoang/xiaozhi-vn-ricky0611) là trợ lý giọng nói, thường cần backend AI qua Internet. Không trộn firmware, phân vùng Flash hay sơ đồ GPIO.

## 1. Cài AIS3 chuẩn

Yêu cầu macOS/Linux hoặc Windows có WSL2 và USB passthrough, Git, Python >= 3.12, uv, Hugging Face CLI (`hf`), `shasum`, `arduino-cli`, esptool, C compiler, ESP32 Arduino core **3.3.10**.

```bash
git clone https://github.com/xulytiengviet/AIS3.git
cd AIS3
arduino-cli core update-index --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32@3.3.10 --additional-urls https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli lib install 'Adafruit SH110X@2.1.14' 'Adafruit GFX Library@1.12.6' 'Adafruit BusIO@1.17.4'
scripts/fetch_model.sh tinystories
PORT=/dev/ttyACM0 scripts/deploy.sh tinystories
arduino-cli monitor -p /dev/ttyACM0 --config baudrate=115200
```

Thay `/dev/ttyACM0` bằng cổng thực tế. Trên macOS, deploy mặc định dò `/dev/cu.usbmodem*`. Dùng `barista` hoặc `fly` thay `tinystories` để nạp model khác. Một lần chỉ giữ một mô hình trên board. `fetch_model.sh` xác minh SHA-256 và không ghi thiết bị; `deploy.sh` chạy host gates, compile rồi ghi model và firmware.

## 2. Dữ liệu truyện Việt

Tạo `data/vi_stories.txt` theo UTF-8; mỗi truyện cách nhau bằng một dòng trống. Ví dụ:

```text
Ngày xửa ngày xưa, có một bạn thỏ tên là Mây. Bạn thích giúp đỡ các con vật trong rừng.

Một buổi sáng, cậu bé An tìm thấy một hạt giống. Cậu gieo hạt, tưới nước và chờ cây lớn.
```

**Đây chỉ là ví dụ định dạng**, không đủ số token để chạy huấn luyện. Dùng corpus lớn, truyện nhất quán, nội dung có quyền sử dụng và không chứa dữ liệu cá nhân. `prepare_vi.py` chuẩn hóa Unicode NFC, bỏ trùng, chia **theo truyện** thành train/validation để hạn chế rò rỉ.

## 3. Phương án A – huấn luyện mới với tokenizer tiếng Việt

```bash
uv sync
uv run python -m research.tinystories.prepare_vi --input data/vi_stories.txt --vocab 32768
uv run python -m research.tinystories.train_vi \
  --data-dir data/vi_tinystories/vocab-32768 --steps 3000 --tag vi-demo
```

Đầu ra: `data/vi_tinystories/vocab-32768/{tokenizer.json,train.bin,val.bin,metadata.json}` và `runs/ple-vi-demo-s0.pt`. Vocab capacity tối đa 65536 vì token bin lưu `uint16`. Nên tăng corpus và huấn luyện theo đánh giá thực tế; 3.000 bước chỉ là khởi đầu để thử pipeline, **không cam kết văn bản mạch lạc**. Huấn luyện nặng chạy trên máy tính/GPU, ESP32 chỉ suy luận.

## 4. Phương án B – fine-tune checkpoint PyTorch

Trường hợp chưa có checkpoint PyTorch gốc, hãy tạo một checkpoint từ mã AIS3 hiện tại và bộ tokenizer tiếng Anh:

```bash
uv run python -m research.tinystories.prepare --vocab 32768
uv run python -m research.tinystories.train --arm ple --vocab 32768 \
  --d-model 96 --n-layers 6 --n-heads 4 --ple-dim 128 \
  --target-core 559000 --steps 3000 --tag en-base --seed 0
```

Chương trình sẽ sinh `runs/ple-en-base-s0.pt`, chứa tokenizer SHA-256. Dữ liệu tiếng Anh gốc phải được tải lần đầu; đây là một bước huấn luyện tốn tài nguyên chứ không phải file checkpoint đi kèm repo.
Fine-tune chỉ hợp lệ nếu có checkpoint đáng tin cậy `runs/*.pt` với **cùng tokenizer SHA-256 và kiến trúc**. File release `artifacts/tinystories/model.bin` đã lượng tử hóa int4 **không** là checkpoint để fine-tune. Lệnh mẫu dưới đây cần checkpoint đã tạo ở bước trước hoặc từ nguồn tin cậy; **không có sẵn trong clone mới**.

```bash
uv run python -m research.tinystories.prepare_vi \
  --input data/vi_stories.txt --vocab 32768 \
  --tokenizer data/tinystories/vocab-32768/tokenizer.json
uv run python -m research.tinystories.train_vi \
  --data-dir data/vi_tinystories/vocab-32768 \
  --init runs/ple-en-base-s0.pt --steps 1500 --lr 0.0001 --tag vi-finetune
```

Không tắt kiểm tra tokenizer SHA. Muốn tokenizer Việt mới thì retrain từ đầu hoặc thiết kế riêng phương án ánh xạ embedding. Không mở `.pt` từ nguồn không tin cậy: PyTorch pickle có nguy cơ thực thi mã.

## 5. Kiểm thử trên PC trước khi nạp

```bash
uv run python -m research.tinystories.sample \
  --run runs/ple-vi-demo-s0.pt \
  --tokenizer data/vi_tinystories/vocab-32768/tokenizer.json \
  --prompt "Ngày xửa ngày xưa" --tokens 120
```

Kiểm tra loss/perplexity trên tập validation và đọc thủ công: lỗi dấu, ngữ pháp, lặp, tính nhất quán nhân vật, kết thúc truyện. Kết quả benchmark tiếng Anh trong `RESULTS.md` không tự động áp cho tiếng Việt.

## 6. Xuất int4 và nạp N16R8

```bash
uv run python -m research.tinystories.export ple-vi-demo-s0 \
  --tokenizer data/vi_tinystories/vocab-32768/tokenizer.json \
  --out-dir artifacts/vi_tinystories
ARTIFACTS=artifacts/vi_tinystories \
TINYSTORIES_PROMPT="Ngày xửa ngày xưa" \
PORT=/dev/ttyACM0 scripts/deploy.sh tinystories
```

Kiểm tra `artifacts/vi_tinystories/model.bin` vừa phân vùng model (`0xEE0000` = **15.597.568 byte**) của `firmware/esp32_tinystories/partitions.csv`. Khi triển khai, `TINYSTORIES_PROMPT` dùng tokenizer tương ứng để sinh ID prompt; generator xuất **raw ByteLevel bytes** giúp giữ nguyên UTF-8 khi token chia giữa ký tự có dấu.

Đọc đầu ra trên **USB Serial UTF-8**. OLED 128×64 ASCII mặc định chưa có renderer/font tiếng Việt đầy đủ. Muốn hiển thị đầy đủ dấu phải bổ sung bộ font và pipeline render riêng.

## 7. Super Mini AI 4 MB, không PSRAM

Tham khảo [README_VN](https://github.com/Ngducok/esp32s3-Super-mini-AI/blob/main/README_VN.md). Mô hình **118.784 tham số**, 3 lớp, d=64, 4 heads, KV cache 24,5 KB, peak SRAM 161 KB theo số liệu tác giả; **không tương thích file model AIS3 14,9 MB**. Dùng ESP-IDF:

```bash
git clone https://github.com/Ngducok/esp32s3-Super-mini-AI.git
cd esp32s3-Super-mini-AI/firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

Hoặc mở `esp32_ai_runtime/esp32_ai_runtime.ino` trong Arduino IDE; chọn ESP32S3 Dev Module, Flash 4 MB, PSRAM Disabled, CPU 240MHz, partition Huge APP theo README. Sau nạp: Wi-Fi thiết bị `ESP32-Local-AI`, mật khẩu demo `12345678`, vào `http://192.168.4.1`. Đổi mật khẩu demo khi sử dụng trong môi trường có nhiều người.

## 8. XiaoZhi Việt hóa – DIY micro/loa

Tham khảo [Ngthhoang/xiaozhi-vn-ricky0611](https://github.com/Ngthhoang/xiaozhi-vn-ricky0611) (nhánh `develop_vn`) và [custom-board.md](https://github.com/Ngthhoang/xiaozhi-vn-ricky0611/blob/develop_vn/docs/custom-board.md). Chuẩn bị bo tương thích firmware, mic I²S, amplifier/loa, nguồn, các nút và màn hình tùy chọn. **Không đấu nguồn/loa trực tiếp vào GPIO**; không sao chép chân AIS3 sang bo khác.

```bash
git clone --branch develop_vn https://github.com/Ngthhoang/xiaozhi-vn-ricky0611.git
cd xiaozhi-vn-ricky0611
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

ESP-IDF v5.4+ theo README; chọn đúng bo trong menuconfig trước khi build. Có thể dùng [Web Flasher do dự án giới thiệu](https://tienhuyiot.github.io/esp_web_flasher/) **chỉ khi firmware đúng bo**. Sau nạp, cấu hình Wi-Fi 2,4 GHz, ghép mã kích hoạt với dịch vụ tương ứng rồi kiểm thử theo luồng micro → ASR → LLM → TTS → loa. XiaoZhi thường cần kết nối Internet. Chưa có bản firmware AIS3 kết hợp XiaoZhi được xác nhận.

## 9. Tài liệu gốc và ghi công

- [AIS3](https://github.com/xulytiengviet/AIS3) và [RESULTS.md](https://github.com/xulytiengviet/AIS3/blob/main/RESULTS.md): PLE runtime, firmware, kết quả TinyStories, Barista và Fly.
- [Super Mini AI](https://github.com/Ngducok/esp32s3-Super-mini-AI): triển khai Web AI không PSRAM và benchmark, tác giả Dustin Nguyen.
- [XiaoZhi Việt hóa](https://github.com/Ngthhoang/xiaozhi-vn-ricky0611): ESP-IDF, âm thanh và DIY; dự án gốc [78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32).
- Nêu rõ giấy phép từng repo và quyền sử dụng dữ liệu. AIS3 có LICENSE MIT ghi công Viacheslav Sierbov; việc dẫn nguồn không đồng nghĩa mã từ dự án khác đã được sáp nhập.

**Trạng thái xác minh:** các con số là benchmark nguồn công bố. Các script dữ liệu/training mới đã được thêm vào repository, nhưng cần chạy kiểm thử thực tế và benchmark trên phần cứng riêng trước khi tuyên bố đạt chỉ số hay chất lượng tiếng Việt.
