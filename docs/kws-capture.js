/* AIS3 Sen Viet: consent-based, local-only 16 kHz PCM WAV recorder.
   Captures training DATA, does not run a speech recognition model. */
(() => {
  "use strict";
  const button = document.getElementById("kws-record-button");
  if (!button) return;
  const download = document.getElementById("kws-download-button");
  const speaker = document.getElementById("kws-speaker-id");
  const label = document.getElementById("kws-clip-label");
  const status = document.getElementById("kws-record-status");
  const player = document.getElementById("kws-local-player");
  let recorder = null, stream = null, chunks = [], latestWav = null, playUrl = null;
  const capabilities = typeof MediaRecorder !== "undefined" &&
    !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia) &&
    !!(window.AudioContext || window.webkitAudioContext) &&
    !!(window.OfflineAudioContext || window.webkitOfflineAudioContext);
  if (!capabilities) {
    button.disabled = true;
    status.textContent = "Trình duyệt này chưa hỗ trợ ghi âm và chuyển WAV trực tiếp.";
    return;
  }
  const writeAscii = (view, index, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(index + i, text.charCodeAt(i));
  };
  function wavFromSamples(samples) {
    const count = 16000;
    const buffer = new ArrayBuffer(44 + count * 2);
    const view = new DataView(buffer);
    writeAscii(view, 0, "RIFF");
    view.setUint32(4, 36 + count * 2, true);
    writeAscii(view, 8, "WAVE");
    writeAscii(view, 12, "fmt ");
    view.setUint32(16, 16, true); // PCM
    view.setUint16(20, 1, true); // encoding
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, 16000, true);
    view.setUint32(28, 32000, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeAscii(view, 36, "data");
    view.setUint32(40, count * 2, true);
    for (let i = 0; i < count; i++) {
      const f = Math.max(-1, Math.min(1, samples[i] || 0));
      view.setInt16(44 + i * 2, f < 0 ? Math.round(f * 32768) : Math.round(f * 32767), true);
    }
    return new Blob([buffer], {type: "audio/wav"});
  }
  async function toWav(blob) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const context = new Ctx();
    try {
      const decoded = await context.decodeAudioData(await blob.arrayBuffer());
      const length = Math.max(16000, Math.ceil(decoded.duration * 16000));
      const offline = new OfflineCtx(1, length, 16000);
      const source = offline.createBufferSource();
      source.buffer = decoded;
      source.connect(offline.destination);
      source.start(0);
      const rendered = await offline.startRendering();
      const pcm = rendered.getChannelData(0);
      const offset = Math.max(0, Math.floor((pcm.length - 16000) / 2));
      return wavFromSamples(pcm.subarray(offset, offset + 16000));
    } finally {
      await context.close().catch(() => {});
    }
  }
  function cleanup() {
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    recorder = null;
    button.disabled = false;
    button.textContent = "🎙️ Ghi mẫu 1 giây";
  }
  button.addEventListener("click", async () => {
    if (recorder && recorder.state === "recording") {
      recorder.stop();
      return;
    }
    const person = speaker.value.trim();
    if (!/^[a-zA-Z0-9_-]{1,40}$/.test(person)) {
      status.textContent = "Nhập ID người nói bằng chữ/số ASCII, gạch dưới hoặc gạch ngang (1–40 ký tự).";
      speaker.focus();
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}
      });
      chunks = [];
      latestWav = null;
      download.disabled = true;
      if (playUrl) { URL.revokeObjectURL(playUrl); playUrl = null; }
      player.hidden = true;
      const current = new MediaRecorder(stream);
      recorder = current;
      current.addEventListener("dataavailable", e => {
        if (e.data && e.data.size) chunks.push(e.data);
      });
      current.addEventListener("stop", async () => {
        try {
          if (!chunks.length) throw new Error("Không thu được âm thanh");
          const audioBlob = new Blob(chunks, {type: current.mimeType || chunks[0].type});
          latestWav = await toWav(audioBlob);
          playUrl = URL.createObjectURL(latestWav);
          player.src = playUrl;
          player.hidden = false;
          download.disabled = false;
          status.textContent = "Đã tạo WAV 16 kHz, PCM16 mono, 1 giây (" +
            latestWav.size.toLocaleString("vi-VN") + " byte). Chỉ lưu trong phiên trình duyệt.";
        } catch (err) {
          status.textContent = "Không chuyển được bản ghi thành WAV: " + (err.message || err);
        } finally {
          cleanup();
        }
      }, {once: true});
      current.start();
      button.textContent = "⏹ Dừng ghi";
      status.textContent = "Micro đã bật. Hãy nói một từ khóa; tự dừng sau khoảng 1,2 giây.";
      window.setTimeout(() => {
        if (current.state === "recording") current.stop();
      }, 1200);
    } catch (err) {
      cleanup();
      status.textContent = "Không mở được micro: " + (err.message || "Kiểm tra quyền trình duyệt.");
    }
  });
  download.addEventListener("click", () => {
    if (!latestWav) return;
    const person = speaker.value.trim();
    if (!/^[a-zA-Z0-9_-]{1,40}$/.test(person)) {
      status.textContent = "ID người nói không hợp lệ.";
      return;
    }
    const selected = label.value;
    if (!/^[a-z_]{2,30}$/.test(selected)) {
      status.textContent = "Nhãn không hợp lệ."; return;
    }
    const stamp = new Date().toISOString().replace(/[^0-9TZ]/g, "").slice(0, 15);
    const filename = person + "__" + stamp + ".wav";
    const href = URL.createObjectURL(latestWav);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(href), 1000);
    status.textContent = "Đã tải " + filename + ". Hãy chuyển tệp vào data/kws_vi/" +
      selected + "/ theo đúng nhãn.";
  });
})();
