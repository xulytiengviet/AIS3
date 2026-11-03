/* AIS3 Sen Viet — local-only normalization demo, NOT TinyStories tokenizer/ASR/KWS. */
(() => {
  "use strict";
  const $ = selector => document.querySelector(selector);
  const input = $("#vi-lab-input");
  const output = $("#vi-lab-output");
  if (!input || !output) return;
  const status = $("#vi-lab-status");
  const original = input.value;
  const examples = {
    prose: "Ngày xửa ngày xưa, cô bé Mây sống ở một ngôi làng nhỏ. Bé yêu tiếng Việt 🇻🇳.",
    unicode: "Hoa\u0300 sen nở. Tiếng Việt: hóa / hoá, hủy / huỷ, ắ, ệ, ỹ và 👩‍💻.",
    teencode: "ko dc j r =))\nNgày mai gửi email abc@example.com và mã ESP32-S3 nhé!",
    html: "<p>Chào bạn!</p>  Em   học   tiếng   Việt 🇻🇳."
  };
  const encoder = new TextEncoder();
  const escapeRegExp = word => Array.from(word).map(ch =>
    ("\\^$.*+?()[]{}|".includes(ch) ? "\\" + ch : ch)).join("");
  function normalize(text) {
    let s = text.replace(/\r\n?/g, "\n").normalize("NFC");
    if ($("#vi-lab-html").checked) s = s.replace(/<\/?[A-Za-z][^>]*>/g, " ");
    s = s.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, "")
         .replace(/[\ufeff\u200b]/g, "");
    if ($("#vi-lab-lower").checked) s = s.toLowerCase();
    if ($("#vi-lab-teencode").checked) {
      const rules = [["khum", "không"], ["hok", "không"], ["ko", "không"],
        ["dc", "được"], ["đc", "được"], ["j", "gì"], ["r", "rồi"],
        ["=))", "(cười)"], [":))", "(cười)"]];
      for (const [key, value] of rules) {
        const re = new RegExp("(?<![\\p{L}\\p{N}_])" + escapeRegExp(key) +
          "(?![\\p{L}\\p{N}_])", "giu");
        s = s.replace(re, value);
      }
    }
    return s.split("\n").map(x => x.replace(/[ \t]+/g, " ").trim())
            .join("\n").trim().normalize("NFC");
  }
  function update() {
    const source = input.value;
    const result = normalize(source);
    output.value = result;
    $("#vi-metric-before").textContent = String(encoder.encode(source).length);
    $("#vi-metric-after").textContent = String(encoder.encode(result).length);
    $("#vi-metric-codepoints").textContent = String(Array.from(result).length);
    const notes = [];
    if (source !== source.normalize("NFC")) notes.push("Có chuỗi chưa chuẩn Unicode NFC.");
    if (source.includes("hóa") || source.includes("hoá"))
      notes.push("NFC không tự chọn cách viết hóa/hoá.");
    notes.push(source !== result ?
      "Đã có biến đổi; phải dùng cùng quy tắc khi huấn luyện và suy luận." :
      "Nội dung không đổi theo tùy chọn hiện tại.");
    $("#vi-lab-explain").textContent = notes.join(" ");
    status.textContent = "Chỉ chạy trong trình duyệt · Không gửi dữ liệu";
  }
  input.addEventListener("input", update);
  ["vi-lab-lower", "vi-lab-teencode", "vi-lab-html"].forEach(id => {
    document.getElementById(id).addEventListener("change", update);
  });
  document.querySelectorAll("[data-vi-example]").forEach(button => {
    button.addEventListener("click", () => {
      const value = examples[button.dataset.viExample];
      if (value !== undefined) { input.value = value; update(); }
    });
  });
  $("#vi-lab-reset").addEventListener("click", () => {
    input.value = original;
    ["vi-lab-lower", "vi-lab-teencode", "vi-lab-html"].forEach(id => {
      document.getElementById(id).checked = false;
    });
    update();
  });
  $("#vi-lab-copy").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(output.value);
      status.textContent = "Đã sao chép nội dung";
    } catch {
      output.focus(); output.select();
      status.textContent = "Không thể sao chép tự động. Hãy sao chép từ ô kết quả.";
    }
  });
  update();
})();
