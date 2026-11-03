/* AIS3 Sen Viet — Roadmap viewer. All time ranges and KPIs are planning estimates.
   State exists only in this browser page; it is NOT evidence of actual acceptance. */
(() => {
  "use strict";
  const root = document.getElementById("roadmap");
  if (!root) return;
  const filters = [...root.querySelectorAll("[data-route-filter]")];
  const phases = [...root.querySelectorAll(".roadmap-phase[data-tracks]")];
  const routeInfo = {
    all: {
      label: "Tất cả 9 giai đoạn, gồm các tuyến triển khai độc lập. Các khoảng thời gian không cộng dồn như cam kết.",
      time: "Kế hoạch nhiều tuyến — làm song song",
    },
    ais3: {
      label: "AIS3 gốc N16R8: tập trung phần cứng, nạp firmware TinyStories/Barista/Fly, theo dõi RAM/Flash và khả năng khôi phục.",
      time: "Khoảng 1–2 ngày làm việc (đủ linh kiện)",
    },
    supermini: {
      label: "Super Mini: firmware micro-model 4MB riêng, không dùng model AIS3 14,9MB; kiểm tra SoftAP và trang LAN.",
      time: "Khoảng 0,5–1,5 ngày làm việc",
    },
    vi: {
      label: "TinyStories tiếng Việt: corpus → tokenizer → train trên PC/GPU → kiểm tra chất lượng → INT4 → N16R8. Thời gian GPU biến động mạnh.",
      time: "Khoảng 2–5 tuần nếu chưa có corpus; PLE thử 8–72 giờ máy",
    },
    kws: {
      label: "KWS tiếng Việt: ghi âm đa người nói → MFCC/CNN INT8 trên PC → port DSP/TFLite Micro → kiểm thử bo thật.",
      time: "Khoảng 3–8 tuần; CNN host 0,5–6 giờ/lượt thử",
    },
    xiaozhi: {
      label: "XiaoZhi: nhánh firmware âm thanh riêng, kiểm tra mic, loa, Wi-Fi, kích hoạt và backend ASR/LLM/TTS.",
      time: "Khoảng 1–3 ngày (bo và backend tương thích)",
    },
  };
  const help = document.getElementById("roadmap-filter-note");
  const total = document.getElementById("roadmap-visible-count");
  const selectedTime = document.getElementById("roadmap-selected-time");
  function selectRoute(route) {
    if (!routeInfo[route]) return;
    filters.forEach(button => {
      const on = button.dataset.routeFilter === route;
      button.setAttribute("aria-pressed", String(on));
    });
    let count = 0;
    phases.forEach(phase => {
      const tags = (phase.dataset.tracks || "").split(/\s+/);
      const visible = route === "all" || tags.includes("common") || tags.includes(route);
      phase.hidden = !visible;
      if (visible) count++;
    });
    help.textContent = routeInfo[route].label;
    total.textContent = String(count);
    selectedTime.textContent = routeInfo[route].time;
  }
  filters.forEach(button => button.addEventListener("click", () => {
    selectRoute(button.dataset.routeFilter);
  }));
  const evidence = [...root.querySelectorAll("[data-roadmap-evidence]")];
  const bar = document.getElementById("roadmap-evidence-bar");
  const label = document.getElementById("roadmap-evidence-count");
  function refreshEvidence() {
    const checked = evidence.filter(item => item.checked).length;
    bar.style.width = evidence.length ? (100 * checked / evidence.length) + "%" : "0%";
    label.textContent = checked + "/" + evidence.length +
      " mục tự đánh dấu đã chuẩn bị chứng cứ. Chưa thay thế số đo, biên bản hoặc nghiệm thu.";
  }
  evidence.forEach(item => item.addEventListener("change", refreshEvidence));
  const print = document.getElementById("roadmap-print");
  if (print) print.addEventListener("click", () => window.print());
  selectRoute("all");
  refreshEvidence();
})();
