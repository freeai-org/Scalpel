"use strict";

const rounds = [
  {
    round: 0,
    deleted: null,
    current: null,
    layers: 28,
    accuracy: 84.20282856401009,
    preAccuracy: null,
    delta: 0,
    speed: 0.8166249089373021,
    speedDelta: 0,
    risk: null,
  },
  { round: 1, deleted: 5, current: 5, layers: 27, accuracy: 83.99623070088827, preAccuracy: 83.41713635713314, delta: -0.2065978631218157, speed: 0.845371149939182, speedDelta: 3.520127868654943, risk: 0.008869029102259554 },
  { round: 2, deleted: 8, current: 7, layers: 26, accuracy: 83.85109148940304, preAccuracy: 78.54693872504429, delta: -0.3517370746070503, speed: 0.8528787623324925, speedDelta: 4.439474353331763, risk: 0.016412649123470867 },
  { round: 3, deleted: 9, current: 7, layers: 25, accuracy: 83.83804201337801, preAccuracy: 82.16467633307604, delta: -0.3647865506320791, speed: 0.8970447530764989, speedDelta: 9.84783139224219, risk: 0.029109172230848506 },
  { round: 4, deleted: 23, current: 20, layers: 24, accuracy: 83.764309294484, preAccuracy: 82.01247012733306, delta: -0.4385192695260942, speed: 0.9412503432947013, speedDelta: 15.261037594307258, risk: 0.02487559940765003 },
  { round: 5, deleted: 4, current: 4, layers: 23, accuracy: 83.86724260049846, preAccuracy: 63.73936197458991, delta: -0.335585963511631, speed: 0.9563902647393161, speedDelta: 17.115000322962803, risk: 0.19776978237598414 },
  { round: 6, deleted: 18, current: 14, layers: 22, accuracy: 83.85880141940429, preAccuracy: 61.94560127739321, delta: -0.3440271446057963, speed: 1.045080695974674, speedDelta: 27.975608450967783, risk: 0.2655977459721544 },
  { round: 7, deleted: 15, current: 11, layers: 21, accuracy: 83.56469011027056, preAccuracy: 54.12557029636865, delta: -0.638138453739534, speed: 1.0727034390956434, speedDelta: 31.358158115894817, risk: 0.3038998228620118 },
  { round: 8, deleted: 13, current: 9, layers: 20, accuracy: 83.81577594749123, preAccuracy: 51.82227595720591, delta: -0.387052616518857, speed: 1.1218063893443362, speedDelta: 37.37107172057434, risk: 0.24160964172664304 },
  { round: 9, deleted: 25, current: 17, layers: 19, accuracy: 83.84725860323975, preAccuracy: 59.28237596550756, delta: -0.3555699607703411, speed: 1.171817295440159, speedDelta: 43.49516927729758, risk: 0.26094946724742496 },
];

const fieldMetrics = [
  { name: "cat presence", baseline: 98.50517439632043, final: 98.46684553468762 },
  { name: "cat count", baseline: 97.1636642391721, final: 97.01034879264085 },
  { name: "location", baseline: 93.45622119815669, final: 92.99539170506912 },
  { name: "vertical pos.", baseline: 90.36866359447004, final: 89.44700460829493 },
  { name: "action", baseline: 76.31336405529954, final: 75.57603686635944 },
  { name: "overall body", baseline: 75.80645161290323, final: 76.72811059907834 },
  { name: "ears", baseline: 87.41935483870968, final: 86.72811059907835 },
  { name: "tail", baseline: 69.81566820276498, final: 67.69585253456222 },
  { name: "face", baseline: 59.1705069124424, final: 59.1705069124424 },
  { name: "fur state", baseline: 94.00921658986175, final: 94.65437788018434 },
];

const removedLayers = rounds.slice(1).map((item) => item.deleted);
const number = new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function signed(value, digits = 2, suffix = "") {
  const formatted = Math.abs(value).toFixed(digits);
  if (Math.abs(value) < 10 ** -digits) return `0.${"0".repeat(digits)}${suffix}`;
  return `${value > 0 ? "+" : "−"}${formatted}${suffix}`;
}

function initHeroStack() {
  const stack = document.querySelector("#hero-layer-stack");
  if (!stack) return;
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < 19; index += 1) {
    const layer = document.createElement("i");
    layer.className = "hero-layer";
    layer.style.setProperty("--i", index);
    fragment.append(layer);
  }
  for (let index = 0; index < 9; index += 1) {
    const layer = document.createElement("i");
    layer.className = "hero-layer removed";
    layer.style.setProperty("--i", index);
    layer.style.top = `${92 + index * 12}px`;
    fragment.append(layer);
  }
  stack.append(fragment);
}

function pathFor(data, xScale, yScale) {
  return data.map((value, index) => `${index ? "L" : "M"}${xScale(index).toFixed(2)},${yScale(value).toFixed(2)}`).join(" ");
}

function renderTradeoffChart() {
  const root = document.querySelector("#tradeoff-chart");
  if (!root) return;
  const width = 840;
  const height = 390;
  const margin = { top: 28, right: 54, bottom: 42, left: 54 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (index) => margin.left + (index / (rounds.length - 1)) * plotWidth;
  const yAccuracy = (value) => margin.top + ((84.4 - value) / (84.4 - 83.0)) * plotHeight;
  const ySpeed = (value) => margin.top + ((50 - value) / 50) * plotHeight;
  const accuracy = rounds.map((item) => item.accuracy);
  const speed = rounds.map((item) => item.speedDelta);
  const accuracyPath = pathFor(accuracy, x, yAccuracy);
  const speedPath = pathFor(speed, x, ySpeed);
  const areaPath = `${accuracyPath} L${x(rounds.length - 1)},${height - margin.bottom} L${x(0)},${height - margin.bottom} Z`;
  const grid = Array.from({ length: 6 }, (_, index) => {
    const gridY = margin.top + (index / 5) * plotHeight;
    const accuracyTick = 84.4 - (index / 5) * 1.4;
    const speedTick = 50 - index * 10;
    return `
      <line class="chart-grid-line" x1="${margin.left}" y1="${gridY}" x2="${width - margin.right}" y2="${gridY}" />
      <text class="chart-axis-label" x="${margin.left - 12}" y="${gridY + 3}" text-anchor="end">${accuracyTick.toFixed(1)}</text>
      <text class="chart-axis-label" x="${width - margin.right + 12}" y="${gridY + 3}">${speedTick}</text>`;
  }).join("");
  const xTicks = rounds.map((item, index) => `
    <text class="chart-tick" x="${x(index)}" y="${height - 13}" text-anchor="middle">${String(item.round).padStart(2, "0")}</text>
  `).join("");
  const points = rounds.map((item, index) => `
    <circle class="chart-point accuracy" data-series="accuracy" data-index="${index}" cx="${x(index)}" cy="${yAccuracy(item.accuracy)}" r="5" />
    <circle class="chart-point speed" data-series="speed" data-index="${index}" cx="${x(index)}" cy="${ySpeed(item.speedDelta)}" r="5" />
  `).join("");

  root.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="accuracyGradient" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="#69d1bb" stop-opacity=".18" />
          <stop offset="100%" stop-color="#69d1bb" stop-opacity="0" />
        </linearGradient>
      </defs>
      ${grid}
      ${xTicks}
      <path class="chart-area" d="${areaPath}" />
      <path class="chart-accuracy-line" d="${accuracyPath}" />
      <path class="chart-speed-line" d="${speedPath}" />
      ${points}
      <g class="chart-tooltip" opacity="0">
        <rect width="128" height="48"></rect>
        <text x="10" y="18" class="tooltip-title"></text>
        <text x="10" y="35" class="tooltip-value"></text>
      </g>
    </svg>`;

  const tooltip = root.querySelector(".chart-tooltip");
  const title = tooltip.querySelector(".tooltip-title");
  const value = tooltip.querySelector(".tooltip-value");
  root.querySelectorAll(".chart-point").forEach((point) => {
    point.addEventListener("mouseenter", () => {
      const index = Number(point.dataset.index);
      const series = point.dataset.series;
      const row = rounds[index];
      const pointX = Number(point.getAttribute("cx"));
      const pointY = Number(point.getAttribute("cy"));
      const translateX = Math.min(width - 138, Math.max(5, pointX - 64));
      const translateY = Math.max(4, pointY - 60);
      title.textContent = `ROUND ${String(row.round).padStart(2, "0")}`;
      value.textContent = series === "accuracy" ? `accuracy  ${row.accuracy.toFixed(2)}%` : `speed  ${signed(row.speedDelta, 1, "%")}`;
      tooltip.setAttribute("transform", `translate(${translateX} ${translateY})`);
      tooltip.setAttribute("opacity", "1");
    });
    point.addEventListener("mouseleave", () => tooltip.setAttribute("opacity", "0"));
  });
}

function renderFieldChart() {
  const root = document.querySelector("#field-chart");
  if (!root) return;
  root.innerHTML = fieldMetrics.map((field, index) => {
    const delta = field.final - field.baseline;
    return `
      <div class="field-row" style="--delay:${index * 45}ms">
        <span class="field-name">${field.name}</span>
        <span class="field-bars" style="--baseline:${field.baseline}%; --final:${field.final}%">
          <i></i><i></i>
        </span>
        <span class="field-value">${signed(delta, 2)}</span>
      </div>`;
  }).join("");
}

function roundDetailMarkup(item) {
  if (item.round === 0) {
    return `
      <span class="round-tag">BASELINE / REFERENCE</span>
      <h3>28 layers</h3>
      <p>原始冻结 reference · 尚未删除任何层</p>
      <dl>
        <div><dt>Macro accuracy</dt><dd class="detail-accent">${item.accuracy.toFixed(2)}%</dd></div>
        <div><dt>Inference speed</dt><dd>${item.speed.toFixed(3)} it/s</dd></div>
        <div><dt>Deleted layer</dt><dd>—</dd></div>
        <div><dt>Probe risk</dt><dd>—</dd></div>
      </dl>`;
  }
  const recovered = item.accuracy - item.preAccuracy;
  return `
    <span class="round-tag">ROUND ${String(item.round).padStart(2, "0")} / POST-RECOVERY</span>
    <h3>Layer ${item.deleted}</h3>
    <p>删除当前下标 ${item.current} · 原始层 ${item.deleted}</p>
    <dl>
      <div><dt>Layers remain</dt><dd>${item.layers}</dd></div>
      <div><dt>Macro accuracy</dt><dd class="detail-accent">${item.accuracy.toFixed(2)}%</dd></div>
      <div><dt>Recovery gain</dt><dd>${signed(recovered, 2, " pp")}</dd></div>
      <div><dt>Speed vs. baseline</dt><dd>${signed(item.speedDelta, 1, "%")}</dd></div>
      <div><dt>Probe risk</dt><dd>${item.risk.toFixed(4)}</dd></div>
    </dl>`;
}

function renderRoundExplorer(selectedRound = 9) {
  const controls = document.querySelector("#round-controls");
  const map = document.querySelector("#layer-map");
  const detail = document.querySelector("#round-detail");
  const title = document.querySelector("#selected-round-title");
  if (!controls || !map || !detail || !title) return;

  if (!controls.children.length) {
    controls.innerHTML = rounds.map((item) => `
      <button class="round-button" data-round="${item.round}" aria-label="显示 Round ${item.round}">
        ${item.round === 0 ? "BASE" : `R${String(item.round).padStart(2, "0")}`}
      </button>`).join("");
    controls.addEventListener("click", (event) => {
      const button = event.target.closest(".round-button");
      if (button) renderRoundExplorer(Number(button.dataset.round));
    });
  }

  const item = rounds[selectedRound];
  const deletedAt = new Map(rounds.slice(1, selectedRound + 1).map((round) => [round.deleted, round.round]));
  controls.querySelectorAll(".round-button").forEach((button) => {
    const isActive = Number(button.dataset.round) === selectedRound;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });

  map.innerHTML = Array.from({ length: 28 }, (_, layer) => {
    const deletionRound = deletedAt.get(layer);
    const stateClass = deletionRound === selectedRound && selectedRound > 0 ? "just-deleted" : deletionRound ? "deleted" : "";
    const label = deletionRound ? `原始层 ${layer}，在 Round ${deletionRound} 删除` : `原始层 ${layer}，保留`;
    return `<div class="layer-cell ${stateClass}" title="${label}" aria-label="${label}">${String(layer).padStart(2, "0")}</div>`;
  }).join("");
  title.textContent = selectedRound === 0 ? "Baseline · 28 layers" : `Round ${String(selectedRound).padStart(2, "0")} · ${item.layers} layers remain`;
  detail.innerHTML = roundDetailMarkup(item);
}

function renderRoundTable() {
  const body = document.querySelector("#round-table-body");
  if (!body) return;
  body.innerHTML = rounds.slice(1).map((item) => `
    <tr>
      <td>Round ${String(item.round).padStart(2, "0")}</td>
      <td>L${String(item.deleted).padStart(2, "0")}</td>
      <td>${item.layers}</td>
      <td>${item.accuracy.toFixed(2)}%</td>
      <td class="negative">${signed(item.delta, 2, " pp")}</td>
      <td>${item.speed.toFixed(3)} it/s</td>
      <td class="positive">${signed(item.speedDelta, 1, "%")}</td>
      <td class="optional-col">${item.risk.toFixed(4)}</td>
    </tr>`).join("");
}

function initTableToggle() {
  const button = document.querySelector("#table-toggle");
  const card = button?.closest(".table-card");
  if (!button || !card) return;
  button.addEventListener("click", () => {
    const expanded = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", String(!expanded));
    button.firstChild.textContent = expanded ? "展开全部指标 " : "收起完整指标 ";
    card.classList.toggle("expanded", !expanded);
  });
}

function initReveal() {
  document.querySelectorAll("[data-delay]").forEach((element) => {
    element.style.setProperty("--delay", `${element.dataset.delay}ms`);
  });
  if (!("IntersectionObserver" in window)) {
    document.querySelectorAll(".reveal").forEach((element) => element.classList.add("is-visible"));
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));
}

function initHeader() {
  const header = document.querySelector(".site-header");
  const progress = document.querySelector("#page-progress-bar");
  const sections = [...document.querySelectorAll("main section[id]")];
  const links = [...document.querySelectorAll(".desktop-nav a")];
  const onScroll = () => {
    header?.classList.toggle("is-fixed", window.scrollY > 120);
    const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${maxScroll > 0 ? (window.scrollY / maxScroll) * 100 : 0}%`;
    let activeId = "";
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= 150) activeId = section.id;
    }
    links.forEach((link) => link.classList.toggle("active", link.hash === `#${activeId}`));
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

function initMobileMenu() {
  const button = document.querySelector("#menu-button");
  const nav = document.querySelector("#mobile-nav");
  if (!button || !nav) return;
  const close = () => {
    button.setAttribute("aria-expanded", "false");
    nav.classList.remove("open");
    document.body.classList.remove("menu-open");
  };
  button.addEventListener("click", () => {
    const open = button.getAttribute("aria-expanded") === "true";
    button.setAttribute("aria-expanded", String(!open));
    nav.classList.toggle("open", !open);
    document.body.classList.toggle("menu-open", !open);
  });
  nav.querySelectorAll("a").forEach((link) => link.addEventListener("click", close));
  window.addEventListener("resize", () => { if (window.innerWidth > 820) close(); });
}

function initFigureDialog() {
  const dialog = document.querySelector("#figure-dialog");
  const image = document.querySelector("#dialog-image");
  const close = document.querySelector("#dialog-close");
  if (!dialog || !image || !close) return;
  document.querySelectorAll("[data-dialog-image]").forEach((button) => {
    button.addEventListener("click", () => {
      image.src = button.dataset.dialogImage;
      dialog.showModal();
    });
  });
  close.addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    const bounds = dialog.getBoundingClientRect();
    const inside = event.clientX >= bounds.left && event.clientX <= bounds.right && event.clientY >= bounds.top && event.clientY <= bounds.bottom;
    if (!inside) dialog.close();
  });
}

function initCopyButtons() {
  document.querySelectorAll(".copy-button").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent);
        button.textContent = "COPIED ✓";
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(target);
        selection.removeAllRanges();
        selection.addRange(range);
        button.textContent = "SELECTED";
      }
      window.setTimeout(() => { button.textContent = "COPY"; }, 1800);
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initHeroStack();
  renderTradeoffChart();
  renderFieldChart();
  renderRoundExplorer(9);
  renderRoundTable();
  initTableToggle();
  initReveal();
  initHeader();
  initMobileMenu();
  initFigureDialog();
  initCopyButtons();
});
