"use strict";

/* The project page. Every figure comes from /api/about, which reads the model artifacts. */

const PIPELINE = [
  { it: "Catasto\nincendi PAT", en: "PAT fire\ncadastre" },
  { it: "Griglia 500 m\ne campioni", en: "500 m grid\nand samples" },
  { it: "Feature\nambientali", en: "Environmental\nfeatures" },
  { it: "Modello\nXGBoost", en: "XGBoost\nmodel" },
  { it: "Calibrazione\ne classi", en: "Calibration\nand classes" },
  { it: "Mappa\ngiornaliera", en: "Daily\nmap" },
];

function pipelineSvg(lang) {
  const boxW = 96;
  const boxH = 52;
  const gap = 22;
  const width = PIPELINE.length * boxW + (PIPELINE.length - 1) * gap;
  const height = boxH + 26;

  // no width/height attributes: the viewBox alone lets the diagram scale with the column
  const parts = [`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="pipeline">`];

  PIPELINE.forEach((step, index) => {
    const x = index * (boxW + gap);
    const y = 4;
    const last = index === PIPELINE.length - 1;
    const fill = last ? "#720000" : "#f8f9fa";
    const stroke = last ? "#720000" : "#dee2e6";
    const ink = last ? "#ffffff" : "#343a40";

    parts.push(
      `<rect x="${x}" y="${y}" width="${boxW}" height="${boxH}" fill="${fill}" ` +
        `stroke="${stroke}" stroke-width="1"/>`,
      `<rect x="${x}" y="${y}" width="${boxW}" height="3" fill="#c6020f"/>`
    );

    step[lang].split("\n").forEach((line, row) => {
      parts.push(
        `<text x="${x + boxW / 2}" y="${y + 26 + row * 13}" text-anchor="middle" ` +
          `font-family="Helvetica Neue, Arial, sans-serif" font-size="11" fill="${ink}">` +
          `${line}</text>`
      );
    });

    if (!last) {
      const ax = x + boxW;
      const ay = y + boxH / 2;
      parts.push(
        `<line x1="${ax + 3}" y1="${ay}" x2="${ax + gap - 7}" y2="${ay}" ` +
          `stroke="#88949f" stroke-width="1.5"/>`,
        `<polygon points="${ax + gap - 7},${ay - 4} ${ax + gap},${ay} ${ax + gap - 7},${ay + 4}" ` +
          `fill="#88949f"/>`
      );
    }
  });

  parts.push("</svg>");
  return parts.join("");
}

function renderAbout(about, t, lang) {
  const record = about.record;
  const model = about.model;

  const numbers = [
    [record.end_year - record.start_year + 1, t("n_years")],
    [record.cadastre_ignitions, t("n_fires")],
    [record.cells, t("n_cells")],
    [record.features, t("n_features")],
  ];
  document.getElementById("about-numbers").innerHTML = numbers
    .map(
      ([value, label]) =>
        `<div><div class="value">${value == null ? t("no_data") : formatCount(value, lang)}</div>` +
        `<div class="label">${label}</div></div>`
    )
    .join("");

  document.getElementById("pipeline").innerHTML = pipelineSvg(lang);

  const term = (value) => (value ? (TERMS[lang] || {})[value] || value : "");
  const sources = document.querySelector("#about-sources tbody");
  sources.innerHTML = about.sources
    .map((row) => {
      const provider = row.url
        ? `<a href="${row.url}" rel="noreferrer">${row.provider}</a>`
        : row.provider;
      return (
        "<tr>" +
        `<td>${t("family_" + row.category)}</td>` +
        `<td>${provider}</td>` +
        `<td>${term(row.period)}</td>` +
        `<td>${term(row.resolution)}</td>` +
        `<td class="num">${row.features}</td>` +
        `<td>${term(row.license)}</td>` +
        "</tr>"
      );
    })
    .join("");

  const holdout = model.holdout || {};
  const best = Math.max(...Object.values(holdout).filter((value) => value != null));
  const metrics = document.querySelector("#about-metrics tbody");
  metrics.innerHTML = Object.entries(holdout)
    .filter(([, value]) => value != null)
    .sort((a, b) => b[1] - a[1])
    .map(
      ([name, value]) =>
        `<tr class="${value === best ? "best" : ""}">` +
        `<td>${t("model_" + name)}</td>` +
        `<td class="num">${value.toFixed(3).replace(".", lang === "it" ? "," : ".")}</td></tr>`
    )
    .join("");

  const span = (years) => (years ? `${years[0]}-${years[1]}` : "");
  const body = document.querySelector('[data-i18n="about_model_body"]');
  body.textContent = t("about_model_body")
    .replace("{train}", span(model.train_years))
    .replace("{test}", span(model.test_years));
}

function formatCount(value, lang) {
  // Italian leaves four-digit numbers ungrouped by default, and 3174 next to 24.842 reads
  // like a different kind of quantity
  return value.toLocaleString(lang === "it" ? "it-IT" : "en-GB", { useGrouping: "always" });
}
