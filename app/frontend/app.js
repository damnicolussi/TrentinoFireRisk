"use strict";

const state = {
  lang: localStorage.getItem("tfire_lang") || "it",
  meta: null,
  about: null,
  day: null,
  palette: "classes",
  summary: null,
  fires: null,
  visible: new Set(),
  opacity: Number(localStorage.getItem("tfire_opacity")),
  admin: false,
  jobTimer: null,
  calendarMonth: null,
};

const map = { instance: null, base: null, overlay: null, boundary: null, fires: null };
const overlay = { canvas: null, context: null, pixels: null, url: null, objectUrl: null };

const DAYS_PER_YEAR = 365.25;

function t(key) {
  const table = STRINGS[state.lang] || STRINGS.it;
  return table[key] !== undefined ? table[key] : key;
}

function el(id) {
  return document.getElementById(id);
}

function decimal(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return t("no_data");
  const text = value.toFixed(digits);
  return state.lang === "it" ? text.replace(".", ",") : text;
}

function scientific(value) {
  if (value === null || value === undefined) return t("no_data");
  const text = value.toExponential(2);
  return state.lang === "it" ? text.replace(".", ",") : text;
}

function count(value) {
  return value == null
    ? t("no_data")
    : value.toLocaleString(state.lang === "it" ? "it-IT" : "en-GB");
}

function returnPeriod(probability) {
  if (!probability || probability <= 0) return t("no_data");
  const years = 1 / (probability * DAYS_PER_YEAR);
  if (years < 2) return t("years_one");
  const digits = years < 10 ? 1 : 0;
  return t("years_many").replace("{n}", decimal(years, digits));
}

function isoToDisplay(iso) {
  const [year, month, day] = iso.split("-");
  return `${day}/${month}/${year}`;
}

function displayToIso(text) {
  const parts = text.trim().split(/[^0-9]+/).filter(Boolean);
  if (parts.length !== 3) return null;
  const [day, month, year] = parts;
  if (year.length !== 4) return null;
  const iso = `${year}-${month.padStart(2, "0")}-${day.padStart(2, "0")}`;
  const parsed = new Date(iso + "T00:00:00Z");
  return Number.isNaN(parsed.getTime()) || parsed.toISOString().slice(0, 10) !== iso ? null : iso;
}

function todayIso() {
  const now = new Date();
  return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
}

function colorsFor(palette) {
  const danger = state.meta.danger;
  return palette === "rank" ? danger.rank_colors : danger.colors;
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let payload = {};
    try {
      payload = await response.json();
    } catch (error) {
      payload = {};
    }
    const message = payload.reason || payload.detail || `HTTP ${response.status}`;
    const failure = new Error(message);
    failure.status = response.status;
    throw failure;
  }
  return response.json();
}

/* ---- language ---- */

function applyLanguage() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll(".lang button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.lang === state.lang));
  });

  el("day-prev").title = state.lang === "it" ? "giorno precedente" : "previous day";
  el("day-next").title = state.lang === "it" ? "giorno successivo" : "next day";
  el("day-pick").title = t("pick_day");
  el("day").placeholder = t("day_format");
  el("day").title = t("day_format");
  renderFiresTally();

  renderBasemapButtons();
  if (!el("calendar").hidden) renderCalendar(state.calendarMonth);
  if (state.summary) renderDay(state.summary);
  if (state.about) renderAbout(state.about, t, state.lang);
  if (state.admin) refreshAdmin();
  closeDetail();
}

function setLanguage(lang) {
  state.lang = lang;
  localStorage.setItem("tfire_lang", lang);
  applyLanguage();
}

/* ---- calendar ---- */

const WEEK_START = 1;

function locale() {
  return state.lang === "it" ? "it-IT" : "en-GB";
}

function monthOf(iso) {
  return iso.slice(0, 7);
}

function shiftMonth(month, offset) {
  const moved = new Date(month + "-01T00:00:00Z");
  moved.setUTCMonth(moved.getUTCMonth() + offset);
  return moved.toISOString().slice(0, 7);
}

function toggleCalendar() {
  if (el("calendar").hidden) openCalendar();
  else closeCalendar();
}

function openCalendar() {
  el("calendar").hidden = false;
  renderCalendar(monthOf(state.day));
}

function closeCalendar() {
  el("calendar").hidden = true;
}

function renderCalendar(month) {
  state.calendarMonth = month;
  const first = new Date(month + "-01T00:00:00Z");
  const days = new Date(Date.UTC(first.getUTCFullYear(), first.getUTCMonth() + 1, 0)).getUTCDate();
  const offset = (first.getUTCDay() - WEEK_START + 7) % 7;

  const heads = [];
  for (let index = 0; index < 7; index += 1) {
    const sample = new Date(Date.UTC(2024, 0, 1 + index));
    const narrow = sample.toLocaleDateString(locale(), { weekday: "narrow", timeZone: "UTC" });
    heads.push(`<th>${narrow}</th>`);
  }

  const cells = [];
  for (let index = 0; index < offset; index += 1) cells.push("<td></td>");
  for (let day = 1; day <= days; day += 1) {
    const iso = `${month}-${String(day).padStart(2, "0")}`;
    const out = iso < state.meta.first_date || iso > state.meta.last_date;
    const classes = [out ? "out" : "", iso === state.day ? "on" : ""].filter(Boolean).join(" ");
    cells.push(
      `<td><button type="button" data-day="${iso}"${out ? " disabled" : ""}` +
        `${classes ? ` class="${classes}"` : ""}>${day}</button></td>`
    );
  }
  while (cells.length % 7) cells.push("<td></td>");

  const rows = [];
  for (let index = 0; index < cells.length; index += 7) {
    rows.push(`<tr>${cells.slice(index, index + 7).join("")}</tr>`);
  }

  const title = first.toLocaleDateString(locale(), {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  });
  const back = shiftMonth(month, -1) >= monthOf(state.meta.first_date);
  const on = shiftMonth(month, 1) <= monthOf(state.meta.last_date);

  const step = (offsetMonths, enabled, glyph) =>
    `<button type="button" data-month="${shiftMonth(month, offsetMonths)}"` +
    `${enabled ? "" : " disabled"}>${glyph}</button>`;

  el("calendar").innerHTML =
    '<div class="calendar-head">' +
    step(-1, back, "&#9664;") +
    `<span>${title}</span>` +
    step(1, on, "&#9654;") +
    "</div>" +
    `<table><thead><tr>${heads.join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;

  el("calendar")
    .querySelectorAll("[data-month]")
    .forEach((button) => {
      button.addEventListener("click", () => renderCalendar(button.dataset.month));
    });
  el("calendar")
    .querySelectorAll("[data-day]")
    .forEach((button) => {
      button.addEventListener("click", () => loadDay(button.dataset.day));
    });
}

/* ---- map ---- */

function buildMap() {
  const meta = state.meta;
  const bounds = L.latLngBounds(meta.overlay.bounds);
  const margin = meta.map.margin_deg;

  map.instance = L.map("map", {
    minZoom: meta.map.min_zoom,
    maxZoom: meta.map.max_zoom,
    maxBounds: bounds.pad(margin),
    maxBoundsViscosity: 1,
    zoomControl: false,
    attributionControl: true,
  });
  L.control.zoom({ position: "topright" }).addTo(map.instance);

  const fence = () => {
    map.instance.setMinZoom(map.instance.getBoundsZoom(bounds));
  };
  map.instance.fitBounds(bounds);
  fence();
  map.instance.on("resize", fence);

  setBasemap(meta.map.basemaps[0].key);

  fetch("/api/boundary.geojson")
    .then((response) => response.json())
    .then((geojson) => {
      map.boundary = L.geoJSON(geojson, {
        style: { color: "#343a40", weight: 1.6, fill: false, opacity: 0.85 },
        interactive: false,
      }).addTo(map.instance);
    });

  map.instance.on("click", (event) => {
    loadCell(event.latlng.lng, event.latlng.lat);
  });
}

function setBasemap(key) {
  const chosen = state.meta.map.basemaps.find((basemap) => basemap.key === key);
  if (!chosen) return;
  if (map.base) map.instance.removeLayer(map.base);
  map.base = L.tileLayer(chosen.url, {
    attribution: chosen.attribution,
    maxZoom: state.meta.map.max_zoom,
  }).addTo(map.instance);
  if (map.overlay) map.overlay.bringToFront();
  if (map.boundary) map.boundary.bringToFront();
  if (map.fires) map.fires.bringToFront();
  state.basemap = key;
  renderBasemapButtons();
}

function renderBasemapButtons() {
  if (!state.meta) return;
  el("basemaps").innerHTML = state.meta.map.basemaps
    .map(
      (basemap) =>
        `<button type="button" data-basemap="${basemap.key}" ` +
        `aria-pressed="${basemap.key === state.basemap}">${t("basemap_" + basemap.key)}</button>`
    )
    .join("");
  el("basemaps")
    .querySelectorAll("button")
    .forEach((button) => {
      button.addEventListener("click", () => setBasemap(button.dataset.basemap));
    });
}

function packed(hex) {
  return [1, 3, 5].map((start) => parseInt(hex.slice(start, start + 2), 16));
}

async function loadOverlaySource(url) {
  if (overlay.url === url && overlay.pixels) return;

  const image = await new Promise((resolve, reject) => {
    const element = new Image();
    element.onload = () => resolve(element);
    element.onerror = () => reject(new Error(url));
    element.src = url;
  });

  overlay.canvas = overlay.canvas || document.createElement("canvas");
  overlay.canvas.width = image.naturalWidth;
  overlay.canvas.height = image.naturalHeight;
  overlay.context = overlay.canvas.getContext("2d", { willReadFrequently: true });
  overlay.context.clearRect(0, 0, overlay.canvas.width, overlay.canvas.height);
  overlay.context.drawImage(image, 0, 0);
  overlay.pixels = overlay.context.getImageData(0, 0, overlay.canvas.width, overlay.canvas.height);
  overlay.url = url;
}

async function paintOverlay() {
  const hidden = colorsFor(state.palette)
    .map((hex, index) => (state.visible.has(index) ? null : packed(hex)))
    .filter(Boolean);

  const source = overlay.pixels.data;
  const painted = new ImageData(overlay.canvas.width, overlay.canvas.height);
  painted.data.set(source);

  if (hidden.length) {
    const data = painted.data;
    for (let at = 0; at < data.length; at += 4) {
      if (!data[at + 3]) continue;
      for (const [red, green, blue] of hidden) {
        if (data[at] === red && data[at + 1] === green && data[at + 2] === blue) {
          data[at + 3] = 0;
          break;
        }
      }
    }
  }

  overlay.context.putImageData(painted, 0, 0);
  const blob = await new Promise((resolve) => overlay.canvas.toBlob(resolve, "image/png"));
  const next = URL.createObjectURL(blob);

  const bounds = L.latLngBounds(state.meta.overlay.bounds);
  if (map.overlay) map.instance.removeLayer(map.overlay);
  map.overlay = L.imageOverlay(next, bounds, {
    opacity: state.opacity,
    interactive: false,
  }).addTo(map.instance);

  if (overlay.objectUrl) URL.revokeObjectURL(overlay.objectUrl);
  overlay.objectUrl = next;

  if (map.boundary) map.boundary.bringToFront();
  if (map.fires) map.fires.bringToFront();
}

async function drawOverlay() {
  const token = state.meta.danger.token;
  await loadOverlaySource(
    `/api/risk/${state.day}/overlay.png?palette=${state.palette}&v=${token}`
  );
  await paintOverlay();
}

/* ---- the day ---- */

function busy(message) {
  const node = el("status");
  if (!message) {
    node.hidden = true;
    return;
  }
  node.hidden = false;
  node.textContent = message;
  node.classList.toggle("busy", message === t("computing") || message === t("loading"));
}

async function loadDay(day) {
  state.day = day;
  el("day").value = isoToDisplay(day);
  closeCalendar();
  // the project page is also a route, so loading a day must not navigate away from it
  const target = `#map/${day}`;
  sessionStorage.setItem("tfire_day", day);
  if (!location.hash.startsWith("#about") && location.hash !== target) location.hash = target;
  closeDetail();
  busy(t("computing"));

  try {
    const summary = await api(`/api/risk/${day}`);
    state.summary = summary;
    renderDay(summary);
    await drawOverlay();
    busy(null);
  } catch (error) {
    state.summary = null;
    busy(error.message);
    if (map.overlay) {
      map.instance.removeLayer(map.overlay);
      map.overlay = null;
    }
    el("provenance").textContent = "";
  }

  loadFires();
}

function rankLabel(index, percentiles) {
  if (index === 0) return t("rank_below_median");
  if (index === 1) return t("rank_above_median");
  return t("rank_top").replace("{pct}", decimal(100 - percentiles[index - 1], 0));
}

function rankFooter(summary) {
  const breaks = state.meta.danger.breaks;
  const highest = summary.probability.max;
  if (highest == null || highest < breaks[1]) return t("legend_rank_quiet");

  const keys = state.meta.danger.class_keys;
  let index = 0;
  while (index < breaks.length && highest >= breaks[index]) index += 1;
  return t("legend_rank_max")
    .replace("{period}", returnPeriod(highest))
    .replace("{label}", t(keys[index]));
}

function renderDay(summary) {
  const meta = state.meta;
  const absolute = state.palette === "classes";
  const colors = colorsFor(state.palette);
  const rows = meta.danger.class_keys
    .map((key, index) => {
      const edge = index === 0 ? 0 : index - 1;
      const label = absolute ? t(key) : rankLabel(index, meta.danger.rank_percentiles);
      const value = absolute
        ? `${index === 0 ? "<" : "≥"} ${returnPeriod(meta.danger.breaks[edge])}`
        : `${index === 0 ? "<" : "≥"} ${meta.danger.rank_percentiles[edge]}%`;
      const hint = absolute ? ` title="${scientific(meta.danger.breaks[edge])}"` : "";
      const cells = absolute ? `<td class="count">${count(summary.classes[key])}</td>` : "";
      const swatch = `<span class="swatch" style="background:${colors[index]}"></span>`;
      const off = state.visible.has(index) ? "" : " class=\"off\"";
      return (
        `<tr data-class="${index}"${off}>` +
        `<td class="swatch-cell">${swatch}</td>` +
        `<td title="${t("toggle_class")}">${label}</td>` +
        `<td class="break"${hint}>${value}</td>` +
        cells +
        "</tr>"
      );
    })
    .reverse();

  const body = el("legend").querySelector("tbody");
  body.innerHTML = rows.join("");
  body.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("click", () => toggleClass(Number(row.dataset.class)));
  });
  el("legend-all").hidden = state.visible.size === meta.danger.class_keys.length;

  const note = absolute
    ? t("legend_note_classes").replace("{years}", meta.danger.reference_years.join("-"))
    : `${t("legend_note_rank")} ${rankFooter(summary)}`;
  el("legend-note").textContent = note;

  const provenance = summary.provenance;
  const sources = provenance.sources
    .map((source) => {
      const kind = source.split(":")[0];
      const label = t("source_" + kind);
      return label === "source_" + kind ? source : label;
    })
    .join(" + ");
  const vegetation = t("veg_composite")
    .replace("{date}", String(provenance.vegetation.composite).slice(0, 7))
    .replace("{age}", decimal(provenance.vegetation.mean_age_days, 0));

  el("provenance").textContent =
    `${sources} · ${vegetation} · ` +
    `${t("cells_scored").replace("{n}", count(provenance.cells))} · ` +
    `${t("model")} ${provenance.model_version}`;
}

function toggleClass(index) {
  if (state.visible.has(index)) state.visible.delete(index);
  else state.visible.add(index);
  if (state.summary) renderDay(state.summary);
  if (overlay.pixels) paintOverlay();
}

function showAllClasses() {
  state.meta.danger.class_keys.forEach((_, index) => state.visible.add(index));
  if (state.summary) renderDay(state.summary);
  if (overlay.pixels) paintOverlay();
}

function setOpacity(value) {
  state.opacity = value;
  localStorage.setItem("tfire_opacity", String(value));
  el("opacity").value = String(Math.round(value * 100));
  el("opacity-value").textContent = `${Math.round(value * 100)}%`;
  if (map.overlay) map.overlay.setOpacity(value);
}

function stepDay(offset) {
  const moved = new Date(state.day + "T00:00:00Z");
  moved.setUTCDate(moved.getUTCDate() + offset);
  const next = moved.toISOString().slice(0, 10);
  if (next < state.meta.first_date || next > state.meta.last_date) return;
  loadDay(next);
}

/* ---- cell detail ---- */

async function loadCell(lon, lat) {
  if (!state.summary) return;
  try {
    const cell = await api(`/api/risk/${state.day}/cell?lon=${lon}&lat=${lat}`);
    renderCell(cell);
  } catch (error) {
    if (error.status === 404) closeDetail();
  }
}

function renderCell(cell) {
  const meta = state.meta;
  const key = meta.danger.class_keys[cell.danger_class];
  const color = meta.danger.colors[cell.danger_class];

  // an identifier, not a quantity: no thousands separator
  el("detail-title").textContent = `${t("cell")} ${cell.cell_id}`;
  el("detail-coords").textContent =
    `${decimal(cell.lon, 4)} E  ${decimal(cell.lat, 4)} N` +
    (cell.elevation_m === null ? "" : `  ·  ${decimal(cell.elevation_m, 0)} m`);

  const rarity =
    cell.record_percentile === null || cell.record_percentile === undefined
      ? ""
      : `<tr><th>${t("record_percentile")}</th><td>` +
        t("record_percentile_note")
          .replace("{pct}", decimal(cell.record_percentile, 1))
          .replace("{years}", meta.danger.reference_years.join("-")) +
        "</td></tr>";

  el("detail-risk").innerHTML =
    `<tr><th>${t("danger_class")}</th><td>` +
    `<span class="class-tag" style="background:${color}">${t(key)}</span></td></tr>` +
    `<tr><th>${t("return_period")}</th><td>` +
    t("return_period_note").replace("{period}", returnPeriod(cell.probability)) +
    "</td></tr>" +
    rarity +
    `<tr><th>${t("probability")}</th><td>${scientific(cell.probability)}</td></tr>` +
    `<tr><th>${t("percentile")}</th><td>${decimal(cell.rank * 100, 1)}</td></tr>`;

  const units = {
    fwi: ["", 1],
    temp_mean: [" °C", 1],
    rh_mean: [" %", 0],
    wind_speed_mean: [" m/s", 1],
    precip_cum30: [" mm", 0],
    ndvi: ["", 2],
  };
  el("detail-drivers").innerHTML = Object.entries(units)
    .map(([name, [unit, digits]]) => {
      const value = cell.drivers[name];
      const text = value === null ? t("no_data") : decimal(value, digits) + unit;
      return `<tr><th>${t(name)}</th><td>${text}</td></tr>`;
    })
    .join("");

  el("detail").hidden = false;
}

function closeDetail() {
  el("detail").hidden = true;
}

/* ---- recorded fires ---- */

async function loadFires() {
  if (map.fires) {
    map.instance.removeLayer(map.fires);
    map.fires = null;
  }
  state.fires = null;
  renderFiresTally();
  if (!el("show-fires").checked) return;

  let geojson;
  try {
    geojson = await api(`/api/risk/${state.day}/fires`);
  } catch (error) {
    state.fires = "failed";
    renderFiresTally();
    return;
  }

  const label = (properties) =>
    `<strong>${properties.loc || ""}</strong><br>` +
    `${decimal(properties.area_ha, 1)} ha` +
    (properties.start_hour === null ? "" : ` · ${properties.start_hour}:00`);

  map.fires = L.featureGroup().addTo(map.instance);
  L.geoJSON(geojson, {
    style: { color: "#0079c0", weight: 2, fillColor: "#0079c0", fillOpacity: 0.3 },
    onEachFeature: (feature, layer) => layer.bindPopup(label(feature.properties)),
  }).addTo(map.fires);

  L.geoJSON(geojson, {
    style: { opacity: 0, fillOpacity: 0 },
    onEachFeature: (feature, layer) => {
      L.circleMarker(layer.getBounds().getCenter(), {
        radius: 6,
        color: "#0079c0",
        weight: 2,
        fillColor: "#ffffff",
        fillOpacity: 0.85,
      })
        .bindPopup(label(feature.properties))
        .addTo(map.fires);
    },
  });

  state.fires = geojson.features.length;
  renderFiresTally();
}

function renderFiresTally() {
  const tally = el("fires-tally");
  if (!tally) return;
  tally.classList.toggle("bad", state.fires === "failed");
  if (state.fires === null) tally.textContent = "";
  else if (state.fires === "failed") tally.textContent = t("fires_failed");
  else tally.textContent = t("fires_tally").replace("{n}", count(state.fires));
}

/* ---- views ---- */

function showView(name) {
  el("view-map").hidden = name !== "map";
  el("view-about").hidden = name !== "about";
  el("datebar").hidden = name !== "map";
  document.querySelectorAll("[data-view]").forEach((node) => {
    node.classList.toggle("current", node.dataset.view === name && node.classList.contains("navlink"));
  });
  if (name === "map" && map.instance) map.instance.invalidateSize();
  if (name === "about" && !state.about) {
    api("/api/about").then((about) => {
      state.about = about;
      renderAbout(about, t, state.lang);
    });
  }
}

function routeFromHash() {
  const [view, day] = location.hash.replace(/^#/, "").split("/");
  if (view === "about") {
    showView("about");
    return;
  }
  showView("map");
  if (day && day !== state.day) loadDay(day);
}

/* ---- admin ---- */

function openAdminModal() {
  if (state.admin) {
    el("admin-drawer").hidden = false;
    refreshAdmin();
    return;
  }
  el("admin-error").hidden = true;
  el("admin-password").value = "";
  el("admin-modal").hidden = false;
  el("admin-password").focus();
}

async function signIn(event) {
  event.preventDefault();
  const error = el("admin-error");
  try {
    await api("/api/admin/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: el("admin-password").value }),
    });
  } catch (failure) {
    error.textContent = failure.status === 503 ? t("admin_disabled") : t("wrong_password");
    error.hidden = false;
    return;
  }
  state.admin = true;
  el("admin-modal").hidden = true;
  el("admin-drawer").hidden = false;
  el("admin-open").textContent = t("admin_title");
  refreshAdmin();
}

async function signOut() {
  await fetch("/api/admin/logout", { method: "POST" });
  state.admin = false;
  clearInterval(state.jobTimer);
  state.jobTimer = null;
  el("admin-drawer").hidden = true;
  el("admin-open").textContent = t("sign_in");
}

async function refreshAdmin() {
  if (!state.admin) return;
  let status;
  let versions;
  let jobs;
  try {
    [status, versions, jobs] = await Promise.all([
      api("/api/admin/status"),
      api("/api/admin/versions"),
      api("/api/admin/jobs"),
    ]);
  } catch (error) {
    if (error.status === 401) signOut();
    return;
  }

  const window = status.warm_window;
  const rows = [
    [t("model_version"), status.model_version],
    [t("holdout_auprc"), decimal(status.holdout_auprc, 3)],
    [t("meteo_through"), status.meteo_through || t("no_data")],
    [t("fwi_through"), status.fwi_through || t("no_data")],
    [t("vegetation_through"), status.vegetation_through || t("no_data")],
    [
      t("risk_maps"),
      `${count(status.risk_maps.count)} · ${(status.risk_maps.bytes / 1e6).toFixed(0)} MB`,
    ],
    [t("warm_window"), `${window[0]} → ${window[window.length - 1]}`],
    [
      t("config_match"),
      `<span class="${status.config_matches_training ? "ok" : "bad"}">` +
        `${status.config_matches_training ? t("yes") : t("no")}</span>`,
    ],
  ];
  el("admin-status").innerHTML = rows
    .map(([label, value]) => `<tr><th>${label}</th><td>${value}</td></tr>`)
    .join("");

  el("admin-version").innerHTML = versions.available
    .map(
      (version) =>
        `<option value="${version}" ${version === versions.active ? "selected" : ""}>` +
        `${version}</option>`
    )
    .join("");

  const running = jobs.jobs.find((job) => job.running);
  const title = running ? t("job_running").replace("{action}", t("job_" + running.action)) : "";
  el("admin-actions").innerHTML = jobs.actions
    .map(
      (action) =>
        `<button type="button" class="ghost" data-action="${action}" ` +
        `${running ? "disabled" : ""} title="${title}">${t("job_" + action)}</button>`
    )
    .join("");
  el("admin-actions")
    .querySelectorAll("button")
    .forEach((button) => {
      button.addEventListener("click", () => startJob(button.dataset.action));
    });

  const latest = jobs.jobs[0];
  el("admin-log").textContent = latest
    ? `${latest.action} · ${latest.started}` +
      (latest.running ? " · …" : ` · exit ${latest.returncode}`) +
      "\n\n" +
      latest.log_tail.join("\n")
    : "";
  el("admin-log").scrollTop = el("admin-log").scrollHeight;

  if (running && !state.jobTimer) {
    state.jobTimer = setInterval(refreshAdmin, 3000);
  } else if (!running && state.jobTimer) {
    clearInterval(state.jobTimer);
    state.jobTimer = null;
  }
}

async function startJob(action) {
  const body = action === "predict" ? { action, date: state.day, days: 1 } : { action };
  try {
    await api("/api/admin/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (error) {
    el("admin-log").textContent = error.message;
    return;
  }
  refreshAdmin();
}

/* ---- boot ---- */

async function boot() {
  document.querySelectorAll(".lang button").forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.lang));
  });
  el("day-prev").addEventListener("click", () => stepDay(-1));
  el("day-next").addEventListener("click", () => stepDay(1));
  el("day-today").addEventListener("click", () => loadDay(startingDay(todayIso())));
  el("day").addEventListener("change", (event) => {
    const iso = displayToIso(event.target.value);
    if (iso === null || iso < state.meta.first_date || iso > state.meta.last_date) {
      event.target.value = isoToDisplay(state.day);
      return;
    }
    loadDay(iso);
  });
  el("day-pick").addEventListener("click", toggleCalendar);
  document.addEventListener(
    "click",
    (event) => {
      if (!el("datebar").contains(event.target)) closeCalendar();
    },
    true
  );
  el("detail-close").addEventListener("click", closeDetail);
  el("show-fires").addEventListener("change", loadFires);
  el("legend-all").addEventListener("click", showAllClasses);
  el("opacity").addEventListener("input", (event) => setOpacity(Number(event.target.value) / 100));
  document.querySelectorAll('input[name="palette"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      state.palette = radio.value;
      state.meta.danger.class_keys.forEach((_, index) => state.visible.add(index));
      if (state.summary) {
        renderDay(state.summary);
        drawOverlay();
      }
    });
  });
  el("admin-open").addEventListener("click", openAdminModal);
  el("admin-cancel").addEventListener("click", () => {
    el("admin-modal").hidden = true;
  });
  el("admin-close").addEventListener("click", () => {
    el("admin-drawer").hidden = true;
  });
  el("admin-login").addEventListener("submit", signIn);
  el("admin-logout").addEventListener("click", signOut);
  el("admin-version-apply").addEventListener("click", async () => {
    await api("/api/admin/versions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version: el("admin-version").value }),
    });
    state.meta = await api("/api/meta");
    refreshAdmin();
    loadDay(state.day);
  });
  window.addEventListener("hashchange", routeFromHash);
  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT") return;
    if (event.key === "ArrowLeft") stepDay(-1);
    if (event.key === "ArrowRight") stepDay(1);
    if (event.key === "Escape") {
      closeCalendar();
      closeDetail();
      el("admin-modal").hidden = true;
    }
  });

  state.meta = await api("/api/meta");
  if (!localStorage.getItem("tfire_lang")) state.lang = state.meta.default_language;
  state.meta.danger.class_keys.forEach((_, index) => state.visible.add(index));
  setOpacity(state.opacity > 0 ? state.opacity : state.meta.overlay.opacity);

  applyLanguage();
  buildMap();

  showView(location.hash.startsWith("#about") ? "about" : "map");
  await loadDay(openingDay());
}

function openingDay() {
  const requested = location.hash.split("/")[1];
  const ours = sessionStorage.getItem("tfire_day");
  const shared = requested && requested !== ours && /^\d{4}-\d{2}-\d{2}$/.test(requested);
  return startingDay(shared ? requested : todayIso());
}

function startingDay(iso) {
  if (iso < state.meta.first_date) return state.meta.first_date;
  return iso > state.meta.last_date ? state.meta.last_date : iso;
}

boot();
