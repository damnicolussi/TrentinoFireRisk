"use strict";

const STRINGS = {
  it: {
    today: "oggi",
    day_format: "gg/mm/aaaa",
    pick_day: "scegli la data",
    nav_about: "Progetto",
    nav_map: "Mappa",
    sign_in: "Accedi",
    sign_out: "Esci",
    cancel: "Annulla",
    apply: "Applica",
    password: "Password",
    sign_in_hint: "Area riservata all'operatore. Non serve per consultare la mappa.",
    disclaimer:
      "Prototipo di ricerca. Non è un bollettino ufficiale di pericolo incendi e non ha " +
      "valore operativo o legale: per le allerte reali fare riferimento alla Provincia " +
      "autonoma di Trento.",

    palette_classes: "Classi assolute",
    palette_rank: "Percentile del giorno",
    layer_fires: "Incendi storici del giorno",
    basemap_osm: "Mappa",
    basemap_satellite: "Satellite",
    basemap_relief: "Rilievo",

    legend_title: "Pericolo di innesco",
    opacity: "Opacità",
    toggle_class: "Clic per mostrare o nascondere questa classe",
    show_all: "mostra tutte le classi",
    legend_note_classes:
      "Soglie fisse sulla probabilità giornaliera per cella: i percentili 90, 99, 99,9 e " +
      "99,99 di tutte le celle in tutti i giorni {years}. Le classi dicono quanto raro è " +
      "un valore rispetto al record.",
    legend_note_rank:
      "Percentile all'interno del solo giorno mostrato: la scala si riempie sempre, anche " +
      "quando il giorno è tranquillo. Non confrontabile fra date.",
    legend_rank_max: "Massimo del giorno: {period}, classe «{label}».",
    legend_rank_quiet: "Anche il massimo di questo giorno resta nella norma del record.",

    very_low: "nella norma",
    low: "sopra la norma",
    moderate: "alto",
    high: "molto alto",
    very_high: "estremo",

    rank_top: "top {pct}%",
    rank_above_median: "sopra la mediana",
    rank_below_median: "sotto la mediana",

    cell: "Cella",
    elevation: "quota",
    danger_class: "classe",
    probability: "probabilità",
    percentile: "percentile del giorno",
    return_period: "tempo di ritorno",
    return_period_note: "circa 1 innesco ogni {period} per cella, a condizioni costanti",
    record_percentile: "rarità nel record",
    record_percentile_note: "più alta del {pct}% dei giorni-cella {years}",
    years_one: "1 anno",
    years_many: "{n} anni",
    drivers: "Fattori del giorno",
    fwi: "FWI",
    temp_mean: "temperatura media",
    rh_mean: "umidità relativa",
    wind_speed_mean: "vento medio",
    precip_cum30: "pioggia 30 giorni",
    ndvi: "NDVI",
    no_data: "non disponibile",

    computing: "Calcolo della mappa in corso…",
    loading: "Caricamento…",
    outside: "Punto fuori dal territorio modellato.",
    cells_scored: "{n} celle",
    model: "modello",
    veg_composite: "composito Landsat {date} ({age} giorni)",
    source_cached: "tabelle storiche in cache",
    source_archive: "archivio Open-Meteo",
    source_forecast: "previsione ECMWF IFS",
    fires_tally: "({n} a catasto)",
    fires_failed: "(catasto non disponibile)",

    about_title: "Il progetto",
    about_lede:
      "Trentino Fire Risk stima, per ogni giorno e per ogni cella di 500 metri della provincia " +
      "di Trento, la probabilità che si inneschi un incendio boschivo. È il sistema " +
      "sviluppato per una tesi di laurea, addestrato sul catasto incendi della Provincia e su " +
      "quarantuno anni di dati ambientali.",
    about_how_title: "Come funziona",
    about_how_body:
      "Il territorio è diviso in una griglia regolare. Per ogni cella e per ogni giorno il " +
      "sistema mette insieme meteorologia, indici di pericolo, stato della vegetazione, " +
      "morfologia, uso del suolo e presenza umana, e passa il tutto a un modello addestrato sugli " +
      "inneschi realmente avvenuti. Le date passate sono ricostruite dalle serie storiche, quelle " +
      "future dalle previsioni meteo, fino a dieci giorni avanti.",
    about_data_title: "I dati",
    about_data_body:
      "Le fonti sono tutte pubbliche. La tabella è generata dal registro delle feature usato dalla " +
      "pipeline, quindi descrive quello che il modello riceve davvero.",
    about_model_title: "Il modello e quanto vale",
    about_model_body:
      "Un modello ad alberi (XGBoost) addestrato sugli anni {train}, con gli anni {test} tenuti " +
      "completamente fuori dall'addestramento e dalla scelta degli iperparametri. Il confronto " +
      "sotto è su quegli anni mai visti; la metrica è l'area sotto la curva precisione/richiamo, " +
      "dove gli inneschi sono meno del 4% dei casi.",
    colophon_affiliation: "Università di Trento",
    colophon_data: "Dati incendi: Servizio Foreste della Provincia autonoma di Trento.",
    colophon_code: "Codice sorgente:",

    family_topography: "Morfologia",
    family_geography: "Geografia",
    family_landcover: "Uso del suolo",
    family_human: "Presenza umana",
    family_calendar: "Calendario",
    family_vegetation: "Vegetazione",
    family_meteo: "Meteorologia",
    family_fwi: "Indici di pericolo",
    family_stacking: "Transfer da altri incendi",

    col_family: "Famiglia",
    col_source: "Fonte",
    col_period: "Periodo",
    col_resolution: "Risoluzione",
    col_features: "Feature",
    col_license: "Licenza",
    col_model: "Modello",
    col_auprc: "AUPRC sugli anni tenuti fuori",
    fig_pr: "Curve precisione/richiamo sugli anni tenuti fuori.",
    fig_shap: "Contributo delle famiglie di feature alla previsione.",

    n_years: "anni di record",
    n_fires: "inneschi a catasto",
    n_cells: "celle da 500 m",
    n_features: "feature per cella-giorno",

    model_xgboost: "Trentino Fire Risk",
    model_random_forest: "Random forest",
    model_logistic: "Regressione logistica",
    model_fwi_only: "Solo FWI",

    admin_title: "Amministrazione",
    admin_freshness: "Stato dei dati",
    admin_versions: "Versione del modello",
    admin_jobs: "Operazioni",
    job_predict: "Calcola oggi",
    job_warm: "Precalcola la finestra",
    "job_refresh-vegetation": "Aggiorna vegetazione",
    "job_refresh-era5": "Estendi il backbone meteo",
    "job_rebuild-dataset": "Ricostruisci dataset",
    job_retrain: "Riaddestra",
    job_running: "In esecuzione: {action}",
    wrong_password: "Password errata.",
    admin_disabled: "Nessuna password amministratore configurata sul server.",

    meteo_through: "meteo fino al",
    fwi_through: "FWI fino al",
    vegetation_through: "vegetazione fino al",
    risk_maps: "mappe in cache",
    model_version: "versione attiva",
    holdout_auprc: "AUPRC anni tenuti fuori",
    config_match: "config uguale a quella di addestramento",
    warm_window: "finestra sempre pronta",
    yes: "sì",
    no: "no",
  },

  en: {
    today: "today",
    day_format: "dd/mm/yyyy",
    pick_day: "pick a date",
    nav_about: "Project",
    nav_map: "Map",
    sign_in: "Sign in",
    sign_out: "Sign out",
    cancel: "Cancel",
    apply: "Apply",
    password: "Password",
    sign_in_hint: "Operator area. Not needed to read the map.",
    disclaimer:
      "Research prototype. This is not an official fire danger bulletin and has no " +
      "operational or legal standing: for real warnings refer to the Autonomous Province " +
      "of Trento.",

    palette_classes: "Absolute classes",
    palette_rank: "Within-day percentile",
    layer_fires: "Recorded fires of the day",
    basemap_osm: "Map",
    basemap_satellite: "Satellite",
    basemap_relief: "Relief",

    legend_title: "Ignition danger",
    opacity: "Opacity",
    toggle_class: "Click to show or hide this class",
    show_all: "show every class",
    legend_note_classes:
      "Fixed thresholds on the daily per-cell probability: the 90th, 99th, 99.9th and " +
      "99.99th percentiles of every cell on every day of {years}. The classes say how rare " +
      "a value is against the record.",
    legend_note_rank:
      "Percentile within the displayed day only: the scale always fills, even on a quiet " +
      "day. Not comparable across dates.",
    legend_rank_max: "Highest cell today: {period}, class “{label}”.",
    legend_rank_quiet: "Even today's highest cell stays within the norm of the record.",

    very_low: "within the norm",
    low: "above the norm",
    moderate: "high",
    high: "very high",
    very_high: "extreme",

    rank_top: "top {pct}%",
    rank_above_median: "above the median",
    rank_below_median: "below the median",

    cell: "Cell",
    elevation: "elevation",
    danger_class: "class",
    probability: "probability",
    percentile: "percentile within the day",
    return_period: "return period",
    return_period_note: "about 1 ignition every {period} per cell, conditions unchanged",
    record_percentile: "rarity in the record",
    record_percentile_note: "higher than {pct}% of all cell-days in {years}",
    years_one: "1 year",
    years_many: "{n} years",
    drivers: "Drivers on the day",
    fwi: "FWI",
    temp_mean: "mean temperature",
    rh_mean: "relative humidity",
    wind_speed_mean: "mean wind",
    precip_cum30: "rain, 30 days",
    ndvi: "NDVI",
    no_data: "not available",

    computing: "Computing the map…",
    loading: "Loading…",
    outside: "That point is outside the modeled area.",
    cells_scored: "{n} cells",
    model: "model",
    veg_composite: "Landsat composite {date} ({age} days old)",
    source_cached: "cached historical tables",
    source_archive: "Open-Meteo archive",
    source_forecast: "ECMWF IFS forecast",
    fires_tally: "({n} on record)",
    fires_failed: "(cadastre unavailable)",

    about_title: "The project",
    about_lede:
      "Trentino Fire Risk estimates, for every day and every 500 metre cell of the province of " +
      "Trento, the probability that a wildfire ignites. It is the system built for an MSc " +
      "thesis, trained on the provincial fire cadastre and on forty-one years of environmental " +
      "data.",
    about_how_title: "How it works",
    about_how_body:
      "The province is cut into a regular grid. For each cell and each day the system assembles " +
      "weather, fire-danger indices, vegetation state, terrain, land cover and human presence, " +
      "and hands all of it to a model trained on the ignitions that actually happened. Past " +
      "dates are rebuilt from the historical series, future ones from the weather forecast, up " +
      "to ten days ahead.",
    about_data_title: "The data",
    about_data_body:
      "Every source is public. The table below is generated from the feature registry the " +
      "pipeline is validated against, so it describes what the model actually receives.",
    about_model_title: "The model, and what it is worth",
    about_model_body:
      "A gradient-boosted tree model (XGBoost) trained on {train}, with {test} held out of both " +
      "the fitting and the hyperparameter search. The comparison below is on those unseen years; " +
      "the metric is the area under the precision-recall curve, where ignitions are under 4% of " +
      "the cases.",
    colophon_affiliation: "University of Trento",
    colophon_data: "Fire data: Forest Service of the Autonomous Province of Trento.",
    colophon_code: "Source code:",

    family_topography: "Terrain",
    family_geography: "Geography",
    family_landcover: "Land cover",
    family_human: "Human presence",
    family_calendar: "Calendar",
    family_vegetation: "Vegetation",
    family_meteo: "Meteorology",
    family_fwi: "Fire danger indices",
    family_stacking: "Transfer from other fires",

    col_family: "Family",
    col_source: "Source",
    col_period: "Period",
    col_resolution: "Resolution",
    col_features: "Features",
    col_license: "License",
    col_model: "Model",
    col_auprc: "AUPRC on the held-out years",
    fig_pr: "Precision-recall curves on the held-out years.",
    fig_shap: "Contribution of each feature family to the prediction.",

    n_years: "years of record",
    n_fires: "recorded ignitions",
    n_cells: "cells of 500 m",
    n_features: "features per cell-day",

    model_xgboost: "Trentino Fire Risk",
    model_random_forest: "Random forest",
    model_logistic: "Logistic regression",
    model_fwi_only: "FWI only",

    admin_title: "Administration",
    admin_freshness: "Data freshness",
    admin_versions: "Model version",
    admin_jobs: "Operations",
    job_predict: "Predict today",
    job_warm: "Precompute the window",
    "job_refresh-vegetation": "Refresh vegetation",
    "job_refresh-era5": "Extend the weather backbone",
    "job_rebuild-dataset": "Rebuild dataset",
    job_retrain: "Retrain",
    job_running: "Running: {action}",
    wrong_password: "Wrong password.",
    admin_disabled: "No admin password is configured on the server.",

    meteo_through: "weather through",
    fwi_through: "FWI through",
    vegetation_through: "vegetation through",
    risk_maps: "cached maps",
    model_version: "active version",
    holdout_auprc: "held-out AUPRC",
    config_match: "config identical to the training one",
    warm_window: "always-warm window",
    yes: "yes",
    no: "no",
  },
};

/* Registry values are American English, like the rest of the code. Only the words that are
   not numbers or proper nouns need an Italian rendering; anything absent passes through. */
const TERMS = {
  it: {
    static: "statico",
    "static (TanDEM-X acquisitions 2011-2015)": "statico (acquisizioni TanDEM-X 2011-2015)",
    "any date": "qualunque data",
    daily: "giornaliera",
    computed: "calcolato",
    "computed with xclim": "calcolato con xclim",
    "30 m to 500 m": "30 m → 500 m",
    "100 m to 500 m": "100 m → 500 m",
    "vector to 500 m": "vettoriale → 500 m",
    "vector and 100 m to 500 m": "vettoriale e 100 m → 500 m",
    "0.1 degrees (~9 km)": "0,1 gradi (~9 km)",
    "0.1 degrees (~9 km), 0.25 degrees on the served tail":
      "0,1 gradi (~9 km), 0,25 gradi sulla coda servita",
    "as the ERA5-Land backbone": "come il backbone ERA5-Land",
    "1984-2024, monthly composites": "1984-2024, compositi mensili",
    "1984 to 2026": "1984-2026",
    "1984 to 2026, hourly to daily": "1984-2026, oraria → giornaliera",
    "OSM snapshot; Natura 2000 2019; WorldPop 2000-2020":
      "istantanea OSM; Natura 2000 2019; WorldPop 2000-2020",
    "Copernicus, free and open": "Copernicus, libera e aperta",
    "PAT open data": "PAT, dati aperti",
    "USGS, public domain": "USGS, dominio pubblico",
    "ODbL (OSM); CC BY 4.0 (WorldPop); PAT open data (Natura 2000)":
      "ODbL (OSM); CC BY 4.0 (WorldPop); dati aperti PAT (Natura 2000)",
  },
  en: {
    "30 m to 500 m": "30 m → 500 m",
    "100 m to 500 m": "100 m → 500 m",
    "vector to 500 m": "vector → 500 m",
    "vector and 100 m to 500 m": "vector and 100 m → 500 m",
    "1984 to 2026": "1984-2026",
    "1984 to 2026, hourly to daily": "1984-2026, hourly → daily",
  },
};
