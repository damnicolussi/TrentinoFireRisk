# Serving image: the code and its dependencies, no data. The data subset a prediction needs is
# assembled on the build host by `tfire package-runtime` and mounted at /app/data and /app/models.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # xclim compiles the fire indices through numba, which caches next to its own source unless
    # told otherwise. The container does not run as the owner of site-packages, so without this
    # every date that recomputes the FWI dies on "no locator available".
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    MPLCONFIGDIR=/tmp/matplotlib-cache

WORKDIR /app

# rasterio's wheel bundles GDAL but links libexpat from the system, which the slim image omits
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY app/ ./app/

# editable, so `find_project_root` lands on /app: it walks up from the package file looking for
# pyproject.toml, and a site-packages install has none above it.
#
# `sources` is not optional for a serving container even though it fetches nothing: inference
# recomputes the FWI with xclim, reads rasters through rasterio, and `features.human` imports
# osmnx at module level. `viz` is here for the retrain action, which needs matplotlib to write
# its figures on a host that mounts the build tree.
RUN pip install -e ".[sources,models,viz,app]" \
    && useradd --create-home --uid 10001 tfire \
    && chown -R tfire:tfire /app

USER tfire

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz').read()"]

# one worker: the sessions, the job slot, the per-date locks and the rate counters live in the
# process, so a second worker would answer with a different half of the state
CMD ["tfire", "serve", "--host", "0.0.0.0", "--port", "8000"]
