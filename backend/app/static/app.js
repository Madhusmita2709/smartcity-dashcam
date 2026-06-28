const state = {
  jobs: [],
  customClasses: [],
  violations: [],
  map: null,
  markersLayer: null,
  heatLayer: null,
};

const elements = {
  uploadForm: document.getElementById("uploadForm"),
  videoFiles: document.getElementById("videoFiles"),
  audioRemoval: document.getElementById("audioRemoval"),
  faceBlurEnabled: document.getElementById("faceBlurEnabled"),
  faceBlurMethod: document.getElementById("faceBlurMethod"),
  faceBlurIntensity: document.getElementById("faceBlurIntensity"),
  faceBlurIntensityValue: document.getElementById("faceBlurIntensityValue"),
  frameMethod: document.getElementById("frameMethod"),
  frameValue: document.getElementById("frameValue"),
  motionThreshold: document.getElementById("motionThreshold"),
  detectionModel: document.getElementById("detectionModel"),
  confidenceThreshold: document.getElementById("confidenceThreshold"),
  classOptions: document.getElementById("classOptions"),
  customClass: document.getElementById("customClass"),
  addClassButton: document.getElementById("addClassButton"),
  customClassList: document.getElementById("customClassList"),
  manualGeoFields: document.getElementById("manualGeoFields"),
  latitude: document.getElementById("latitude"),
  longitude: document.getElementById("longitude"),
  uploadStatus: document.getElementById("uploadStatus"),
  configPreview: document.getElementById("configPreview"),
  jobList: document.getElementById("jobList"),
  detectionList: document.getElementById("detectionList"),
  heatmapSummary: document.getElementById("heatmapSummary"),
  refreshHeatmap: document.getElementById("refreshHeatmap"),
  heatmapObjectFilter: document.getElementById("heatmapObjectFilter"),
  heatmapStart: document.getElementById("heatmapStart"),
  heatmapEnd: document.getElementById("heatmapEnd"),
  violationList: document.getElementById("violationList"),
  
};

function getGeoMode() {
  return document.querySelector('input[name="geoMode"]:checked').value;
}

function getSelectedClasses() {
  const selected = Array.from(elements.classOptions.querySelectorAll('input[type="checkbox"]:checked'))
    .map((input) => input.value);
  return [...selected, ...state.customClasses];
}

function buildConfig() {
  const mode = getGeoMode();
  const violations = [];
  if (document.getElementById("vTripleRiding").checked) {
    violations.push("triple_riding");
  }
  if (document.getElementById("vWrongWay").checked) {
    violations.push("wrong_way");
  }
  if (document.getElementById("vOverspeed").checked)
    violations.push("overspeed");

  return {
    audio_removal: elements.audioRemoval.checked,
    face_blur: {
      enabled: elements.faceBlurEnabled.checked,
      method: elements.faceBlurMethod.value,
      intensity: Number(elements.faceBlurIntensity.value),
    },
    frame_extraction: {
      method: elements.frameMethod.value,
      value: Number(elements.frameValue.value),
      motion_threshold: Number(elements.motionThreshold.value),
    },
    object_detection: {
      model: elements.detectionModel.value.trim() || "yolov8n",
      classes: getSelectedClasses(),
      confidence_threshold: Number(elements.confidenceThreshold.value),
    },
    geo_tagging: {
      mode,
      latitude: mode === "manual" ? Number(elements.latitude.value) : null,
      longitude: mode === "manual" ? Number(elements.longitude.value) : null,
    },
    violation_detection: {
      taskkillenabled: violations.length > 0,
      list_violations: violations
    }
  };
}

function renderConfigPreview() {
  elements.configPreview.textContent = JSON.stringify(buildConfig(), null, 2);
}

function renderCustomClasses() {
  elements.customClassList.innerHTML = "";
  state.customClasses.forEach((name) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip secondary";
    chip.textContent = `${name} x`;
    chip.addEventListener("click", () => {
      state.customClasses = state.customClasses.filter((item) => item !== name);
      renderCustomClasses();
      renderConfigPreview();
    });
    elements.customClassList.appendChild(chip);
  });
}

function addCustomClass() {
  const candidate = elements.customClass.value.trim().toLowerCase();
  if (!candidate || state.customClasses.includes(candidate) || getSelectedClasses().includes(candidate)) {
    elements.customClass.value = "";
    return;
  }
  state.customClasses.push(candidate);
  elements.customClass.value = "";
  renderCustomClasses();
  renderConfigPreview();
}

async function uploadVideos(event) {
  event.preventDefault();
  const files = Array.from(elements.videoFiles.files);
  if (!files.length) {
    elements.uploadStatus.textContent = "Choose at least one video file.";
    return;
  }

  const config = buildConfig();
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("config_json", JSON.stringify(config));

  elements.uploadStatus.textContent = "Uploading videos...";
  const response = await fetch("/upload", { method: "POST", body: formData });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Upload failed." }));
    elements.uploadStatus.textContent = error.detail || "Upload failed.";
    return;
  }

  const payload = await response.json();
  state.jobs = payload.items.map((item) => ({ ...item, processing: false, stages: null, results: null }));
  elements.uploadStatus.textContent = `Uploaded ${payload.items.length} video(s).`;
  renderJobs();
}

async function processVideo(videoId) {
  updateJob(videoId, { processing: true });
  renderJobs();
  const selectedViolations = [];

if (document.getElementById("vTripleRiding").checked) {
  selectedViolations.push("triple_riding");
}
if (document.getElementById("vWrongWay").checked) {
  selectedViolations.push("wrong_way");
}
if (document.getElementById("vOverspeed").checked) {
    selectedViolations.push("overspeed");
}

const response = await fetch(`/process/${videoId}`, {
  method: "POST",headers: {"Content-Type": "application/json"},body: JSON.stringify({violations: selectedViolations})
});
  const payload = await response.json();
  updateJob(videoId, {
    processing: false,
    status: payload.status,
    stages: payload.stages,
  });
  await loadResults(videoId);
  renderJobs();
  await refreshHeatmap();
}

async function loadResults(videoId) {
  const response = await fetch(`/results/${videoId}`);
  if (!response.ok) {
    return;
  }
  const payload = await response.json();
  updateJob(videoId, { results: payload, status: payload.status });
  renderDetections();
}

function updateJob(videoId, partial) {
  state.jobs = state.jobs.map((job) => (job.video_id === videoId ? { ...job, ...partial } : job));
}

function renderJobs() {
  elements.jobList.innerHTML = "";
  if (!state.jobs.length) {
    elements.jobList.innerHTML = "<p class='summary'>Uploaded jobs will appear here.</p>";
    return;
  }

  state.jobs.forEach((job) => {
    const card = document.createElement("article");
    card.className = "job-card";
    const stageSummary = job.stages
      ? `<pre>${JSON.stringify(job.stages, null, 2)}</pre>`
      : "<p class='summary'>Waiting to process.</p>";

    card.innerHTML = `
      <header>
        <div>
          <strong>${job.filename}</strong>
          <div class="summary">Video ID ${job.video_id} • ${job.status}</div>
        </div>
        <button type="button" ${job.processing ? "disabled" : ""}>${job.processing ? "Processing..." : "Process"}</button>
      </header>
      ${stageSummary}
    `;

    card.querySelector("button").addEventListener("click", () => processVideo(job.video_id));
    elements.jobList.appendChild(card);
  });
}

function renderDetections() {
  const allDetections = state.jobs.flatMap((job) => job.results?.detections || []);
  elements.detectionList.innerHTML = "";
  if (!allDetections.length) {
    elements.detectionList.innerHTML = "<p class='summary'>Processed detections will appear here.</p>";
    return;
  }

  allDetections.slice(0, 100).forEach((detection) => {
    const card = document.createElement("article");
    card.className = "detection-card";
    card.innerHTML = `
      <header>
        <strong>${detection.object_class}</strong>
        <span>${(detection.confidence * 100).toFixed(1)}%</span>
      </header>
      <div class="summary">
        Frame ${detection.frame_index} • ${detection.timestamp_seconds.toFixed(2)}s
      </div>
      <div class="summary">
        ${detection.latitude ?? "n/a"}, ${detection.longitude ?? "n/a"}
      </div>
    `;
    elements.detectionList.appendChild(card);
  });
}

function initMap() {
  state.map = L.map("map").setView([20.2961, 85.8245], 13);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  state.markersLayer = L.layerGroup().addTo(state.map);
}

async function refreshHeatmap() {
  const params = new URLSearchParams();
  if (elements.heatmapObjectFilter.value) {
    params.set("object_type", elements.heatmapObjectFilter.value);
  }
  if (elements.heatmapStart.value) {
    params.set("start_time", elements.heatmapStart.value);
  }
  if (elements.heatmapEnd.value) {
    params.set("end_time", elements.heatmapEnd.value);
  }

  const response = await fetch(`/heatmap?${params.toString()}`);
  if (!response.ok) {
    elements.heatmapSummary.textContent = "Unable to load heatmap data.";
    return;
  }

  const points = await response.json();
  elements.heatmapSummary.textContent = `${points.length} geo-tagged detections currently match the active filters.`;

  state.markersLayer.clearLayers();
  if (state.heatLayer) {
    state.map.removeLayer(state.heatLayer);
  }

  const heatmapData = points.map((point) => {
    const marker = L.marker([point.latitude, point.longitude]);
    marker.bindPopup(`${point.object_class} • confidence ${point.intensity.toFixed(2)} • video ${point.video_id}`);
    marker.addTo(state.markersLayer);
    return [point.latitude, point.longitude, point.intensity];
  });

  if (heatmapData.length) {
    state.heatLayer = L.heatLayer(heatmapData, {
      radius: 28,
      blur: 18,
      maxZoom: 17,
      gradient: { 0.1: "#f1c27d", 0.4: "#e58f45", 0.7: "#c94c1a", 1.0: "#821f04" },
    }).addTo(state.map);
    state.map.fitBounds(L.latLngBounds(heatmapData.map(([lat, lon]) => [lat, lon])), { padding: [24, 24] });
  }
}

function toggleManualGeoFields() {
  const manual = getGeoMode() === "manual";
  elements.manualGeoFields.classList.toggle("hidden", !manual);
  renderConfigPreview();
}

elements.faceBlurIntensity.addEventListener("input", () => {
  elements.faceBlurIntensityValue.textContent = elements.faceBlurIntensity.value;
  renderConfigPreview();
});

[
  elements.audioRemoval,
  elements.faceBlurEnabled,
  elements.faceBlurMethod,
  elements.frameMethod,
  elements.frameValue,
  elements.motionThreshold,
  elements.detectionModel,
  elements.confidenceThreshold,
  elements.latitude,
  elements.longitude,
].forEach((element) => element.addEventListener("input", renderConfigPreview));

elements.classOptions.addEventListener("change", renderConfigPreview);
document.querySelectorAll('input[name="geoMode"]').forEach((radio) => radio.addEventListener("change", toggleManualGeoFields));
elements.addClassButton.addEventListener("click", addCustomClass);
elements.customClass.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addCustomClass();
  }
});
elements.uploadForm.addEventListener("submit", uploadVideos);
elements.refreshHeatmap.addEventListener("click", refreshHeatmap);
elements.heatmapObjectFilter.addEventListener("change", refreshHeatmap);
elements.heatmapStart.addEventListener("input", refreshHeatmap);
elements.heatmapEnd.addEventListener("input", refreshHeatmap);

document.getElementById("vTripleRiding")
  .addEventListener("change", renderConfigPreview);

document.getElementById("vWrongWay")
  .addEventListener("change", renderConfigPreview);

document.getElementById("vOverspeed")
  .addEventListener("change", renderConfigPreview);

renderConfigPreview();
renderJobs();
renderDetections();
toggleManualGeoFields();
initMap();
refreshHeatmap();
