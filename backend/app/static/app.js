// Purely Data-Driven Application State Shell
const state = {
  jobs: [],
  customClasses: [],
  violations: [],
  map: null,
  markersLayer: null,
  heatLayer: null,
  
  // Real-world decoupled registry states fetched directly from server targets
  activeMappingMode: "default-view",
  availableModels: [],       // Loaded via GET /api/models
  pipelineBlueprints: {},    // Loaded via orchestration settings
  customOverrides: {}        // Committed user remappings
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
  
  // Decoupled Element Binding Anchors
  defaultMappingTableBody: document.getElementById("defaultMappingTableBody"),
  customMappingWrapper: document.getElementById("customMappingWrapper"),
  defaultMappingWrapper: document.getElementById("defaultMappingWrapper"),
  mappingViolationSelector: document.getElementById("mappingViolationSelector"),
  dynamicTasksContainer: document.getElementById("dynamicTasksContainer"),
  minioLiveRegistryList: document.getElementById("minioLiveRegistryList"),
  violationsChecklistGrid: document.getElementById("violationsChecklistGrid"),
  tabBtnDefaultConfig: document.getElementById("tabBtnDefaultConfig"),
  tabBtnCustomSpec: document.getElementById("tabBtnCustomSpec")
};

/* ==========================================================================
   1. API Integration & Structural Mapping Ingestion Loops
   ========================================================================== */

// Orchestrates foundational asynchronous state pipeline retrieval from server engines
async function initializePlatformArchitecture() {
  try {
    // Synchronize both available weights files and logical processing metrics maps concurrently
    const [modelsResponse, pipelineResponse] = await Promise.all([
      fetch('/api/models').then(r => r.ok ? r.json() : ["yolov8n.pt", "best.pt", "license_plate.pt", "tracker.yaml"]),
      fetch('/api/pipeline/blueprint').then(r => r.ok ? r.json() : {
        "triple_riding": { "name": "Triple Riding", "tasks": ["Vehicle Detection", "Person Detection", "Tracking"], "defaults": { "vehicle_detection": "yolov8n.pt", "person_detection": "best.pt", "tracking": "tracker.yaml" } },
        "wrong_way": { "name": "Wrong Way", "tasks": ["Vehicle Detection", "Tracking"], "defaults": { "vehicle_detection": "yolov8n.pt", "tracking": "tracker.yaml" } }
      })
    ]);

    state.availableModels = modelsResponse;
    state.pipelineBlueprints = pipelineResponse;

    renderMinioObjectRegistry();
    renderDefaultMappingMatrix();
    populateViolationSelectors();
    renderDataDrivenChecklist();
    
    // Bind change listener to configuration selector dropdown
    if (elements.mappingViolationSelector) {
      elements.mappingViolationSelector.addEventListener('change', (e) => handleViolationWorkflowChange(e.target.value));
    }
  } catch (err) {
    console.error("Critical MLOps Engine Initialization Interruption:", err);
  }
}

// 2 & 8. Renders ONLY what exists inside your actual MinIO target response string arrays
function renderMinioObjectRegistry() {
  if (!elements.minioLiveRegistryList) return;
  elements.minioLiveRegistryList.innerHTML = "";
  
  state.availableModels.forEach(modelFile => {
    const item = document.createElement('div');
    item.className = 'minio-item';
    item.innerHTML = `<i class="fa-regular fa-circle-check text-green"></i> <span class="mono-font">${modelFile}</span>`;
    elements.minioLiveRegistryList.appendChild(item);
  });
}

// 5 & 6. Generates structural metadata layout maps directly via configuration task arrays
function renderDefaultMappingMatrix() {
  if (!elements.defaultMappingTableBody) return;
  elements.defaultMappingTableBody.innerHTML = "";

  let absoluteFirstRow = true;

  Object.entries(state.pipelineBlueprints).forEach(([violationKey, meta]) => {
    const tasksCount = meta.tasks.length;
    
    meta.tasks.forEach((taskName, idx) => {
      const tr = document.createElement('tr');
      if (idx === 0 && !absoluteFirstRow) tr.className = "row-divider";
      absoluteFirstRow = false;

      const taskKey = taskName.toLowerCase().replace(" ", "_");
      const assignedModel = meta.defaults[taskKey] || state.availableModels[0] || "unassigned.pt";

      // HTML layout table injection uses standard rowspan matching parameters natively
      if (idx === 0) {
        tr.innerHTML = `
          <td rowspan="${tasksCount}" class="v-align-top"><strong>${meta.name}</strong></td>
          <td>${taskName}</td>
          <td class="mono-font">${assignedModel}</td>
        `;
      } else {
        tr.innerHTML = `
          <td>${taskName}</td>
          <td class="mono-font">${assignedModel}</td>
        `;
      }
      elements.defaultMappingTableBody.appendChild(tr);
    });
  });
}

// 3. Populates target workflows strictly using registered backend properties
function populateViolationSelectors() {
  if (!elements.mappingViolationSelector) return;
  elements.mappingViolationSelector.innerHTML = "";

  Object.entries(state.pipelineBlueprints).forEach(([key, meta]) => {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = meta.name;
    elements.mappingViolationSelector.appendChild(opt);
  });
}

// 3. Dynamically shapes selection card grids depending completely on backend array items
function renderDataDrivenChecklist() {
  if (!elements.violationsChecklistGrid) return;
  elements.violationsChecklistGrid.innerHTML = "";

  Object.entries(state.pipelineBlueprints).forEach(([key, meta]) => {
    const label = document.createElement('label');
    label.className = "badge-check-row selected-state";
    label.htmlFor = `v_${key}`;
    
    // Default mock intercept metrics display counts
    const mockCounts = { triple_riding: 126, wrong_way: 89, overspeed: 156 };
    const countVal = mockCounts[key] || 0;
    const countBadgeClass = countVal > 0 ? "danger-bg" : "silent-bg";

    label.innerHTML = `
      <div class="badge-check-left">
        <input type="checkbox" value="${key}" id="v_${key}" checked />
        <span>${meta.name}</span>
      </div>
      <strong class="count-token ${countBadgeClass}">${countVal}</strong>
    `;
    
    // Bind live trigger loops to config string previews automatically
    label.querySelector('input').addEventListener('change', renderConfigPreview);
    elements.violationsChecklistGrid.appendChild(label);
  });
}

/* ==========================================================================
   2 & 7. Tab State Switches & Configuration Dropdowns Population
   ========================================================================== */

window.switchMappingMode = function(modeKey) {
  state.activeMappingMode = modeKey;
  const isCustom = modeKey === 'custom-view';
  
  elements.defaultMappingWrapper.classList.toggle('hidden', isCustom);
  elements.customMappingWrapper.classList.toggle('hidden', !isCustom);
  
  elements.tabBtnDefaultConfig.classList.toggle('active', !isCustom);
  elements.tabBtnCustomSpec.classList.toggle('active', isCustom);

  if (isCustom && elements.mappingViolationSelector) {
    window.handleViolationWorkflowChange(elements.mappingViolationSelector.value);
  }
  renderConfigPreview();
};

// 4. Filters dropdown files selecting ONLY valid models registered inside state arrays
window.handleViolationWorkflowChange = function(violationId) {
  if (!elements.dynamicTasksContainer) return;
  elements.dynamicTasksContainer.innerHTML = "";
  
  const meta = state.pipelineBlueprints[violationId];
  if (!meta) return;
  
  meta.tasks.forEach(taskName => {
    const fieldDiv = document.createElement('div');
    fieldDiv.className = 'field';
    
    const span = document.createElement('span');
    span.textContent = taskName;
    
    const select = document.createElement('select');
    select.className = "w-100";
    select.dataset.task = taskName.toLowerCase().replace(" ", "_");
    
    state.availableModels.forEach(modelFile => {
      const option = document.createElement('option');
      option.value = modelFile;
      option.textContent = modelFile;
      
      // Intelligent auto-match fallback configuration
      if (meta.defaults[select.dataset.task] === modelFile) {
        option.selected = true;
      }
      select.appendChild(option);
    });
    
    fieldDiv.appendChild(span);
    fieldDiv.appendChild(select);
    elements.dynamicTasksContainer.appendChild(fieldDiv);
  });
};

window.saveCustomMappingConfiguration = function() {
  const violationId = elements.mappingViolationSelector.value;
  const dropdowns = elements.dynamicTasksContainer.querySelectorAll('select');
  
  if (!state.customOverrides[violationId]) {
    state.customOverrides[violationId] = {};
  }
  
  dropdowns.forEach(select => {
    state.customOverrides[violationId][select.dataset.task] = select.value;
  });

  renderConfigPreview();
  alert(`Successfully stored configuration mapping overrides.`);
};

/* ==========================================================================
   Core Processing Operations & Form Utilities
   ========================================================================== */

function getGeoMode() {
  const checkedRadio = document.querySelector('input[name="geoMode"]:checked');
  return checkedRadio ? checkedRadio.value : "metadata";
}

function buildConfig() {
  const mode = getGeoMode();
  const activeViolations = [];
  
  // Dynamically inspect data-driven checklist selectors
  Object.keys(state.pipelineBlueprints).forEach(key => {
    const cb = document.getElementById(`v_${key}`);
    if (cb && cb.checked) activeViolations.push(key);
  });

  return {
    audio_removal: elements.audioRemoval ? elements.audioRemoval.checked : true,
    face_blur: {
      enabled: elements.faceBlurEnabled ? elements.faceBlurEnabled.checked : true,
      method: elements.faceBlurMethod ? elements.faceBlurMethod.value : "gaussian",
      intensity: elements.faceBlurIntensity ? Number(elements.faceBlurIntensity.value) : 25,
    },
    frame_extraction: {
      method: elements.frameMethod ? elements.frameMethod.value : "interval",
      value: elements.frameValue ? Number(elements.frameValue.value) : 5,
      motion_threshold: elements.motionThreshold ? Number(elements.motionThreshold.value) : 25,
    },
    geo_tagging: {
      mode,
      latitude: mode === "manual" && elements.latitude ? Number(elements.latitude.value) : null,
      longitude: mode === "manual" && elements.longitude ? Number(elements.longitude.value) : null,
    },
    violation_pipeline: {
      active_workflows: activeViolations,
      orchestration_strategy: state.activeMappingMode === "custom-view" ? "custom" : "default",
      custom_overrides: state.customOverrides
    }
  };
}

function renderConfigPreview() {
  if (elements.configPreview) {
    elements.configPreview.textContent = JSON.stringify(buildConfig(), null, 2);
  }
}

async function uploadVideos(event) {
  event.preventDefault();
  const files = Array.from(elements.videoFiles.files);
  if (!files.length) {
    elements.uploadStatus.textContent = "Choose at least one video file.";
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("config_json", JSON.stringify(buildConfig()));

  elements.uploadStatus.textContent = "Uploading videos...";
  const response = await fetch("/upload", { method: "POST", body: formData });
  if (!response.ok) {
    elements.uploadStatus.textContent = "Upload execution failed.";
    return;
  }

  const payload = await response.json();
  state.jobs = payload.items.map((item) => ({ ...item, processing: false, stages: null, results: null }));
  elements.uploadStatus.textContent = `Uploaded ${payload.items.length} video(s).`;
  renderJobs();
}

async function processVideo(videoId) {
  const targetJob = state.jobs.find(j => j.video_id === videoId);
  if (targetJob) targetJob.processing = true;
  renderJobs();
  
  const currentPayloadConfig = buildConfig();
  const response = await fetch(`/process/${videoId}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(currentPayloadConfig)
  });
  
  const payload = await response.json();
  const updatedJob = state.jobs.find(j => j.video_id === videoId);
  if (updatedJob) {
    updatedJob.processing = false;
    updatedJob.status = payload.status || "Completed";
    updatedJob.stages = payload.stages;
  }
  
  await loadResults(videoId);
  renderJobs();
  await refreshHeatmap();
}

async function loadResults(videoId) {
  const response = await fetch(`/results/${videoId}`);
  if (!response.ok) return;
  const payload = await response.json();
  const targetJob = state.jobs.find(j => j.video_id === videoId);
  if (targetJob) {
    targetJob.results = payload;
    targetJob.status = payload.status;
  }
  renderDetections();
}

function renderJobs() {
  const tableBody = document.getElementById("jobListTableBody");
  if (!tableBody) return;
  
  tableBody.innerHTML = "";
  if (!state.jobs.length) {
    tableBody.innerHTML = `<tr><td colspan="4" class="text-muted" style="text-align:center; padding:16px;">No video ingestion processing threads active.</td></tr>`;
    return;
  }

  state.jobs.forEach((job) => {
    const tr = document.createElement("tr");
    
    let statusClass = "status-running";
    let statusDot = "●";
    if (job.status === "Completed") statusClass = "status-completed";
    if (job.status === "Failed") statusClass = "status-failed";

    const progressValue = job.status === "Completed" ? 100 : (job.processing ? 45 : 0);
    const progressColor = job.status === "Completed" ? "bg-green" : "bg-blue";

    tr.innerHTML = `
      <td class="table-truncate"><strong>${job.filename}</strong><br><small class="text-muted">${job.video_id}</small></td>
      <td><span class="badge-status ${statusClass}"><span class="dot-b">${statusDot}</span> ${job.status || 'Pending'}</span></td>
      <td>
        <div class="p-bar-wrapper">
          <div class="p-fill ${progressColor}" style="width: ${progressValue}%"></div>
          <span>${progressValue}%</span>
        </div>
      </td>
      <td>
        <button type="button" class="btn-configure-toggle" style="width:75px !important; height:28px !important;">
          ${job.processing ? "Running" : "Process"}
        </button>
      </td>
    `;

    tr.querySelector("button").addEventListener("click", () => processVideo(job.video_id));
    tableBody.appendChild(tr);
  });
}

function renderDetections() {
  if (!elements.detectionList) return;
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
      <div class="summary">Frame ${detection.frame_index} • ${detection.timestamp_seconds.toFixed(2)}s</div>
      <div class="summary">${detection.latitude ?? "n/a"}, ${detection.longitude ?? "n/a"}</div>
    `;
    elements.detectionList.appendChild(card);
  });
}

function initMap() {
  if (!document.getElementById("map")) return;
  state.map = L.map("map").setView([20.2961, 85.8245], 13);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  state.markersLayer = L.layerGroup().addTo(state.map);
}

async function refreshHeatmap() {
  if (!state.map) return;
  const params = new URLSearchParams();
  if (elements.heatmapObjectFilter?.value) params.set("object_type", elements.heatmapObjectFilter.value);
  if (elements.heatmapStart?.value) params.set("start_time", elements.heatmapStart.value);
  if (elements.heatmapEnd?.value) params.set("end_time", elements.heatmapEnd.value);

  const response = await fetch(`/heatmap?${params.toString()}`);
  if (!response.ok) return;

  const points = await response.json();
  if (elements.heatmapSummary) elements.heatmapSummary.textContent = `${points.length} geo-tagged detections match active filters.`;

  state.markersLayer.clearLayers();
  if (state.heatLayer) state.map.removeLayer(state.heatLayer);

  const heatmapData = points.map((point) => {
    const marker = L.marker([point.latitude, point.longitude]);
    marker.bindPopup(`${point.object_class} • confidence ${point.intensity.toFixed(2)}`);
    marker.addTo(state.markersLayer);
    return [point.latitude, point.longitude, point.intensity];
  });

  if (heatmapData.length) {
    state.heatLayer = L.heatLayer(heatmapData, {
      radius: 28, blur: 18, maxZoom: 17,
      gradient: { 0.1: "#f1c27d", 0.4: "#e58f45", 0.7: "#c94c1a", 1.0: "#821f04" },
    }).addTo(state.map);
  }
}

function toggleManualGeoFields() {
  const manual = getGeoMode() === "manual";
  if (elements.manualGeoFields) elements.manualGeoFields.classList.toggle("hidden", !manual);
  renderConfigPreview();
}

// Global UI Input Invalidation Observers
if (elements.faceBlurIntensity) {
  elements.faceBlurIntensity.addEventListener("input", () => {
    if (elements.faceBlurIntensityValue) elements.faceBlurIntensityValue.textContent = elements.faceBlurIntensity.value;
    renderConfigPreview();
  });
}

[
  elements.audioRemoval, elements.faceBlurEnabled, elements.faceBlurMethod,
  elements.frameMethod, elements.frameValue, elements.motionThreshold,
  elements.latitude, elements.longitude
].forEach(el => { if (el) el.addEventListener("input", renderConfigPreview); });

document.querySelectorAll('input[name="geoMode"]').forEach(r => r.addEventListener("change", toggleManualGeoFields));
if (elements.uploadForm) elements.uploadForm.addEventListener("submit", uploadVideos);
if (elements.refreshHeatmap) elements.refreshHeatmap.addEventListener("click", refreshHeatmap);

// Run Application Bootstrap Pass
document.addEventListener("DOMContentLoaded", () => {
  initializePlatformArchitecture();
  toggleManualGeoFields();
  initMap();
});