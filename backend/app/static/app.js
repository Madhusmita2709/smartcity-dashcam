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
  availableModels: [],       // Loaded dynamically via Step 1 (GET /api/models)
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
    const [modelsResponse, mappingResponse] = await Promise.all([
      fetch("/api/models"),
      fetch("/api/default-mapping")
    ]);

    if (!modelsResponse.ok || !mappingResponse.ok) {
      throw new Error("Initialization failed.");
    }

    const modelsData = await modelsResponse.json();
    const mappingData = await mappingResponse.json();

    state.availableModels = modelsData.models || modelsData;
    state.pipelineBlueprints = mappingData;

    renderMinioObjectRegistry();
    renderDefaultMappingMatrix();
    populateViolationSelectors();
    renderDataDrivenChecklist();

    if (elements.mappingViolationSelector) {
      elements.mappingViolationSelector.addEventListener(
        "change",
        (e) => handleViolationWorkflowChange(e.target.value)
      );

      handleViolationWorkflowChange(
        elements.mappingViolationSelector.value
      );
    }

  } catch (err) {
    console.error(err);
    if (elements.uploadStatus) {
      elements.uploadStatus.textContent = "Unable to connect to backend.";
    }
  }
}

// Renders ONLY what exists inside your actual weights storage directory context
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

// Generates structural metadata layout maps directly via configuration task arrays
function renderDefaultMappingMatrix() {
  if (!elements.defaultMappingTableBody) return;
  elements.defaultMappingTableBody.innerHTML = "";

  let first = true;

  Object.values(state.pipelineBlueprints).forEach(meta => {
    // Crucial: The function signature must accept 'task'
    meta.tasks.forEach((task, index) => {
      const tr = document.createElement("tr");

      if (index === 0 && !first) tr.className = "row-divider";
      first = false;

      const assignedModel = task.type === "execution_module"
        ? (task.default.toLowerCase() === "bytetrack" ? "ByteTrack Tracker" : task.default)
        : task.default;

      if (index === 0) {
        tr.innerHTML = `
        <td rowspan="${meta.tasks.length}">
            <strong>${meta.name}</strong>
        </td>
        <td>${task.name}</td>
        <td class="mono-font">${assignedModel}</td>
        `;
      } else {
        tr.innerHTML = `
        <td>${task.name}</td>
        <td class="mono-font">${assignedModel}</td>
        `;
      }

      elements.defaultMappingTableBody.appendChild(tr);
    });
  });
}
// Populates target workflows strictly using registered backend properties
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

// Dynamically shapes selection card grids depending completely on backend items
function renderDataDrivenChecklist() {
  if (!elements.violationsChecklistGrid) return;
  elements.violationsChecklistGrid.innerHTML = "";

  Object.entries(state.pipelineBlueprints).forEach(([key, meta]) => {
    const label = document.createElement('label');
    label.className = "badge-check-row selected-state";
    label.htmlFor = `v_${key}`;
    
    label.innerHTML = `
      <div class="badge-check-left">
        <input type="checkbox" value="${key}" id="v_${key}" checked />
        <span>${meta.name}</span>
      </div>
    `;
    
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

window.handleViolationWorkflowChange = function(violationId) {
    if (!elements.dynamicTasksContainer) return;
    elements.dynamicTasksContainer.innerHTML = "";

    const meta = state.pipelineBlueprints[violationId];
    if (!meta) return;

    meta.tasks.forEach(task => {
        const wrapper = document.createElement("div");
        wrapper.className = "field";

        const label = document.createElement("span");
        label.textContent = task.name;

        const select = document.createElement("select");
        select.className = "w-100";
        select.dataset.task = task.id;

        if (task.type === "execution_module") {
            const option = document.createElement("option");
            option.value = task.default;
            option.textContent = task.default.toLowerCase() === "bytetrack" ? "ByteTrack Tracker" : `${task.default} Module`;
            option.selected = true;
            select.appendChild(option);
        } else {
            state.availableModels.forEach(model => {
                const option = document.createElement("option");
                option.value = model;
                option.textContent = model;
                if (model === task.default) option.selected = true;
                select.appendChild(option);
            });
        }

        wrapper.appendChild(label);
        wrapper.appendChild(select);
        elements.dynamicTasksContainer.appendChild(wrapper);
    });
}

window.saveCustomMappingConfiguration = async function() {
    const violationId = elements.mappingViolationSelector.value;
    const overrides = {};

    elements.dynamicTasksContainer
        .querySelectorAll("select")
        .forEach(select => {
            overrides[select.dataset.task] = select.value;
        });

    const payload = {
        violation: violationId,
        overrides: overrides
    };

    try {
        const response = await fetch("/api/custom-mapping", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error();

        const updated = await fetch("/api/default-mapping");
        state.pipelineBlueprints = await updated.json();

        renderDefaultMappingMatrix();
        renderConfigPreview();
        handleViolationWorkflowChange(violationId);

        elements.uploadStatus.textContent = "✓ Configuration saved successfully.";
    } catch(err) {
        console.error(err);
        elements.uploadStatus.textContent = "❌ Unable to save configuration.";
    }
}

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