/**
 * ==========================================================================
 * MODULE 1: GLOBAL STATE, DOM ELEMENTS, DICTIONARY & HELPER PLUGINS
 * ==========================================================================
 */

// 1. DATA-DRIVEN UNIFIED APPLICATION STATE MATRIX
const state = {
  jobs: [],
  customClasses: [],
  violations: [],
  defaultMapping: {},
  customMapping: {},
  availableModels: [],
  map: null,
  markersLayer: null,
  heatLayer: null,
  
  
  // Your Original Core Engine Architecture Mappings
  activeMappingMode: "default-view",
  availableModels: [],       // Dynamically populated via GET /api/models
  pipelineBlueprints: {},    // Dynamically populated via GET /api/default-mapping
  customOverrides: {},       // Committed custom orchestrations matrix
  
  // Teammate's Spatial Analytics Extensions
  timelinePath: null, 
  currentDate: null,
  currentVideoId: null,
  activeRouteController: null, // AbortController instance preventing rapid-click routing races
  activeTimelineController: null
};

// 2. CONSOLIDATED GLOBAL RECURSIVE DOM SELECTORS
const elements = {
  // Your Core Dashboard Interface Anchors
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
  
  // Decoupled Structural Registration Multi-Variant Anchor Targets
  defaultMappingTableBody: document.getElementById("defaultMappingTableBody"),
  customMappingWrapper: document.getElementById("customMappingWrapper"),
  defaultMappingWrapper: document.getElementById("defaultMappingWrapper"),
  customMappingAccordion: document.getElementById("customMappingAccordion"),
  availableModels: document.getElementById("availableModels"),
  violationsChecklistGrid: document.getElementById("violationsChecklistGrid"),
  tabBtnDefaultConfig: document.getElementById("tabBtnDefaultConfig"),
  tabBtnCustomSpec: document.getElementById("tabBtnCustomSpec"),

  // Teammate's Historical Analytical Filter Visualizer Bindings
  archiveDayFilter: document.getElementById("archiveDayFilter"),
  archiveMonthFilter: document.getElementById("archiveMonthFilter"),
  archiveYearFilter: document.getElementById("archiveYearFilter"),
  dashboardVideoSelect: document.getElementById("dashboardVideoSelect"),
  dashboardViolationFilter: document.getElementById("dashboardViolationFilter"),
  violationList: document.getElementById("violationList"),
  checkViolationsBtn: document.getElementById("checkViolationsBtn"),
  vNoHelmet: document.getElementById("vNoHelmet"),
  vPhoneUsage: document.getElementById("vPhoneUsage")
};

// ==========================================================================
// CENTRALIZED DATA TRANSLATION UTILITY MODULE (FINAL APPROVED PASS)
// ==========================================================================
const Dictionary = {
  aliases: {
    no_helmet: ["helmet", "nh", "no_helmet"],
    phone_usage: ["phone", "mobile", "phone_usage"],
    wrong_way: ["wrong", "wrong_way"],
    triple_riding: ["triple", "triple_riding"],
    no_number_plate: ["number_plate", "plate", "no_number_plate"],
    overspeed: ["overspeed", "speed", "overspeeding"],
    lane_change: ["lane", "lane_change", "lane_changing"]
  },

  getViolationLabel(type = "") {
    const norm = normalizeViolationType(type);
    const tokens = norm.split("_");
    
    // Fixed: Strict token boundary verification to prevent substring collisions
    const matchedKey = Object.keys(this.aliases).find(k => 
      this.aliases[k].some(a => tokens.includes(a) || norm === a)
    );

    return matchedKey 
      ? matchedKey.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
      : type.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
  },

  matchesFilter(pointType, selectedFilter) {
    if (!selectedFilter || selectedFilter === "all_violations" || selectedFilter === "all") return true;
    
    const pointNorm = normalizeViolationType(pointType);
    const filterNorm = normalizeViolationType(selectedFilter);

    const targetKey = Object.keys(this.aliases).find(key => key === filterNorm || this.aliases[key].includes(filterNorm));
    if (!targetKey) return pointNorm === filterNorm;

    const pointTokens = pointNorm.split("_");
    return this.aliases[targetKey].some(a => 
      pointTokens.includes(a) || pointNorm === a
    );
  },
  getViolationColorClass(type = ""){
    const norm = normalizeViolationType(type);

    if (
        norm.includes("helmet") ||
        norm.includes("phone") ||
        norm.includes("wrong")
    ){
        return "text-accent-strong";
    }

    return "text-accent-muted";
  },
  createMapPopupHTML(title, actualLabel, confidenceValue, videoId, plate = null, imgUrl = null) {
    // Fixed: Defensive wrapper protecting against empty/null labels
    const labelString = String(actualLabel ?? "").toLowerCase();
    const themeClass = ["helmet", "phone", "wrong"].some(keyword => labelString.includes(keyword))
      ? "popup-theme-accent"
      : "popup-theme-secondary";

    return `
      <div class="map-popup-container">
        <strong class="map-popup-title ${themeClass}">${title}</strong>
        <hr class="map-popup-divider"/>
        <div class="map-popup-body">
          ${plate ? `<b>Plate Reference:</b> <span class="mono-font">${plate}</span><br/>` : ''}
          <b>Violation Type:</b> ${this.getViolationLabel(actualLabel)}<br/>
          <b>Confidence:</b> ${(Number(confidenceValue ?? 0) * 100).toFixed(1)}%<br/>
          <b>Video Track ID:</b> ${videoId}
        </div>
        ${imgUrl ? `<img src="${imgUrl}" class="map-popup-image" alt="Evidence Layer Data Match" />` : ''}
      </div>
    `;
  }
};
// ==========================================================================
// DECOUPLED TELEMETRY & TEXT PARSING HELPERS
// ==========================================================================

function normalizeViolationType(type = "") {
    return String(type)
        .toLowerCase()
        .trim()
        .replace(/\s+/g, "_");
}

// 4. CORE RECURSIVE METADATA INTERPOLATION HELPERS
function findNearestRoutePoint(routeArray, timestampSeconds) {
    if (!routeArray || routeArray.length === 0) return null;

    const ts = Number(timestampSeconds ?? 0);

    const index = Math.max(0,Math.min(Math.round(ts), routeArray.length - 1));

    return routeArray[index] || null;
}

/* ==========================================================================
   2. Engine Tab State Switches & Configuration Dropdowns Population
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

function renderCustomMappingAccordion() {
  if (!elements.customMappingAccordion) return;
  elements.customMappingAccordion.innerHTML = "";

  Object.entries(state.pipelineBlueprints).forEach(([violationId, meta]) => {
    const details = document.createElement("details");
    details.open = true;

    const summary = document.createElement("summary");
    summary.innerHTML = `<strong>${meta.name}</strong>`;
    details.appendChild(summary);

    meta.tasks.forEach(task => {
      const wrapper = document.createElement("div");
      wrapper.className = "field";
      wrapper.style.marginTop = "10px";

      const label = document.createElement("span");
      label.textContent = task.name;

      const select = document.createElement("select");
      select.dataset.violation = violationId;
      select.dataset.task = task.id;

      if (task.type === "execution_module") {
        const option = document.createElement("option");
        option.value = task.default;
        option.textContent = task.default;
        select.appendChild(option);
      } else {
        state.availableModels.forEach(model => {
          const option = document.createElement("option");
          option.value = model;
          option.textContent = model;

          const saved = state.customOverrides?.[violationId]?.[task.id] ?? task.default;
          if (model === saved) {
            option.selected = true;
          }
          select.appendChild(option);
        });
      }

      wrapper.appendChild(label);
      wrapper.appendChild(select);
      details.appendChild(wrapper);
    });
    elements.customMappingAccordion.appendChild(details);
  });
}

window.saveCustomMappingConfiguration = async function () {
  try {
    const grouped = {};
    document.querySelectorAll("#customMappingAccordion select").forEach(select => {
      const violation = select.dataset.violation;
      const task = select.dataset.task;
      if (!grouped[violation]) {
        grouped[violation] = {};
      }
      grouped[violation][task] = select.value;
    });

    for (const violation in grouped) {
      const payload = {
        violation: violation,
        overrides: grouped[violation]
      };

      const response = await fetch("/api/custom-mapping", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Failed to save ${violation}`);
      }
    }
    state.customOverrides = grouped;
    console.log("AFTER SAVE:", state.customOverrides);

    await initializePlatformArchitecture();
    renderConfigPreview();
    elements.uploadStatus.textContent = "✓ All custom mappings saved successfully.";
  }
  catch (err) {
    console.error(err);
    elements.uploadStatus.textContent = "❌ Unable to save custom mappings.";
  }
};
function getGeoMode() {
  return document.querySelector('input[name="geoMode"]:checked').value;
}
/* ==========================================================================
   3. Core Processing Operations & Form Utilities
   ========================================================================== */

function buildConfig() {
  console.log("CUSTOM OVERRIDES =", state.customOverrides);
  const mode = getGeoMode();
  const activeViolations = [];
  
  Object.keys(state.pipelineBlueprints).forEach(key => {
    const cb = document.getElementById(`v_${key}`);
    if (cb && cb.checked) activeViolations.push(key);
  });
  console.log(activeViolations);
  console.log("BUILDCONFIG:", state.customOverrides);
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
    violation_detection: {
      taskkillenabled: activeViolations.length > 0,
      list_violations: activeViolations
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

/* ==========================================================================
   4. Componentized Detection Rendering Panel (FINAL APPROVED PASS)
   ========================================================================== */

function renderDetections() {
  if (!elements.detectionList) return;
  elements.detectionList.innerHTML = "";
  
  const trackingContext = { totalCount: 0, renderedKeys: new Set() };

  state.jobs.forEach((job) => {
    renderStandardDetections(job, trackingContext);
    renderStageDetections(job, trackingContext); // Fixed: Naming standard unified
  });

  if (trackingContext.totalCount === 0) {
    elements.detectionList.innerHTML = "<p class='summary'>Processed trace anomalies appear here dynamically.</p>";
  }
}

function renderStandardDetections(job, ctx) {
  const standardDetections = job.results?.detections || [];
  standardDetections.forEach((det) => {
    const timestamp = Number(det.timestamp_seconds ?? 0);
    const uniqueKey = `${job.video_id}_std_${det.object_class}_${det.frame_index || timestamp}`;
    
    if (ctx.renderedKeys.has(uniqueKey)) return;
    ctx.renderedKeys.add(uniqueKey);

    ctx.totalCount++;
    
    const lat = Number(det.latitude ?? 0);
    const lon = Number(det.longitude ?? 0);
    const locationText = (det.latitude !== undefined && det.longitude !== undefined) 
      ? `${lat.toFixed(6)}, ${lon.toFixed(6)}` 
      : "n/a";

    appendDetectionCard(
      det.object_class, 
      det.confidence, 
      `Frame ${det.frame_index}`, 
      locationText, 
      false
    );
  });
}

function renderStageDetections(job, ctx) {
  if (!job.stages) return;

  Object.keys(job.stages).forEach((stageKey) => {
    const stageViolations = job.stages[stageKey]?.violations || [];
    stageViolations.forEach((v) => {
      const timestamp = Number(v.timestamp_seconds ?? 0);
      const trackingIdToken = v.track_id ?? v.bike_track_id ?? v.frame_index ?? "0";
      
      // Fixed: Secure multi-entity key tracking prevents timestamp collision anomalies
      const uniqueKey = `${job.video_id}_stage_${stageKey}_${trackingIdToken}_${timestamp.toFixed(2)}`;
      
      if (ctx.renderedKeys.has(uniqueKey)) return;
      ctx.renderedKeys.add(uniqueKey);

      ctx.totalCount++;
      
      let latVal = v.latitude;
      let lonVal = v.longitude;
      
      if (latVal === undefined || lonVal === undefined) {
        const routePoint = findNearestRoutePoint(job.results?.route, timestamp);
        if (routePoint) {
          latVal = routePoint.latitude;
          lonVal = routePoint.longitude;
        }
      }

      const resolvedGeoText = (latVal !== undefined && lonVal !== undefined) 
        ? `📍 ${Number(latVal).toFixed(6)}, ${Number(lonVal).toFixed(6)}` 
        : "Pending GPS alignment";
        
      // Fixed: Dynamic tracking label scales cleanly across multi-modal processors
      const trackId = v.track_id ?? v.bike_track_id ?? "N/A";
      const metaLabel = `Time: ${timestamp.toFixed(2)}s • Track ID: ${trackId}`;
      
      appendDetectionCard(`🚨 ${v.violation_type || stageKey}`, v.confidence, metaLabel, resolvedGeoText, true);
    });
  });
}

function appendDetectionCard(label, confidenceValue, summaryLine, locationLine, isViolation = false) {
  const card = document.createElement("article");
  card.className = "detection-card";
  const titleClass = isViolation ? "detection-title violation" : "detection-title";
  const score = Number(confidenceValue ?? 0);

  card.innerHTML = `
    <header class="detection-header">
      <strong class="${titleClass}">${Dictionary.getViolationLabel(label).toUpperCase()}</strong>
      <span>${(score * 100).toFixed(1)}%</span>
    </header>
    <div class="summary">${summaryLine}</div>
    <div class="summary data-route-coordinate">${locationLine}</div>
  `;
  elements.detectionList.appendChild(card);
}
/* ==========================================================================
   5. Leaflet Canvas & OSRM Route Tracing (FINAL APPROVED PASS)
   ========================================================================== */

function initMap() {
  if (!document.getElementById("map")) return;
  
  // Set up the high-contrast baseline Map viewport portal
  state.map = L.map("map").setView([20.2961, 85.8245], 13);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, 
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(state.map);
  
  state.markersLayer = L.layerGroup().addTo(state.map);
}

async function renderSnappedRoute(rawRoute) {
  // Boundary guards protect from empty route or null map instance failures
  if (!state.map || !rawRoute || rawRoute.length === 0) return;

  if (state.timelinePath) {
    state.map.removeLayer(state.timelinePath);
    state.timelinePath = null;
  }

  // Defensive Cancellation Check: Kill running routing fetches to prevent race overwrites
  if (state.activeRouteController) {
    state.activeRouteController.abort();
  }
  state.activeRouteController = new AbortController();
  
  // Capture explicit local reference to track active thread scope
  const controller = state.activeRouteController;

  try {
    const chunkSize = 40; 
    const fetchPromises = [];

    for (let i = 0; i < rawRoute.length; i += chunkSize) {
      const chunk = rawRoute.slice(i, i + chunkSize + 1); 
      if (chunk.length < 2) continue;

      const coordinateString = chunk.map(coord => `${coord.longitude},${coord.latitude}`).join(';');
      const radiusString = chunk.map(() => "45").join(';');
      const osrmMatchUrl = `https://router.project-osrm.org/match/v1/driving/${coordinateString}?overview=full&geometries=geojson&radiuses=${radiusString}`;
      
      fetchPromises.push(
        fetch(osrmMatchUrl, { signal: controller.signal })
          .then(res => {
            // Explicit response check catches HTTP errors before passing to json parser
            if (!res.ok) throw new Error(`OSRM HTTP route stream error: ${res.status}`);
            return res.json();
          })
          .then(data => {
            if (data.matchings && data.matchings.length > 0) {
              return data.matchings[0].geometry.coordinates.map(coord => [coord[1], coord[0]]);
            } else {
              return chunk.map(coord => [coord.latitude, coord.longitude]);
            }
          })
          .catch(err => {
            if (err.name === 'AbortError') {
              console.debug('Stale street-routing trace batch aborted.');
            } else {
              console.warn('Batch chunk route fallback applied:', err);
            }
            return chunk.map(coord => [coord.latitude, coord.longitude]);
          })
      );
    }

    // Await execution safely to let cascading maps synchronizers read accurate layout geometries
    const allChunkPoints = await Promise.all(fetchPromises);

    // Terminate rendering cascade instantly if another route thread has superceded this loop
    if (controller !== state.activeRouteController) return;

    const flatSnappedPoints = allChunkPoints.flat();
    if (flatSnappedPoints.length === 0) return;
    
    state.timelinePath = L.polyline(flatSnappedPoints, {
      color: '#8d4f18',   // Unified primary accent theme token
      weight: 6,          
      opacity: 0.9,
      dashArray: '1, 1',  
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(state.map);

    state.map.fitBounds(state.timelinePath.getBounds(), { padding: [40, 40] });
    
  } catch (err) {
    if (err.name !== "AbortError") {
        console.error(
            "Geospatial route coordinate stream breakdown:",
            err
        );
    }
}
}

function clearDashboardLayers() {
  if (state.timelinePath) { 
    state.map.removeLayer(state.timelinePath); 
    state.timelinePath = null; 
  }
  if (state.markersLayer) {
    state.markersLayer.clearLayers();
  }
  if (state.heatLayer) { 
    state.map.removeLayer(state.heatLayer); 
    state.heatLayer = null; 
  }
  
  // Kill running coordinate fetches instantly if dashboard is forcefully wiped clean
  if (state.activeRouteController) {
    state.activeRouteController.abort();
    state.activeRouteController = null; // Clean up memory pointers safely
  }

  state.violations = [];
  if (typeof renderViolations === "function") renderViolations();
  
  if (elements.heatmapSummary) {
    elements.heatmapSummary.textContent = "0 elements matching active calendar selection.";
  }
}
/* ==========================================================================
   6. Dynamic Splitting Heatmap Engine (FINAL APPROVED PASS)
   ========================================================================= */

async function refreshHeatmap() {
  if (!state.map) return;
  
  // Fixed: Streamlined pointer array selection fallback using native array accessor tokens
  const currentVideoId = state.currentVideoId ?? state.jobs.at(-1)?.video_id;
  if (!currentVideoId) {
    clearDashboardLayers();
    return;
  }

  try {
    const points = await fetchHeatmapPoints(currentVideoId);
    
    // Fixed: Defensive wrapper checks avoid failures if run ahead of main map engine setups
    if (state.markersLayer) {
      state.markersLayer.clearLayers();
    }
    if (state.heatLayer) {
      state.map.removeLayer(state.heatLayer);
    }

    const selectedFilter = elements.dashboardViolationFilter?.value || "all_violations";
    const { heatmapCoordinates, matchedCount } = processAndRenderMarkers(points, currentVideoId, selectedFilter);

    if (elements.heatmapSummary) {
      elements.heatmapSummary.textContent = `${matchedCount} geo-tagged elements matching filters active.`;
    }

    if (heatmapCoordinates.length > 0) {
      renderHeatLayer(heatmapCoordinates);
    }
  } catch (err) {
    console.error("Heatmap rendering execution sequence failure:", err);
    if (elements.heatmapSummary) {
      elements.heatmapSummary.textContent = "Unable to process pipeline heatmap arrays.";
    }
  }
}

async function fetchHeatmapPoints(videoId) {
  const params = new URLSearchParams({ video_id: videoId });
  if (elements.heatmapObjectFilter?.value) params.set("object_type", elements.heatmapObjectFilter.value);
  if (elements.heatmapStart?.value) params.set("start_time", elements.heatmapStart.value);
  if (elements.heatmapEnd?.value) params.set("end_time", elements.heatmapEnd.value);

  const response = await fetch(`/heatmap?${params.toString()}`);
  if (!response.ok) {
    throw new Error(`Heatmap API response error: ${response.status}`);
  }
  return await response.json();
}

function processAndRenderMarkers(points, videoId, selectedFilter) {
  const heatmapCoordinates = [];
  let matchedCount = 0;

  points.forEach((point) => {
    if (point.latitude === undefined || point.longitude === undefined) return;
    
    const rawType = point.violation_type || point.object_class || "";
    
    if (Dictionary.matchesFilter(rawType, selectedFilter)) {
      matchedCount++;
      const lat = Number(point.latitude ?? 0);
      const lon = Number(point.longitude ?? 0);
      
      // Fixed: Nullish coalescing preserves 0 intensity scaling drops safely
      heatmapCoordinates.push([lat, lon, point.intensity ?? 0.6]);

      const marker = L.marker([lat, lon]);
      
      // Fixed: Future-proof signatures pass plate numbers and evidence references automatically
      const popupContent = Dictionary.createMapPopupHTML(
        "VIOLATION CAPTURED",
        rawType,
        point.confidence ?? 0.85,
        point.video_id ?? videoId,
        point.plate_number,
        point.image_url
      );
      marker.bindPopup(popupContent);
      
      if (state.markersLayer) {
        state.markersLayer.addLayer(marker);
      }
    }
  });

  return { heatmapCoordinates, matchedCount };
}

function renderHeatLayer(coordinates) {
  state.heatLayer = L.heatLayer(coordinates, {
    radius: 28, 
    blur: 18, 
    maxZoom: 17,
    gradient: { 0.1: "#f1c27d", 0.4: "#e58f45", 0.7: "#c94c1a", 1.0: "#821f04" },
  }).addTo(state.map);

  const bounds = L.latLngBounds(coordinates.map(([lat, lon]) => [lat, lon]));
  
  // Fixed: Map view framing runs exclusively if coordinates yield valid Leaflet geometric shapes
  if (bounds.isValid()) {
    state.map.fitBounds(bounds, { padding: [40, 40] });
  }
}

/* ==========================================================================
   7. Asynchronous Historical Timelines & Dashboard Synchronization (FINAL PASS)
   ========================================================================== */

function getAssembledDateString() {
  if (!elements.archiveYearFilter?.value || !elements.archiveMonthFilter?.value || !elements.archiveDayFilter?.value) {
    const today = new Date();
    const yyyy = today.getFullYear();
    const mm = String(today.getMonth() + 1).padStart(2, '0');
    const dd = String(today.getDate()).padStart(2, '0');
    return `${yyyy}-${mm}-${dd}`;
  }
  
  const year = elements.archiveYearFilter.value;
  let month = elements.archiveMonthFilter.value;
  
  const monthMap = { 
    "January":"01", "February":"02", "March":"03", "April":"04", 
    "May":"05", "June":"06", "July":"07", "August":"08", 
    "September":"09", "October":"10", "November":"11", "December":"12" 
  };
  
  if (monthMap[month]) month = monthMap[month];
  const day = String(elements.archiveDayFilter.value).padStart(2, '0');
  
  return `${year}-${month}-${day}`;
}

async function initDashboardFilters() {
  try {
    const response = await fetch('/api/filter-metadata');
    if (response.ok) {
      const metadata = await response.json();
      if (metadata.default_date) {
        const [year, month, day] = metadata.default_date.split("-");

        if (elements.archiveDayFilter) {
          const optionExists = Array.from(elements.archiveDayFilter.options).some(opt => opt.value === day);
          if (!optionExists) {
            const newOpt = document.createElement("option");
            newOpt.value = day; newOpt.textContent = day;
            elements.archiveDayFilter.appendChild(newOpt);
          }
          elements.archiveDayFilter.value = day;
        }
        
        if (elements.archiveMonthFilter) {
          const revMonthMap = { 
            "01":"January", "02":"February", "03":"March", "04":"April", 
            "05":"May", "06":"June", "07":"July", "08":"August", 
            "09":"September", "10":"October", "11":"November", "12":"December" 
          };
          elements.archiveMonthFilter.value = revMonthMap[month] || "July"; 
        }
        
        if (elements.archiveYearFilter) {
          elements.archiveYearFilter.value = year;
        }
      }
    }
  } catch (metaErr) {
    console.warn("Falling back to baseline system static selection boundaries:", metaErr);
  }
  
  await handleDateMigration();
}

async function handleDateMigration() {
  if (!elements.dashboardVideoSelect) return;
  const selectedDate = getAssembledDateString();
  state.currentDate = selectedDate;

  try {
    const response = await fetch(`/api/videos-by-date/${selectedDate}`);
    if (!response.ok) throw new Error(`Video track fetch failure HTTP: ${response.status}`);
    const payload = await response.json();

    if (!payload.video_ids || payload.video_ids.length === 0) {
      elements.dashboardVideoSelect.innerHTML = '<option value="">No video tracks parsed</option>';
      state.currentVideoId = null;
      clearDashboardLayers();
      return;
    }

    elements.dashboardVideoSelect.innerHTML = payload.video_ids.map(
      id => `<option value="${id}">Video Track Ref #${id}</option>`
    ).join("");

    // Fixed: Modern safe element indexing readability token (.at)
    const targetVideoId = payload.video_ids.at(0);
    if (targetVideoId) {
      elements.dashboardVideoSelect.value = targetVideoId;
      await syncDashboardTimeline(targetVideoId);
    } else {
      clearDashboardLayers();
    }
  } catch (err) {
    console.error("Date migration parsing sequence error:", err);
    elements.dashboardVideoSelect.innerHTML = '<option value="">No video tracks parsed</option>';
    clearDashboardLayers();
  }
}

async function syncDashboardTimeline(videoId) {
  if (!videoId) {
    clearDashboardLayers();
    return;
  }
  
  state.currentVideoId = Number(videoId);

  const selectedViolationType = normalizeViolationType(
    elements.dashboardViolationFilter?.value ?? "all_violations"
  );

  // Defensive Signal Interceptor: Drop lingering track queries to resolve click race conditions
  if (state.activeTimelineController) {
    state.activeTimelineController.abort();
  }
  state.activeTimelineController = new AbortController();
  const controller = state.activeTimelineController;

  const url = `/api/timeline-with-violations/${videoId}?violation_type=${encodeURIComponent(selectedViolationType)}`;

  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`Timeline synchronization HTTP error: ${response.status}`);
    const syncData = await response.json();

    // Check thread consistency locally before continuing rendering operations
    if (controller !== state.activeTimelineController) return;

    // 1. Asynchronously await road snapping calculations before evaluating sub-layers
    if (syncData.route && syncData.route.length > 0) {
      await renderSnappedRoute(syncData.route);
    } else if (state.map && state.timelinePath) {
      state.map.removeLayer(state.timelinePath);
      state.timelinePath = null;
    }

    // 2. Clear out running map markers to completely prevent duplicate artifacting
    if (state.markersLayer) {
      state.markersLayer.clearLayers();
    }

    const rawViolations = syncData.violations ?? [];
    state.violations = rawViolations.map(v => ({ ...v, video_id: videoId }));

    if (typeof renderViolations === "function") {
      renderViolations();
    }

    let matchedCount = 0;
    state.violations.forEach(v => {
      const vType = v.violation_type ?? "";

      if (
        Dictionary.matchesFilter(vType, selectedViolationType) &&
        v.latitude !== undefined &&
        v.longitude !== undefined
      ) {
        matchedCount++;

        const marker = L.marker([
          Number(v.latitude),
          Number(v.longitude)
        ]);

        marker.bindPopup(
          Dictionary.createMapPopupHTML(
            "VIOLATION CAPTURED",
            vType,
            v.confidence ?? 0.85,
            videoId,
            v.plate_number,
            v.image_url
          )
        );

        state.markersLayer?.addLayer(marker);
      }
    });

    if (elements.heatmapSummary) {
      elements.heatmapSummary.textContent =
        `${matchedCount} geo-tagged elements matching filters active.`;
    }

    await refreshHeatmap();

} catch (syncErr) {
    if (syncErr.name === "AbortError") {
        console.debug("Stale timeline tracking query sequence safely dropped.");
    } else {
        console.error("Timeline layers synchronization query pass failed:", syncErr);
    }
} finally {
    // Always release the controller if this request is still the active one
    if (controller === state.activeTimelineController) {
        state.activeTimelineController = null;
    }
}
}

async function loadPipelineBlueprint() {

    const response = await fetch("/api/default-mapping");

    if (!response.ok)
        return;

    const data = await response.json();
    console.log("DEFAULT MAPPING =", data);

    state.defaultMapping = data;
    state.pipelineBlueprints = data;
    state.customMapping = {};
    const models = new Set();
    Object.values(data).forEach(violation => {

        (violation.tasks || []).forEach(task => {

            if (task.type === "model") {
                models.add(task.default);
            }

        });

    });

    state.availableModels = [...models];
    console.log("MODELS =", state.availableModels);
    renderDefaultConfiguration();
    renderAvailableModels();
    console.log("Render called");
    renderViolationChecklist();
    renderCustomMappingAccordion();
    renderConfigPreview();
}

function renderDefaultConfiguration() {

    const table = document.getElementById("defaultMappingTableBody");

    if (!table)
        return;

    table.innerHTML = "";

    Object.entries(state.defaultMapping).forEach(([violation, config]) => {

        (config.tasks || []).forEach(task => {

            const row = document.createElement("tr");

            row.innerHTML = `
                <td>${config.name || violation}</td>
                <td>${task.name}</td>
                <td>${task.default}</td>
            `;

            table.appendChild(row);

        });

    });
  }

  function renderAvailableModels() {

    const container = document.getElementById("availableModels");

    if (!container) return;

    container.innerHTML = "";

    state.availableModels.forEach(model => {

        const item = document.createElement("div");

        item.className = "minio-file-item";

        item.textContent = model;

        container.appendChild(item);

    });

}

function renderViolationChecklist() {

    const container = document.getElementById("violationChecklist");
    if (!container) return;

    container.innerHTML = "";

    Object.entries(state.defaultMapping).forEach(([key, value]) => {

        const label = document.createElement("label");
        label.className = "checklist-item";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.id = `v_${key}`;
        checkbox.value = key;

        // default state
        checkbox.checked = false;

        // update JSON preview whenever user changes selection
        checkbox.addEventListener("change", renderConfigPreview);

        label.appendChild(checkbox);
        label.append(" " + value.name);

        container.appendChild(label);
    });

    renderConfigPreview();
}

/* ==========================================================================
   8. Generic Violations Card List Renderer (FINAL PASS)
   ========================================================================== */

function renderViolations() {
  if (!elements.violationList) return;
  elements.violationList.innerHTML = "";
  
  if (!state.violations || !state.violations.length) {
    elements.violationList.innerHTML = "<p class='summary'>No violations have triggered operational logging checkpoints.</p>";
    return;
  }

  state.violations.forEach((v, index) => {
    const card = document.createElement("article");
    card.className = "panel violation-item-card";
    
    // Fixed: Keep original casing for unknown fallback formatting rules
    const rawDisplayType = v.violation_type ?? "Anomaly";
    const displayType = normalizeViolationType(rawDisplayType);
    
    // Fixed: Explicit type verification prints valid metadata rows on zero counts
    const countInfo = Number.isFinite(v.person_count) ? `• ${v.person_count} persons ` : "";
    
    // String trimming guard prevents empty-string payload records from falsifying references
    const evidenceLink = v.image_url?.trim() 
      ? `<a href="${v.image_url.trim()}" target="_blank" class="violation-evidence-link">View Evidence Image</a>` 
      : `<span class="violation-no-evidence">No snapshot capture stored</span>`;

    // Fixed: Color checks process optimized token matches via the dictionary module
    const highlightClass = Dictionary.getViolationColorClass(displayType);

    // Absolute defensive parameter assignments prevent thread breakdown on faulty numbers
    const confidence = Number(v.confidence ?? 0);
    const timestamp = Number(v.timestamp_seconds ?? 0);
    const lat = Number(v.latitude ?? 0);
    const lon = Number(v.longitude ?? 0);

    const geoText = (v.latitude !== undefined && v.longitude !== undefined)
      ? `Lat: ${lat.toFixed(6)}, Lon: ${lon.toFixed(6)}`
      : "n/a";

    card.innerHTML = `
      <header class="violation-header">
        <div>
          <strong class="${highlightClass}">${Dictionary.getViolationLabel(rawDisplayType).toUpperCase()} — Frame ${v.frame_index ?? "N/A"}</strong>
          <div class="summary violation-meta">Video ID: ${v.video_id} • ${timestamp.toFixed(2)}s ${countInfo}• confidence ${(confidence * 100).toFixed(1)}%</div>
          <div class="summary data-route-coordinate">
            📍 ${geoText}
          </div>
          ${evidenceLink}
        </div>
        <label class="violation-checkbox-label">
          <input type="checkbox" ${v.checked ? "checked" : ""} /> Reviewed
        </label>
      </header>
    `;

    // Conditional structural selector guarding completely isolates reference exceptions
    const checkbox = card.querySelector("input[type='checkbox']");
    checkbox?.addEventListener("change", (e) => {
      state.violations[index].checked = e.target.checked;
    });
    
    elements.violationList.appendChild(card);
  });
}

/* ==========================================================================
   9. Production Setup Interceptors & Event Listeners (FINAL PASS)
   ========================================================================== */

function registerDashboardInterceptors() {
  // 1. Fixed: Calendar group binding removes duplicate layout logic blocks
  [
    elements.archiveYearFilter,
    elements.archiveMonthFilter,
    elements.archiveDayFilter
  ].forEach(element => {
    element?.addEventListener("change", handleDateMigration);
  });

  // 2. Video track selection menu interceptor
  elements.dashboardVideoSelect?.addEventListener("change", async (e) => {
    // Fixed: Defensive token checking safely reads context properties
    const selectedVideoId = e.target?.value;
    if (selectedVideoId) {
      await syncDashboardTimeline(selectedVideoId);
    } else {
      clearDashboardLayers();
    }
  });

  // 3. Central violation category compliance filter
  elements.dashboardViolationFilter?.addEventListener("change", async () => {
    const activeVideoId = elements.dashboardVideoSelect?.value;
    if (activeVideoId) {
      await syncDashboardTimeline(activeVideoId);
    }
  });

  // 4. Fixed: Heatmap multi-event loop drives tuning updates declarations cleanly
  [
    [elements.heatmapObjectFilter, "change"],
    [elements.heatmapStart, "input"],
    [elements.heatmapEnd, "input"]
  ].forEach(([element, eventName]) => {
    element?.addEventListener(eventName, refreshHeatmap);
  });
}
/* ==========================================================================
   10. Integrated DOM Lifecycle Setup (MODULE 10)
   ========================================================================== */

document.addEventListener("DOMContentLoaded", async () => {
  console.log("🚀 Initializing Integrated Analytics & Ingestion Engine Lifecycle...");

  // 1. Core Map Interface Portal Instantiation
  if (typeof initMap === "function") {
    initMap();
  } else {
    console.error("❌ Critical Component Failure: initMap core engine structure not resolved.");
  }

  // 2. Form Submission Action Listener Registration
  if (elements.uploadForm) {
    elements.uploadForm.addEventListener("submit", uploadVideos);
  }

  // 3. Application Interface Component Setup Interceptors
  if (typeof registerDashboardInterceptors === "function") {
    registerDashboardInterceptors();
  }

  // 4. Asynchronous Filter Matrix Initialization & Boot Sequence Cascade
  if (typeof initDashboardFilters === "function") {
    await initDashboardFilters();
  } else {
    console.warn("⚠️ Meta-Filter Warning: initDashboardFilters sequence skipped or unavailable.");
  }
  await loadPipelineBlueprint();
  console.log("✅ Application Engine Architecture successfully mounted.");
});