/* SignAR FSL decoder foundation. The decoder never invents model predictions.
   Model input uses a versioned hand-only or holistic landmark contract. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const video = $("video");
  const landmarkCanvas = $("landmarkCanvas"),
    landmarkCtx = landmarkCanvas.getContext("2d");
  const toggleLandmarksBtn = $("toggleLandmarksBtn");
  const processCanvas = document.createElement("canvas"),
    processCtx = processCanvas.getContext("2d");
  const personCheckCanvas = document.createElement("canvas"),
    personCheckCtx = personCheckCanvas.getContext("2d");
  const cameraStage = $("cameraStage"),
    placeholder = $("cameraPlaceholder"),
    status = $("cameraStatus");
  const liveBadge = $("liveBadge"),
    trackingBadge = $("trackingBadge"),
    guide = $("positionGuide");
  const startBtn = $("startCameraBtn"),
    stopBtn = $("stopCameraBtn"),
    switchBtn = $("switchCameraBtn");
  const fullscreenBtn = $("fullscreenBtn"),
    installBtn = $("installBtn"),
    detected = $("detectedSign");
  const confidenceValue = $("confidenceValue"),
    confidenceBar = $("confidenceBar"),
    translation = $("translationText");
  const typePill = $("typePill"),
    state = $("recognitionState"),
    toast = $("toast"),
    overlay = $("arOverlay");
  const trackingFpsMetric = $("trackingFps"),
    inferenceLatencyMetric = $("inferenceLatency"),
    topKScores = $("topKScores"),
    scoreThresholdInput = $("scoreThreshold"),
    scoreThresholdValue = $("scoreThresholdValue"),
    diagnosticHandSource = $("diagnosticHandSource"),
    diagnosticFeatureHealth = $("diagnosticFeatureHealth"),
    diagnosticFeatureStats = $("diagnosticFeatureStats"),
    diagnosticWindow = $("diagnosticWindow"),
    diagnosticVotes = $("diagnosticVotes"),
    diagnosticDecision = $("diagnosticDecision"),
    diagnosticStageTimings = $("diagnosticStageTimings"),
    exportFeatureWindowsBtn = $("exportFeatureWindowsBtn");
  const arSign = $("arSign"),
    arMeaning = $("arMeaning"),
    rawOutput = $("rawOutput"),
    sentenceOutput = $("sentenceOutput");
  const speakBtn = $("speakBtn"),
    deleteBtn = $("deleteBtn"),
    clearBtn = $("clearSentenceBtn"),
    copyBtn = $("copySentenceBtn");
  const autoSpeak = $("autoSpeakToggle"),
    voiceLanguage = $("voiceLanguage"),
    speechStatus = $("speechStatus"),
    modelStatus = $("modelStatus");
  const navButtons = [...document.querySelectorAll(".nav-btn")],
    sections = [...document.querySelectorAll(".page-section")];

  const PIPELINE = window.SignARPipeline;
  const PREDICTION_HISTORY = window.SignARPredictionHistory;
  const STREAM_SEGMENTER = window.SignARStreamSegmenter;
  const FACE_DETECTION_BASES = PIPELINE.config.assets.faceDetection;
  const TENSORFLOW_BASES = PIPELINE.config.assets.tensorflow;
  let activeSchema = PIPELINE.getSchema(PIPELINE.config.features.version),
    SEQUENCE_LENGTH = activeSchema.sequenceLength,
    FEATURE_DIMENSION = activeSchema.dimension;
  const FEATURE_SAMPLE_INTERVAL_MS =
      1000 / PIPELINE.config.sequence.samplingFps,
    INFERENCE_INTERVAL_MS = PIPELINE.config.sequence.inferenceIntervalMs,
    MINIMUM_BUFFER_FRAMES = PIPELINE.config.sequence.minimumInferenceFrames;
  let MAX_FEATURE_FRAMES = activeSchema.windowFrames;
  const PERSON_CHECK_DIMENSION = PIPELINE.config.tracking.personCheckDimension,
    PERSON_CHECK_INTERVAL_MS = PIPELINE.config.tracking.personCheckIntervalMs;
  let minimumConfidencePercent =
    PIPELINE.config.recognition.minimumConfidencePercent;
  const MINIMUM_CLASS_RECALL = 0.8;
  const NO_SIGN_LABEL = "NO SIGN";
  const MODEL_REQUEST_TIMEOUT_MS = 20000;
  const MODEL_BENCHMARK_TIMEOUT_MS = 60000;
  const MODEL_LOAD_TIMEOUT_MS = 30000;
  const EXTERNAL_SCRIPT_TIMEOUT_MS = 12000;
  const DEBUG_TELEMETRY =
    new URLSearchParams(window.location.search).get("debug") === "1";
  const trackingState = PIPELINE.createTrackingState();
  const tracker = window.SignARTracker.create({
    onWarning: (message, error) => console.warn(message, error),
  });
  let stream = null,
    cameraStartGeneration = 0,
    cameraStarting = false,
    trackerReady = false,
    faceDetection = null,
    faceDetectionInitPromise = null,
    faceDetectionBase = FACE_DETECTION_BASES[0],
    faceDetectionLoadPromise = null,
    faceDetectionBusy = false,
    faceDetectionGeneration = 0,
    tensorFlowLoadPromise = null,
    tensorFlowWebGpuBackendPromise = null,
    modelManifestPromise = null,
    detectedFaceBoxes = [],
    multiplePeopleDetected = false,
    singlePersonFrames = 0,
    lastFaceDetectionAt = -PERSON_CHECK_INTERVAL_MS,
    facingMode = "environment",
    rafId = null,
    processing = false,
    lastVideoTime = -1,
    lastFrameAt = 0,
    lastTrackingResultAt = 0,
    lastCompletedFrameAt = 0,
    lastMetricsAt = 0,
    trackingFps = 0,
    cameraTrackListeners = [];
  let processEvery = trackingState.processIntervalMs,
    model = null,
    modelConfig = null,
    modelReady = false,
    modelLoading = false,
    modelLoadPromise = null;
  let segmentFrames = [],
    segmentFrameTimes = [],
    predictionHistory = PREDICTION_HISTORY.createStreaming(),
    streamSegmenter = STREAM_SEGMENTER.create({
      pauseMs: PIPELINE.config.sequence.segmentationPauseMs,
      transitionWindows: PIPELINE.config.sequence.segmentationTransitionWindows,
      minimumWindows: PIPELINE.config.sequence.segmentationTransitionWindows,
    }),
    previousFeatures = null,
    signerVisible = false,
    inferenceBusy = false,
    lastInferenceAt = 0,
    predictionResult = null,
    lastFeatureAt = 0,
    trackingLossReleased = false,
    exportedFeatureWindows = [],
    stageTimings = {
      capture: [],
      landmarks: [],
      features: [],
      inference: [],
      commit: [],
    },
    lastTimingRenderAt = 0;
  let recognitionGeneration = 0;
  let lastHandSource = "Waiting for tracker",
    lastSelectedHandSlot = "none",
    lastFacePresent = false,
    lastPosePresent = false,
    lastValidHandPoints = 0,
    lastFeatureExportAt = 0,
    lastDebugTelemetryAt = 0;
  let gapResetCount = 0,
    lastGapResetReason = "none",
    lastPredictionDecision = "Waiting for a holistic model window.";
  let averageInferenceMs = 0,
    inferenceCount = 0,
    speechTimer = null,
    noSignSince = null,
    pendingPauseMs = 0,
    sentence = [],
    recognizedSigns = [],
    sentenceRevision = 0,
    speechRevision = 0;
  let installPrompt = null,
    voices = [];
  let landmarksVisible = true;

  const WORD_SEPARATOR_PAUSE_MS =
    PIPELINE.config.recognition.wordSeparatorPauseMs;
  function beginNoSignPause(time) {
    if (noSignSince === null) noSignSince = time;
  }
  function finishNoSignPause(time) {
    if (noSignSince === null) return;
    pendingPauseMs = Math.max(0, time - noSignSince);
    noSignSince = null;
  }
  function labelMetadataFor(label) {
    const normalized = String(label || '').trim().toUpperCase();
    if (!normalized) return { unitType: null, motionType: 'static', displayText: '' };
    const stored = modelConfig?.labelMetadata?.[normalized] || {};
    const unitType = normalized === NO_SIGN_LABEL
      ? 'background'
      : stored.unitType || (/^[A-Z]$/.test(normalized) ? 'fingerspelling' : 'word');
    const motionType = normalized === NO_SIGN_LABEL
      ? 'static'
      : unitType === 'word' || /^[JZ]$/.test(normalized)
        ? 'dynamic'
        : (stored.motionType || 'static');
    return {
      unitType,
      motionType,
      displayText: String(stored.displayText || '').trim() || normalized,
    };
  }
  function adaptTrackingResolution(elapsed) {
    PIPELINE.adaptTrackingState(trackingState, elapsed);
    processEvery = trackingState.processIntervalMs;
  }
  function toastMessage(text) {
    toast.textContent = text;
    toast.classList.add("show");
    clearTimeout(toastMessage.timer);
    toastMessage.timer = setTimeout(() => toast.classList.remove("show"), 2800);
  }
  function setState(text, cls = "") {
    state.textContent = text;
    state.className = ("recognition-state " + cls).trim();
  }
  function withTimeout(promise, timeoutMs, description) {
    let timer;
    return Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = window.setTimeout(
          () => reject(new Error(description + " timed out after " + Math.round(timeoutMs / 1000) + " seconds.")),
          timeoutMs,
        );
      }),
    ]).finally(() => window.clearTimeout(timer));
  }
  async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timer = window.setTimeout(
      () => controller.abort(),
      MODEL_REQUEST_TIMEOUT_MS,
    );
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } catch (error) {
      if (controller.signal.aborted)
        throw new Error(
          "Timed out reading " + url + " after " +
            Math.round(MODEL_REQUEST_TIMEOUT_MS / 1000) + " seconds.",
        );
      throw error;
    } finally {
      window.clearTimeout(timer);
    }
  }
  function loadExternalScript(src, validate, description) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      let settled = false;
      const finish = (error) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        script.onload = null;
        script.onerror = null;
        if (error) {
          script.remove();
          reject(error);
        } else resolve();
      };
      const timer = window.setTimeout(
        () =>
          finish(
            new Error(
              "Timed out loading " + src + " after " +
                Math.round(EXTERNAL_SCRIPT_TIMEOUT_MS / 1000) + " seconds.",
            ),
          ),
        EXTERNAL_SCRIPT_TIMEOUT_MS,
      );
      script.src = src;
      script.crossOrigin = "anonymous";
      script.onload = () => {
        try {
          if (validate && !validate())
            throw new Error(description + " loaded but was not available.");
          finish();
        } catch (error) {
          finish(error);
        }
      };
      script.onerror = () =>
        finish(new Error("Could not load " + src + "."));
      document.head.appendChild(script);
    });
  }
  function cameraStatus(text, online = false) {
    status.classList.toggle("online", online);
    status.innerHTML = "<span></span> " + text;
  }
  function updateSentence() {
    const hasSentence = sentence.length > 0;
    rawOutput.textContent = recognizedSigns.length
      ? recognizedSigns.join(" | ")
      : "Wala pang na-decode na senyas.";
    sentenceOutput.textContent = hasSentence
      ? sentence.join("")
      : "Ang na-decode na pangungusap ay lalabas dito.";
    deleteBtn.disabled = !hasSentence;
    clearBtn.disabled = !hasSentence;
    copyBtn.disabled = !hasSentence;
    speakBtn.disabled = !hasSentence || !("speechSynthesis" in window);
  }
  function stopSpeech() {
    speechRevision++;
    if (speechTimer) clearTimeout(speechTimer);
    speechTimer = null;
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    speakBtn.textContent = "Speak";
    speakBtn.classList.remove("speaking");
    speakBtn.disabled = !sentence.length || !("speechSynthesis" in window);
  }
  function chooseFilipinoVoice() {
    if (!("speechSynthesis" in window)) return null;
    voices = window.speechSynthesis.getVoices() || [];
    const selectedLanguage = voiceLanguage?.value || "fil-PH";
    if (selectedLanguage === "auto") return null;
    return (
      voices.find(
        (v) => v.lang.toLowerCase() === selectedLanguage.toLowerCase(),
      ) ||
      voices.find((v) =>
        new RegExp(
          "^" +
            selectedLanguage
              .split("-")[0]
              .replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
          "i",
        ).test(v.lang),
      ) ||
      voices.find((v) => /^fil(-|$)/i.test(v.lang)) ||
      voices.find((v) => /^tl(-|$)/i.test(v.lang)) ||
      voices.find((v) => /^en-PH$/i.test(v.lang)) ||
      null
    );
  }
  function updateSpeechSupport() {
    if (
      !("speechSynthesis" in window) ||
      typeof SpeechSynthesisUtterance === "undefined"
    ) {
      speechStatus.textContent =
        "Voice playback is not supported in this browser.";
      speakBtn.disabled = true;
      return;
    }
    const selected = chooseFilipinoVoice();
    speechStatus.textContent = selected
      ? "Voice ready: " + selected.name
      : "Voice ready with the browser default.";
  }
  function speakSentence(manual = false) {
    if (
      (!manual && !autoSpeak?.checked) ||
      !sentence.length ||
      !("speechSynthesis" in window) ||
      typeof SpeechSynthesisUtterance === "undefined"
    )
      return;
    const speakingRevision = ++speechRevision;
    const text = sentence.join("");
    const utterance = new SpeechSynthesisUtterance(text);
    const selectedLanguage = voiceLanguage?.value || "fil-PH";
    utterance.lang = selectedLanguage === "auto" ? "fil-PH" : selectedLanguage;
    utterance.rate = 0.92;
    const voice = chooseFilipinoVoice();
    if (voice) utterance.voice = voice;
    speakBtn.disabled = true;
    speakBtn.textContent = "Speaking...";
    speakBtn.classList.add("speaking");
    speechStatus.textContent = "Speaking the decoded sentence...";
    utterance.onend = () => {
      if (speakingRevision !== speechRevision) return;
      speakBtn.disabled = !sentence.length;
      speakBtn.textContent = "Speak";
      speakBtn.classList.remove("speaking");
      updateSpeechSupport();
    };
    utterance.onerror = (event) => {
      if (speakingRevision !== speechRevision) return;
      speakBtn.disabled = !sentence.length;
      speakBtn.textContent = "Speak";
      speakBtn.classList.remove("speaking");
      speechStatus.textContent =
        event.error === "canceled"
          ? "Speech playback canceled."
          : "Unable to play voice: " + (event.error || "unknown error") + ".";
    };
    window.speechSynthesis.speak(utterance);
  }
  function scheduleSpeech() {
    if (speechTimer) clearTimeout(speechTimer);
    speechTimer = setTimeout(() => {
      speechTimer = null;
      speakSentence();
    },
    Math.max(
      PIPELINE.config.recognition.speechDelayMs,
      WORD_SEPARATOR_PAUSE_MS,
    ));
  }
  function addRecognizedUnit(label, confidence) {
    if (!Number.isFinite(confidence) || confidence < minimumConfidencePercent)
      return;
    const now = performance.now();
    const commitStarted = performance.now();
    lastPredictionDecision = `SEGMENT ${label}: continuous stream boundary confirmed.`;
    updateTemporalDiagnostics();
    const unit = labelMetadataFor(label);
    const previous = labelMetadataFor(recognizedSigns.at(-1));
    const startsLexicalUnit = unit.unitType === 'word' || unit.unitType === 'phrase';
    const followsLexicalUnit = previous.unitType === 'word' || previous.unitType === 'phrase';
    if (
      sentence.length &&
      sentence.at(-1) !== " " &&
      (startsLexicalUnit || followsLexicalUnit || pendingPauseMs >= WORD_SEPARATOR_PAUSE_MS)
    ) sentence.push(" ");
    sentence.push(unit.displayText);
    pendingPauseMs = 0;
    recognizedSigns.push(label);
    sentenceRevision++;
    updateSentence();
    scheduleSpeech();
    toastMessage("Segmented: " + label + " (" + Math.round(confidence) + "%)");
    recordStageTiming("commit", performance.now() - commitStarted);
  }

  function showNoSign(message) {
    const cameraActive = !!stream;
    const activeCameraMessage =
      message || "No sign detected yet. Keep your hands, face, or upper body in frame.";
    clearTopKScores();
    detected.textContent = modelReady
      ? "NO SIGN"
      : modelLoading
        ? "LOADING"
        : "UNAVAILABLE";
    confidenceValue.textContent = "—";
    confidenceBar.style.width = "0%";
    translation.textContent = modelReady
      ? cameraActive
        ? activeCameraMessage
        : "Decoder ready. Start the camera and keep your hands, face, or upper body in frame."
      : modelLoading
        ? modelStatus.textContent || "Checking and loading the eligible FSL model."
        : modelStatus.textContent;
    typePill.textContent = modelReady
      ? cameraActive
        ? "Waiting for sign"
        : "Decoder ready"
      : modelLoading
        ? "Loading model"
        : "Model unavailable";
    setState(
      modelReady
        ? cameraActive
          ? "NO SIGN"
          : "MODEL READY"
        : modelLoading
          ? "LOADING MODEL"
          : "MODEL NOT READY",
      modelLoading ? "verifying" : modelReady ? "" : "locked",
    );
    overlay.classList.remove("active");
  }
  function updateModelLoadingStatus(message) {
    modelStatus.textContent = message;
    if (modelLoading) showNoSign();
  }
  function clearTopKScores() {
    if (
      topKScores.childElementCount === 1 &&
      topKScores.firstElementChild.textContent === "Waiting for inference"
    )
      return;
    const entry = document.createElement("li");
    entry.textContent = "Waiting for inference";
    topKScores.replaceChildren(entry);
  }
  function resetRecognitionSegment() {
    recognitionGeneration++;
    segmentFrames = [];
    segmentFrameTimes = [];
    predictionHistory.clear();
    streamSegmenter.clear();
    predictionResult = null;
    previousFeatures = null;
    lastFeatureAt = 0;
    trackingLossReleased = false;
    lastInferenceAt = 0;
    clearTopKScores();
    lastPredictionDecision = "Segment cleared; waiting for fresh sign frames.";
    updateTemporalDiagnostics();
  }
  function updateTrackingStatus(r) {
    const left = r.leftHandLandmarks || [];
    const right = r.rightHandLandmarks || [];
    const face = !!r.faceLandmarks?.length;
    const pose = !!r.poseLandmarks?.length;
    const hands =
      Number(PIPELINE.hasUsableHand(left)) +
      Number(PIPELINE.hasUsableHand(right));
    const text = multiplePeopleDetected
      ? "MULTI-PERSON · " + detectedFaceBoxes.length + " FACES"
      : hands
        ? hands + " HAND" + (hands === 1 ? "" : "S") + (face ? " · FACE" : "") + (pose ? " · BODY" : "")
        : face || pose
          ? (face ? "FACE" : "") + (face && pose ? " · " : "") + (pose ? "BODY" : "")
          : "TRACKING: SEARCHING";
    if (trackingBadge.textContent !== text) trackingBadge.textContent = text;
    trackingBadge.classList.toggle(
      "detected",
      hands > 0 || face || pose || multiplePeopleDetected,
    );
    guide?.classList.toggle("is-hidden", hands > 0 || face || pose || multiplePeopleDetected);
  }
  function drawTrackerLandmarks(r) {
    if (!landmarksVisible) return;
    const width = video.videoWidth,
      height = video.videoHeight;
    if (!width || !height) return;
    const rect = video.getBoundingClientRect();
    window.SignARLandmarkRenderer.draw(
      landmarkCanvas,
      r,
      width,
      height,
      rect.width,
      rect.height,
      trackingState,
    );
  }
  function clearTrackerLandmarks() {
    landmarkCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
  }
  function setLandmarksVisible(visible) {
    landmarksVisible = visible;
    landmarkCanvas.hidden = !visible;
    toggleLandmarksBtn.setAttribute("aria-pressed", String(visible));
    toggleLandmarksBtn.textContent = visible
      ? "Hide landmarks"
      : "Show landmarks";
    if (!visible) clearTrackerLandmarks();
  }
  function updateTemporalDiagnostics() {
    if (!diagnosticWindow) return;
    diagnosticWindow.textContent = `buffer ${segmentFrames.length}/${MAX_FEATURE_FRAMES} · model ${Math.min(segmentFrames.length, SEQUENCE_LENGTH)}/${SEQUENCE_LENGTH} · segment ${streamSegmenter.activeLabel || "waiting"} · gaps ${gapResetCount} (${lastGapResetReason})`;
    const current = predictionHistory.recent;
    diagnosticVotes.textContent = current.length
      ? current
          .slice(-6)
          .map((item) => `${item.label}${item.accepted ? " ✓" : " ×"}`)
          .join(" → ")
      : "none";
    diagnosticDecision.textContent = lastPredictionDecision;
  }
  function updateDiagnostics() {
    diagnosticHandSource.textContent = lastHandSource;
    diagnosticFeatureHealth.textContent = lastValidHandPoints
      ? `${lastValidHandPoints}/21 ${lastSelectedHandSlot} hand points`
      : lastFacePresent || lastPosePresent
        ? "Face or body cues tracked"
        : "Searching for hand, face, or body cues";
    diagnosticFeatureStats.textContent = `left/right hands · face ${lastFacePresent ? "present" : "missing"} · body ${lastPosePresent ? "present" : "missing"}`;
    updateTemporalDiagnostics();
    renderStreamingPrediction();
    if (performance.now() - lastTimingRenderAt >= 500) {
      lastTimingRenderAt = performance.now();
      renderStageTimings();
    }
    trackingFpsMetric.textContent = trackingFps ? trackingFps.toFixed(1) : "—";
    inferenceLatencyMetric.textContent = inferenceCount
      ? averageInferenceMs.toFixed(1)
      : "—";
  }
  function recordStageTiming(stage, duration) {
    const values = stageTimings[stage];
    if (!values || !Number.isFinite(duration)) return;
    values.push(Math.max(0, duration));
    if (values.length > 300) values.shift();
  }
  function percentile(values, fraction) {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[
      Math.min(sorted.length - 1, Math.ceil(fraction * sorted.length) - 1)
    ];
  }
  function renderStageTimings() {
    const summary = {};
    for (const [stage, values] of Object.entries(stageTimings))
      summary[stage] = {
        p50: percentile(values, 0.5),
        p95: percentile(values, 0.95),
        count: values.length,
      };
    if (diagnosticStageTimings)
      diagnosticStageTimings.textContent = Object.entries(summary)
        .map(
          ([stage, value]) =>
            `${stage} ${value.p50 === null ? "—" : value.p50.toFixed(1) + "/" + value.p95.toFixed(1) + " ms"}`,
        )
        .join(" · ");
    if (DEBUG_TELEMETRY)
      console.debug("[SignAR] stage timing p50/p95 (ms)", summary);
    return summary;
  }
  function exportFeatureWindows() {
    if (!exportedFeatureWindows.length) {
      toastMessage("No live feature windows to export yet.");
      return;
    }
    const payload = {
      schemaVersion: 1,
      project: "SignAR FSL Decoder",
      featureVersion: PIPELINE.config.features.version,
      featureDimension: FEATURE_DIMENSION,
      sequenceLength: SEQUENCE_LENGTH,
      samplingFps: PIPELINE.config.sequence.samplingFps,
      exportedAt: new Date().toISOString(),
      windows: exportedFeatureWindows,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `signar-live-features-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toastMessage(
      `Exported ${exportedFeatureWindows.length} live feature windows.`,
    );
  }
  if (exportFeatureWindowsBtn) {
    exportFeatureWindowsBtn.hidden = !DEBUG_TELEMETRY;
    exportFeatureWindowsBtn.addEventListener("click", exportFeatureWindows);
  }

  async function warmupModel(loadedModel, schema = activeSchema) {
    const input = window.tf.zeros(PIPELINE.sequenceTensorShape(1, schema));
    let output = null,
      inferenceFeatureDiagnostics = null,
      inferenceTemporalDiagnostics = null;
    try {
      output = loadedModel.predict(input);
      const tensors = Array.isArray(output) ? output : [output];
      await Promise.all(tensors.map((tensor) => tensor.data()));
    } finally {
      input.dispose();
      if (output)
        (Array.isArray(output) ? output : [output]).forEach((tensor) =>
          tensor.dispose?.(),
        );
    }
  }

  async function loadTensorFlowLibrary() {
    if (window.tf) return;
    if (!tensorFlowLoadPromise)
      tensorFlowLoadPromise = (async () => {
        const failures = [];
        for (const src of TENSORFLOW_BASES) {
          try {
            await loadExternalScript(
              src,
              () => !!window.tf,
              "TensorFlow.js",
            );
            return;
          } catch (error) {
            failures.push(src + ": " + error.message);
          }
        }
        throw new Error(
          "Unable to load TensorFlow.js. Check your internet connection or content blocker. " +
            failures.join(" "),
        );
      })();
    try {
      await tensorFlowLoadPromise;
    } catch (error) {
      tensorFlowLoadPromise = null;
      throw error;
    }
  }

  async function loadTensorFlowWasmBackend() {
    if (window.tf?.wasm?.setWasmPaths) return;
    const failures = [];
    for (const source of PIPELINE.config.assets.tensorflowWasm) {
      try {
        await loadExternalScript(
          source,
          () => !!window.tf?.wasm?.setWasmPaths,
          "TensorFlow.js WebAssembly backend",
        );
        const directory = source.slice(0, source.lastIndexOf("/") + 1);
        window.tf.wasm.setWasmPaths(directory);
        return;
      } catch (error) {
        failures.push(`${source}: ${error.message}`);
      }
    }
    throw new Error(
      `Could not load the pinned TensorFlow.js WASM backend. ${failures.join(" ")}`,
    );
  }

  async function loadTensorFlowWebGpuBackend() {
    if (!navigator.gpu?.requestAdapter)
      throw new Error("WebGPU is not available in this browser.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter is available.");
    if (!tensorFlowWebGpuBackendPromise)
      tensorFlowWebGpuBackendPromise = (async () => {
        const failures = [];
        for (const source of PIPELINE.config.assets.tensorflowWebGpu) {
          try {
            await loadExternalScript(source, null, "TensorFlow.js WebGPU backend");
            return;
          } catch (error) {
            failures.push(`${source}: ${error.message}`);
          }
        }
        throw new Error(
          `Could not load the pinned TensorFlow.js WebGPU backend. ${failures.join(" ")}`,
        );
      })();
    try {
      await tensorFlowWebGpuBackendPromise;
    } catch (error) {
      tensorFlowWebGpuBackendPromise = null;
      throw error;
    }
  }

  function validateModelShape(loadedModel, config, schema = activeSchema) {
    const input = loadedModel.inputs?.[0]?.shape || [];
    const output = loadedModel.outputs?.[0]?.shape || [];
    const expectedInputShape = PIPELINE.sequenceTensorShape(null, schema);
    if (
      input.length !== expectedInputShape.length ||
      (input[0] !== null && input[0] !== 1) ||
      input.some(
        (size, index) =>
          expectedInputShape[index] !== null &&
          size !== expectedInputShape[index],
      )
    )
      throw new Error(
        "Model input shape mismatch; expected [batch," +
          schema.sequenceLength +
          "," +
          schema.dimension +
          "]",
      );
    if (output.at(-1) !== config.labels.length)
      throw new Error("Model class count mismatch");
  }

  async function benchmarkTensorFlowBackends(modelUrl, config, schema) {
    const tf = window.tf;
    const results = [];
    for (const backend of ["webgpu", "webgl", "wasm", "cpu"]) {
      let candidateModel = null;
      try {
        if (backend === "webgpu") await loadTensorFlowWebGpuBackend();
        if (backend === "wasm") await loadTensorFlowWasmBackend();
        if (!(await tf.setBackend(backend)))
          throw new Error("Backend registration failed.");
        await tf.ready();
        candidateModel = await tf.loadLayersModel(modelUrl);
        validateModelShape(candidateModel, config, schema);
        const times = [];
        for (let run = 0; run < 6; run++) {
          const input = tf.zeros(PIPELINE.sequenceTensorShape(1, schema));
          const started = performance.now();
          let output = null;
          try {
            output = candidateModel.predict(input);
            const tensors = Array.isArray(output) ? output : [output];
            await Promise.all(tensors.map((tensor) => tensor.data()));
          } finally {
            input.dispose();
            if (output)
              (Array.isArray(output) ? output : [output]).forEach((tensor) =>
                tensor.dispose?.(),
              );
          }
          if (run > 0) times.push(performance.now() - started);
        }
        times.sort((a, b) => a - b);
        results.push({
          backend,
          p50Ms: times[Math.floor(times.length / 2)],
          samples: times.length,
        });
      } catch (error) {
        results.push({ backend, error: error.message });
      } finally {
        try {
          candidateModel?.dispose();
        } catch (error) {
          console.warn("Could not release benchmark model weights.", error);
        }
      }
    }
    const fastest = results
      .filter((item) => Number.isFinite(item.p50Ms))
      .sort((a, b) => a.p50Ms - b.p50Ms)[0];
    if (!fastest) {
      await tf.setBackend("cpu");
      await tf.ready();
      throw new Error(
        "TensorFlow.js could not run this model on WebGPU, WebGL, WASM, or CPU.",
      );
    }
    await tf.setBackend(fastest.backend);
    await tf.ready();
    if (DEBUG_TELEMETRY)
      console.info("[SignAR] TensorFlow.js backend benchmark", {
        selected: fastest.backend,
        results,
      });
    return { selected: fastest.backend, results };
  }

  async function loadModelManifest() {
    if (!modelManifestPromise)
      modelManifestPromise = (async () => {
        const response = await fetchWithTimeout("./model/vocabulary.json", {
          cache: "no-store",
        });
        if (!response.ok)
          throw new Error(
            "Could not read ./model/vocabulary.json (HTTP " +
              response.status +
              ")",
          );
        return response.json();
      })();
    try {
      return await modelManifestPromise;
    } catch (error) {
      modelManifestPromise = null;
      throw error;
    }
  }
  async function loadModelMetadata(config) {
    const metadataPath = config.metadataPath || "./model/metadata.json";
    const response = await fetchWithTimeout(metadataPath, {
      cache: "no-store",
    });
    if (response.status === 404 && !config.metadataPath) {
      return {
        labels: config.labels,
        sequence_length: config.sequenceLength,
        feature_count: config.featureDimension,
      };
    }
    if (!response.ok)
      throw new Error(
        "Could not read " + metadataPath + " (HTTP " + response.status + ")",
      );
    const metadata = await response.json();
    if (
      !metadata ||
      !Array.isArray(metadata.labels) ||
      !metadata.labels.length ||
      metadata.labels.some(
        (label) => typeof label !== "string" || !label.trim(),
      ) ||
      !Number.isInteger(metadata.sequence_length) ||
      !Number.isInteger(metadata.feature_count)
    )
      throw new Error(
        "Model metadata must declare ordered labels, sequence_length, and feature_count",
      );
    return metadata;
  }
  function candidateBlockers(config) {
    const evaluation = config.evaluation || {},
      blockers = [];
    const heldOut = evaluation.leaveOneSignerOut || {};
    if (
      config.rejectLabel !== NO_SIGN_LABEL ||
      config.labels?.[0] !== NO_SIGN_LABEL
    )
      blockers.push("NO SIGN must be the explicit first reject class");
    if (config.acceptanceProfile !== "signar-standard-v1")
      blockers.push("manifest must declare the signar-standard-v1 validation profile");
    if (!Number.isFinite(heldOut.meanAccuracy) || heldOut.meanAccuracy < 0.9)
      blockers.push(
        "leave-one-signer-out mean accuracy is below 90% or missing",
      );
    if (!Array.isArray(heldOut.folds) || heldOut.folds.length < 5)
      blockers.push("evaluation must include all five signer-held-out folds");
    if (
      !Number.isFinite(evaluation.noSignFalseAcceptRate) ||
      evaluation.noSignFalseAcceptRate > 0.1
    )
      blockers.push("NO SIGN false-accept rate is above 10% or missing");
    if (
      !Number.isFinite(evaluation.noSignShare) ||
      evaluation.noSignShare < 0.01 ||
      evaluation.noSignShare > 0.25
    )
      blockers.push("NO SIGN must make up 1% to 25% of the training data");
    if (
      !Number.isInteger(evaluation.noSignSampleCount) ||
      evaluation.noSignSampleCount < 10
    )
      blockers.push("training data needs at least 10 NO SIGN samples");
    if (evaluation.evaluationScope !== "signer-independent")
      blockers.push("evaluation must use leave-one-signer-out splits");
    if (config.testMode) blockers.push("marked as a pipeline test");
    if (
      Number.isFinite(evaluation.testAccuracy) &&
      Number.isFinite(evaluation.minimumAccuracy) &&
      evaluation.testAccuracy < evaluation.minimumAccuracy
    )
      blockers.push(
        "test accuracy is " +
          Math.round(evaluation.testAccuracy * 100) +
          "% (" +
          Math.round(evaluation.minimumAccuracy * 100) +
          "% required)",
      );
    if (evaluation.deploymentEligible !== true)
      blockers.push(
        "evaluation is not marked eligible for the SignAR validation profile",
      );
    if (!config.trained) blockers.push("manifest has trained:false");
    const labels = Array.isArray(config.labels) ? config.labels : [],
      signerCounts = config.distinctSignersPerLabel,
      signerMinimum = config.minimumSignersPerLabel,
      signerMinimumValid =
        Number.isInteger(signerMinimum) && signerMinimum >= 1;
    const classRecall = evaluation.classRecall,
      minimumClassRecall = evaluation.minimumClassRecall;
    if (
      !Number.isFinite(minimumClassRecall) ||
      minimumClassRecall < MINIMUM_CLASS_RECALL
    )
      blockers.push(
        "manifest does not meet the " +
          Math.round(MINIMUM_CLASS_RECALL * 100) +
          "% per-label recall requirement",
      );
    else if (
      !classRecall ||
      typeof classRecall !== "object" ||
      Array.isArray(classRecall) ||
      labels.some(
        (label) =>
          label !== NO_SIGN_LABEL &&
          (!Number.isFinite(classRecall[label]) ||
            classRecall[label] < minimumClassRecall),
      )
    )
      blockers.push(
        "one or more labels have held-out recall below the required threshold",
      );
    const stress = evaluation.stressTestsCorrectOrRejected;
    if (
      !stress ||
      typeof stress !== "object" ||
      Object.values(stress).length === 0 ||
      Object.values(stress).some(
        (score) => !Number.isFinite(score) || score < 0.9,
      )
    )
      blockers.push(
        "one or more stress checks are below the required threshold",
      );
    if (!signerMinimumValid)
      blockers.push(
        "manifest does not declare a minimum of at least one signer per label",
      );
    if (
      !signerCounts ||
      typeof signerCounts !== "object" ||
      Array.isArray(signerCounts) ||
      !labels.length
    )
      blockers.push("manifest is missing signer coverage for each label");
    else if (
      signerMinimumValid &&
      labels.some(
        (label) =>
          !Number.isInteger(signerCounts[label]) ||
          signerCounts[label] < signerMinimum,
      )
    )
      blockers.push(
        "every sign label needs at least " + signerMinimum + " recorded signer",
      );
    return [...new Set(blockers)];
  }
  async function loadModel() {
    if (modelLoadPromise) return modelLoadPromise;
    modelLoadPromise = (async () => {
      try {
        modelLoading = true;
        modelReady = false;
        updateModelLoadingStatus("Reading the installed model manifest…");
        showNoSign();
        modelConfig = await withTimeout(
          loadModelManifest(),
          MODEL_REQUEST_TIMEOUT_MS,
          "Reading the model manifest",
        );
        updateModelLoadingStatus("Checking the model metadata and eligibility…");
        const metadata = await withTimeout(
          loadModelMetadata(modelConfig),
          MODEL_REQUEST_TIMEOUT_MS,
          "Reading the model metadata",
        );
        if (
          JSON.stringify(metadata.labels) !==
            JSON.stringify(modelConfig.labels) ||
          metadata.sequence_length !== modelConfig.sequenceLength ||
          metadata.feature_count !== modelConfig.featureDimension ||
          metadata.feature_version !== modelConfig.featureVersion ||
          metadata.reject_label !== modelConfig.rejectLabel
        )
          throw new Error(
            "Model metadata does not match the vocabulary labels or input dimensions",
          );
        modelConfig.labels = metadata.labels;
        modelConfig.sequenceLength = metadata.sequence_length;
        modelConfig.featureDimension = metadata.feature_count;
        const declaredLabelMetadata = modelConfig.labelMetadata || {};
        if (!declaredLabelMetadata || typeof declaredLabelMetadata !== 'object' || Array.isArray(declaredLabelMetadata))
          throw new Error('Manifest labelMetadata must be an object keyed by sign label.');
        modelConfig.labelMetadata = Object.fromEntries(modelConfig.labels.map((label) => {
          const normalizedLabel = String(label).trim().toUpperCase();
          const declared = declaredLabelMetadata[normalizedLabel] || {};
          const unitType = normalizedLabel === NO_SIGN_LABEL
            ? 'background'
            : declared.unitType || (/^[A-Z]$/.test(normalizedLabel) ? 'fingerspelling' : 'word');
          const motionType = normalizedLabel === NO_SIGN_LABEL
            ? 'static'
            : unitType === 'word' || /^[JZ]$/.test(normalizedLabel)
              ? 'dynamic'
              : (declared.motionType || 'static');
          if (!['background', 'fingerspelling', 'word', 'phrase'].includes(unitType) ||
              !['static', 'dynamic'].includes(motionType) ||
              ((unitType === 'background') !== (normalizedLabel === NO_SIGN_LABEL)))
            throw new Error('Manifest contains invalid sign-unit or motion metadata for ' + label + '.');
          return [normalizedLabel, {
            unitType,
            motionType: unitType === 'background' ? 'static' : motionType,
            displayText: String(declared.displayText || '').trim(),
          }];
        }));
        const modelSchema = PIPELINE.getSchema(modelConfig.featureVersion);
        if (!modelSchema)
          throw new Error("The model declares an unsupported landmark feature schema: " + modelConfig.featureVersion);
        if (modelConfig.sequenceLength !== modelSchema.sequenceLength || modelConfig.featureDimension !== modelSchema.dimension)
          throw new Error("Vocabulary input-shape contract mismatch");
        const schemaChanged = activeSchema.version !== modelSchema.version;
        activeSchema = modelSchema;
        SEQUENCE_LENGTH = modelSchema.sequenceLength;
        FEATURE_DIMENSION = modelSchema.dimension;
        MAX_FEATURE_FRAMES = modelSchema.windowFrames;
        if (schemaChanged) resetRecognitionSegment();
        const blockers = candidateBlockers(modelConfig);
        if (blockers.length || !modelConfig.modelPath) {
          modelReady = false;
          modelLoading = false;
          const reason = blockers.length
            ? "Candidate is not active: " + blockers.join("; ") + ". "
            : "Candidate is not marked as deployable. ";
          modelStatus.textContent = modelConfig.modelPath
            ? "Interpreter read ./model/vocabulary.json. " +
              reason +
              "A model file in the folder does not enable predictions until the candidate passes validation."
            : "No model is installed. Landmark tracking works, but sign recognition needs an eligible model with at least two sign labels.";
          showNoSign();
          return;
        }
        if (
          !Array.isArray(modelConfig.labels) ||
          modelConfig.labels.length < 2 ||
          modelConfig.labels.some(
            (label) => typeof label !== "string" || !label.trim(),
          ) ||
          new Set(modelConfig.labels).size !== modelConfig.labels.length
        )
          throw new Error(
            "A deployable model must contain at least two unique sign labels",
          );
        if (
          modelConfig.outputClasses !== undefined &&
          modelConfig.outputClasses !== modelConfig.labels.length
        )
          throw new Error(
            "Manifest output class count does not match its sign labels",
          );
        updateModelLoadingStatus("Loading the TensorFlow.js inference library…");
        await withTimeout(
          loadTensorFlowLibrary(),
          EXTERNAL_SCRIPT_TIMEOUT_MS * TENSORFLOW_BASES.length + 2000,
          "Loading TensorFlow.js",
        );
        const modelUrl = new URL(modelConfig.modelPath, window.location.href)
          .href;
        updateModelLoadingStatus("Checking available browser inference backends…");
        modelConfig.backendBenchmark = await withTimeout(
          benchmarkTensorFlowBackends(modelUrl, modelConfig, activeSchema),
          MODEL_BENCHMARK_TIMEOUT_MS,
          "Checking browser inference backends",
        );
        updateModelLoadingStatus("Loading the FSL model weights…");
        model = await withTimeout(
          window.tf.loadLayersModel(modelUrl),
          MODEL_LOAD_TIMEOUT_MS,
          "Loading the FSL model weights",
        );
        try {
          validateModelShape(model, modelConfig, activeSchema);
          await withTimeout(
            warmupModel(model, activeSchema),
            MODEL_LOAD_TIMEOUT_MS,
            "Preparing the FSL model",
          );
        } catch (error) {
          model.dispose();
          model = null;
          throw error;
        }
        modelReady = true;
        modelLoading = false;
        const scopeNote =
          modelConfig.evaluation?.evaluationScope === "sample-level"
            ? " Evaluation used samples from the available signer(s); new-signer performance is unknown."
            : "";
        const noSignNote = modelConfig.labels.some(
          (label) => label.trim().toUpperCase() === NO_SIGN_LABEL,
        )
          ? " A trained no-sign class is available."
          : " This model has no no-sign class, so idle or unfamiliar input can be labeled as a sign.";
        const liveCameraSamples =
          modelConfig.evaluation?.liveCameraSamplesPerLabel || {};
        const unverifiedCameraLabels = modelConfig.labels.filter(
          (label) => label !== NO_SIGN_LABEL && !(liveCameraSamples[label] > 0),
        );
        const cameraCoverageNote = unverifiedCameraLabels.length
          ? " Optional live-camera samples are absent for " +
            unverifiedCameraLabels.slice(0, 8).join(", ") +
            (unverifiedCameraLabels.length > 8 ? ", and " + (unverifiedCameraLabels.length - 8) + " more" : "") +
            "; camera performance for these labels is unverified, but this does not affect model eligibility."
          : "";
        const backendLabel = {
          webgpu: "WebGPU",
          webgl: "WebGL",
          wasm: "WebAssembly",
          cpu: "CPU",
        }[modelConfig.backendBenchmark.selected] || modelConfig.backendBenchmark.selected;
        modelStatus.textContent =
          "SignAR model loaded: " +
          modelConfig.modelPath +
          ". Fastest measured decoder: " + backendLabel +
          ". Evaluation reflects the recorded dataset; results can differ across signers, cameras, and environments." +
          scopeNote +
          noSignNote +
          cameraCoverageNote;
        showNoSign();
      } catch (error) {
        modelReady = false;
        modelLoading = false;
        modelLoadPromise = null;
        console.info("FSL model unavailable:", error.message);
        modelStatus.textContent =
          "Model unavailable: " +
          error.message +
          " Sign recognition is disabled; landmark tracking remains available.";
        showNoSign();
      }
    })();
    return modelLoadPromise;
  }

  function pushFeatureFrame(frame, time) {
    const lastQueuedTime = segmentFrameTimes.at(-1);
    if (
      lastQueuedTime !== undefined &&
      time - lastQueuedTime < FEATURE_SAMPLE_INTERVAL_MS
    )
      return;
    segmentFrames.push(frame);
    segmentFrameTimes.push(time);
    while (segmentFrames.length > MAX_FEATURE_FRAMES) {
      segmentFrames.shift();
      segmentFrameTimes.shift();
    }
  }

  function saveDebugFeatureWindow(handSlot) {
    const now = performance.now();
    if (
      !DEBUG_TELEMETRY ||
      segmentFrames.length < MINIMUM_BUFFER_FRAMES ||
      now - lastFeatureExportAt < 500
    )
      return;
    lastFeatureExportAt = now;
    exportedFeatureWindows.push({
      capturedAt: new Date().toISOString(),
      label: "",
      handSlot,
      handSource: lastHandSource,
      facePresent: lastFacePresent,
      posePresent: lastPosePresent,
      features: PIPELINE.selectRecentSequenceWindow(
        segmentFrames,
        SEQUENCE_LENGTH,
        activeSchema.windowFrames,
      ),
    });
    if (exportedFeatureWindows.length > 100) exportedFeatureWindows.shift();
  }

  function renderStreamingPrediction() {
    if (!modelReady || !model) {
      showNoSign(
        "Landmark tracking works; a validated model for the active holistic feature schema is required for sign recognition.",
      );
      return;
    }
    const prediction = predictionResult;
    const multiPersonNote = multiplePeopleDetected
      ? " Multiple people are in view; decoding continues from the current holistic landmark track."
      : "";
    if (!prediction || !prediction.best) {
      showNoSign(
        (signerVisible
          ? "Waiting for a stable sign."
          : "Keep your hands, face, or upper body in frame when ready.") + multiPersonNote,
      );
      return;
    }
    const top = prediction.ranked.slice(0, 3);
    topKScores.replaceChildren(
      ...top.map((item) => {
        const entry = document.createElement("li");
        entry.textContent = `${item.label} | EMA ${item.score.toFixed(3)} (${item.confidence.toFixed(1)}%)`;
        return entry;
      }),
    );
    if (
      prediction.release ||
      prediction.best.label.toUpperCase() === NO_SIGN_LABEL
    ) {
      showNoSign(
        "No sign unit now. A short pause closes the current segment; keep signing to continue the phrase." + multiPersonNote,
      );
      return;
    }
    const { best, margin, accepted, streak, requiredFrames } = prediction;
    const signMetadata = labelMetadataFor(best.label);
    const unitName = {
      fingerspelling: 'Fingerspelling letter',
      word: 'Word sign',
      phrase: 'Phrase sign',
    }[signMetadata.unitType] || 'Sign';
    const segmentActive = streamSegmenter.activeLabel === best.label;
    detected.textContent = accepted
      ? segmentActive
        ? best.label
        : "SEGMENTING " + best.label
      : "POSSIBLE " + best.label;
    confidenceValue.textContent = best.confidence.toFixed(1) + "%";
    confidenceBar.style.width = Math.round(best.confidence) + "%";
    translation.textContent = (accepted
      ? segmentActive
        ? `${best.label} is active; the stream closes it at the next stable sign or pause.`
        : `${best.label} is being checked before it joins the continuous phrase.`
      : `Best match ${best.label}; confidence or top-two margin is below the decode threshold.`) + multiPersonNote;
    typePill.textContent = segmentActive
      ? unitName + " · continuous phrase"
      : (signMetadata.motionType === 'dynamic' ? "Dynamic " : "Static ") + unitName.toLowerCase();
    arSign.textContent = best.label;
    arMeaning.textContent = signMetadata.displayText;
    overlay.classList.toggle("active", accepted);
    setState(
      accepted
        ? segmentActive
          ? "IN PHRASE"
          : "VERIFYING"
        : "LOW CONFIDENCE",
      accepted ? "verifying" : "",
    );
    lastPredictionDecision = accepted
      ? `WAIT ${best.label}: ${Math.min(streak, requiredFrames)}/${requiredFrames} consecutive frames, margin ${margin.toFixed(1)} pp.`
      : `REJECT ${best.label}: confidence ${best.confidence.toFixed(1)}%, margin ${margin.toFixed(1)} pp.`;
  }

  function applyTemperatureScaling(scores, temperature) {
    if (
      !Number.isFinite(temperature) ||
      temperature <= 0 ||
      Math.abs(temperature - 1) < 1e-6
    )
      return Array.from(scores);
    const logits = Array.from(
      scores,
      (score) => Math.log(Math.max(Number(score), 1e-8)) / temperature,
    );
    const maxLogit = Math.max(...logits);
    const scaled = logits.map((value) => Math.exp(value - maxLogit));
    const total = scaled.reduce((sum, value) => sum + value, 0);
    return scaled.map((value) => value / total);
  }

  async function runStreamingInference() {
    const now = performance.now();
    if (
      !modelReady ||
      !model ||
      inferenceBusy ||
      segmentFrames.length < MINIMUM_BUFFER_FRAMES ||
      now - lastInferenceAt < INFERENCE_INTERVAL_MS
    )
      return;
    inferenceBusy = true;
    lastInferenceAt = now;
    const generation = recognitionGeneration;
    const started = performance.now();
    let input = null;
    let output = null;
    try {
      const frames = PIPELINE.selectRecentSequenceWindow(
        segmentFrames,
        SEQUENCE_LENGTH,
        activeSchema.windowFrames,
      );
      const streamInputValues = new Float32Array(
        SEQUENCE_LENGTH * FEATURE_DIMENSION,
      );
      for (let index = 0; index < frames.length; index++)
        streamInputValues.set(frames[index], index * FEATURE_DIMENSION);
      input = window.tf.tensor3d(streamInputValues, [
        1,
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION,
      ]);
      output = model.predict(input);
      const tensors = Array.isArray(output) ? output : [output];
      const tensor = tensors[0];
      if (
        !tensor ||
        tensor.shape.length !== 2 ||
        tensor.shape[0] !== 1 ||
        tensor.shape[1] !== modelConfig.labels.length
      )
        throw new Error(
          "Model prediction shape does not match the manifest label count.",
        );
      const probabilities = await tensor.data();
      if (generation !== recognitionGeneration) return;
      const temperature =
        modelConfig.evaluation?.temperatureScaling?.finalTemperature ?? 1;
      const calibratedProbabilities = applyTemperatureScaling(
        probabilities,
        temperature,
      );
      const predictionAt = performance.now();
      predictionResult = predictionHistory.update(
        calibratedProbabilities,
        modelConfig.labels,
        {
          rejectLabel: modelConfig.rejectLabel || NO_SIGN_LABEL,
          minimumConfidencePercent,
          minimumMarginPercent:
            PIPELINE.config.recognition.minimumMarginPercent,
          now: predictionAt,
        },
      );
      if (predictionResult.accepted) finishNoSignPause(predictionAt);
      else beginNoSignPause(predictionAt);
      const segmentation = streamSegmenter.update(
        predictionResult.accepted ? predictionResult.label : null,
        predictionResult.accepted ? predictionResult.confidence : 0,
        predictionAt,
      );
      for (const event of segmentation.events) addRecognizedUnit(event.label, event.confidence);
      if (predictionResult.release) {
        lastPredictionDecision =
          "NO SIGN: waiting for the phrase boundary pause.";
      } else if (!predictionResult.accepted) {
        lastPredictionDecision = `REJECT ${predictionResult.label}: confidence ${predictionResult.best.confidence.toFixed(1)}%, margin ${predictionResult.margin.toFixed(1)} pp.`;
      } else if (segmentation.events.length) {
        lastPredictionDecision = `SEGMENT ${segmentation.events.map((event) => event.label).join(", ")}: ${segmentation.events.at(-1).boundary} boundary.`;
      } else {
        lastPredictionDecision = `STREAM ${predictionResult.label}: ${segmentation.activeWindows || 0} stable windows; waiting for a sign change or pause.`;
      }
      if (DEBUG_TELEMETRY && performance.now() - lastDebugTelemetryAt >= 500) {
        lastDebugTelemetryAt = performance.now();
        console.debug("[SignAR] inference", {
          time: new Date().toISOString(),
          handSlots: {
            left: PIPELINE.hasUsableHand(lastTrackerResults?.leftHandLandmarks),
            right: PIPELINE.hasUsableHand(
              lastTrackerResults?.rightHandLandmarks,
            ),
          },
          selectedHandSlot: lastSelectedHandSlot,
          facePresent: lastFacePresent,
          posePresent: lastPosePresent,
          handSource: lastHandSource,
          top3: predictionResult.ranked
            .slice(0, 3)
            .map(({ label, score }) => ({ label, score })),
          decision: lastPredictionDecision,
        });
      }
    } catch (error) {
      console.error("Inference error:", error);
      resetRecognitionSegment();
      const detail = String(
        error?.message || error || "Unknown inference error",
      )
        .replace(/\s+/g, " ")
        .slice(0, 220);
      lastPredictionDecision = "INFERENCE ERROR: " + detail;
      translation.textContent =
        "Nagkaroon ng error sa model inference: " + detail;
      setState("MODEL ERROR");
    } finally {
      if (input) input.dispose();
      if (output)
        (Array.isArray(output) ? output : [output]).forEach((tensor) =>
          tensor?.dispose?.(),
        );
      const elapsed = performance.now() - started;
      recordStageTiming("inference", elapsed);
      inferenceCount++;
      averageInferenceMs += (elapsed - averageInferenceMs) / inferenceCount;
      inferenceBusy = false;
    }
  }

  let lastTrackerResults = null;
  function onHandResults(results) {
    const now = performance.now();
    const segmentationTick = streamSegmenter.tick(now);
    for (const event of segmentationTick.events) addRecognizedUnit(event.label, event.confidence);
    lastTrackerResults = results;
    const gap = lastTrackingResultAt ? now - lastTrackingResultAt : 0;
    if (gap > PIPELINE.config.sequence.trackingResetMs) {
      gapResetCount++;
      lastGapResetReason = `tracking loss ${Math.round(gap)} ms`;
      resetRecognitionSegment();
    }
    lastTrackingResultAt = now;
    recordStageTiming("landmarks", results.signarLandmarkMs || 0);
    drawTrackerLandmarks(results);
    updateTrackingStatus(results);
    lastHandSource = results.signarHandSource || "unknown";
    lastFacePresent =
      detectedFaceBoxes.length > 0 || !!results.faceLandmarks?.length;
    lastPosePresent = !!results.poseLandmarks?.length;
    const selected = PIPELINE.chooseHand(
      results,
      video.videoWidth || PIPELINE.config.camera.width,
      video.videoHeight || PIPELINE.config.camera.height,
    );
    const featureStarted = performance.now();
    const feature = PIPELINE.extractForVersion(
      results,
      activeSchema.version,
      video.videoWidth || PIPELINE.config.camera.width,
      video.videoHeight || PIPELINE.config.camera.height,
    );
    recordStageTiming("features", performance.now() - featureStarted);
    lastSelectedHandSlot = selected?.slot || "none";
    lastValidHandPoints = selected?.validPoints || 0;
    const hasTrackedCue = activeSchema.version === PIPELINE.config.features.version
      ? PIPELINE.hasAnyFeatureRegion(feature)
      : !!feature;
    if (!feature) {
      signerVisible = false;
      const release = streamSegmenter.update(null, 0, now);
      for (const event of release.events) addRecognizedUnit(event.label, event.confidence);
      const trackingLossMs = lastFeatureAt ? now - lastFeatureAt : 0;
      if (trackingLossMs > PIPELINE.config.sequence.trackingResetMs) {
        gapResetCount++;
        lastGapResetReason = `hand tracking loss ${Math.round(trackingLossMs)} ms`;
        resetRecognitionSegment();
      } else if (
        trackingLossMs > PIPELINE.config.sequence.interpolationLimitMs &&
        !trackingLossReleased
      ) {
        beginNoSignPause(lastFeatureAt || now);
        predictionHistory.release();
        predictionResult = null;
        segmentFrames = [];
        segmentFrameTimes = [];
        previousFeatures = null;
        trackingLossReleased = true;
        updateTemporalDiagnostics();
      }
      guide?.classList.remove("is-hidden");
      return;
    }
    signerVisible = hasTrackedCue;
    guide?.classList.toggle("is-hidden", hasTrackedCue);
    const timestamp = now;
    if (previousFeatures === null) finishNoSignPause(timestamp);
    if (previousFeatures && lastFeatureAt) {
      const frameGap = timestamp - lastFeatureAt;
      if (
        frameGap > FEATURE_SAMPLE_INTERVAL_MS &&
        frameGap < PIPELINE.config.sequence.interpolationLimitMs
      ) {
        const lastQueuedTime =
          segmentFrameTimes.at(-1) ?? lastFeatureAt;
        for (
          let sampleTime = lastQueuedTime + FEATURE_SAMPLE_INTERVAL_MS;
          sampleTime < timestamp;
          sampleTime += FEATURE_SAMPLE_INTERVAL_MS
        ) {
          pushFeatureFrame(
            PIPELINE.interpolateFeatureFrame(
              previousFeatures,
              feature,
              (sampleTime - lastFeatureAt) / frameGap,
            ),
            sampleTime,
          );
        }
      }
    }
    pushFeatureFrame(feature, timestamp);
    previousFeatures = feature;
    lastFeatureAt = timestamp;
    trackingLossReleased = false;
    saveDebugFeatureWindow(selected?.slot || "none");
    if (modelReady) void runStreamingInference();
  }

  function onFaceDetectionResults(r) {
    if (faceDetectionGeneration !== cameraStartGeneration) return;
    detectedFaceBoxes = (r.detections || [])
      .map((d) => d.boundingBox)
      .filter(
        (box) =>
          box &&
          Number.isFinite(box.xCenter) &&
          Number.isFinite(box.yCenter) &&
          Number.isFinite(box.width) &&
          Number.isFinite(box.height) &&
          box.width > 0 &&
          box.height > 0,
      );
    if (detectedFaceBoxes.length >= 2) {
      multiplePeopleDetected = true;
      singlePersonFrames = 0;
    } else {
      singlePersonFrames++;
      if (singlePersonFrames >= 3) multiplePeopleDetected = false;
    }
  }
  function checkFaceDetection(image, time, generation) {
    if (
      !faceDetection ||
      faceDetectionBusy ||
      time - lastFaceDetectionAt < PERSON_CHECK_INTERVAL_MS
    )
      return;
    const detector = faceDetection;
    faceDetectionBusy = true;
    faceDetectionGeneration = generation;
    lastFaceDetectionAt = time;
    Promise.resolve()
      .then(() => detector.send({ image }))
      .catch((error) => {
        if (generation !== cameraStartGeneration) return;
        console.error("Multi-person detection failed:", error);
        faceDetection = null;
        faceDetectionInitPromise = null;
        multiplePeopleDetected = false;
        detectedFaceBoxes = [];
      })
      .finally(() => {
        faceDetectionBusy = false;
      });
  }
  async function loadFaceDetectionLibrary() {
    if (typeof window.FaceDetection === "function") return;
    if (!faceDetectionLoadPromise)
      faceDetectionLoadPromise = (async () => {
        const failures = [];
        for (const base of FACE_DETECTION_BASES) {
          try {
            await new Promise((resolve, reject) => {
              const script = document.createElement("script");
              script.src = base + "/face_detection.js";
              script.crossOrigin = "anonymous";
              script.onload = () =>
                typeof window.FaceDetection === "function"
                  ? resolve()
                  : reject(
                      new Error(
                        "The script loaded without defining FaceDetection.",
                      ),
                    );
              script.onerror = () =>
                reject(new Error("The script request failed."));
              document.head.appendChild(script);
            });
            faceDetectionBase = base;
            return;
          } catch (error) {
            failures.push(base + ": " + error.message);
          }
        }
        throw new Error(
          "Unable to load MediaPipe Face Detection. Check your internet connection or content blocker. " +
            failures.join(" "),
        );
      })();
    try {
      await faceDetectionLoadPromise;
    } catch (error) {
      faceDetectionLoadPromise = null;
      throw error;
    }
  }
  async function initOptionalRecognition() {
    if (faceDetection) return;
    if (!faceDetectionInitPromise)
      faceDetectionInitPromise = (async () => {
        await loadFaceDetectionLibrary();
        const detector = new window.FaceDetection({
          locateFile: (file) => faceDetectionBase + "/" + file,
        });
        detector.setOptions({
          ...PIPELINE.config.tracking.faceDetectionOptions,
        });
        detector.onResults(onFaceDetectionResults);
        if (typeof detector.initialize === "function")
          await tracker.serializeInitialization(() => detector.initialize());
        faceDetection = detector;
      })();
    try {
      await faceDetectionInitPromise;
    } catch (error) {
      faceDetection = null;
      faceDetectionInitPromise = null;
      console.error("Multi-person detection unavailable:", error);
    }
  }
  function cameraError(e) {
    if (!window.isSecureContext)
      return "Camera access requires HTTPS or localhost.";
    if (e?.name === "NotAllowedError")
      return "Pinayagan ang camera access? Payagan ang camera sa browser settings.";
    if (e?.name === "NotFoundError" || e?.name === "OverconstrainedError")
      return "Walang compatible na camera sa device.";
    if (e?.name === "NotReadableError")
      return "Maaaring ginagamit ng ibang app ang camera.";
    return "Camera error: " + (e?.message || e?.name || "unknown");
  }
  function removeCameraTrackListeners() {
    cameraTrackListeners.forEach(({ track, listener }) =>
      track.removeEventListener("ended", listener),
    );
    cameraTrackListeners = [];
  }
  async function handleCameraInterruption(generation) {
    if (generation !== cameraStartGeneration) return;
    await stopCamera(false);
    cameraStatus("Camera disconnected");
    showNoSign(
      "The camera stream ended. Reconnect the camera and start it again.",
    );
    toastMessage(
      "Camera disconnected. Start the camera again to resume tracking.",
    );
  }
  async function startCamera() {
    if (cameraStarting || stream) return;
    if (!window.isSecureContext) {
      toastMessage("HTTPS required");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      toastMessage("Camera API unavailable");
      return;
    }
    cameraStarting = true;
    trackerReady = false;
    const generation = ++cameraStartGeneration;
    startBtn.disabled = true;
    switchBtn.disabled = true;
    try {
      cameraStatus("Opening camera...");
      const trackerReadyPromise = tracker.initialize().then(
        () => null,
        (error) => error,
      );
      const openedStream = await tracker.openCamera(facingMode);
      if (generation !== cameraStartGeneration) {
        openedStream.getTracks().forEach((track) => track.stop());
        return;
      }
      stream = openedStream;
      cameraTrackListeners = stream.getVideoTracks().map((track) => {
        const listener = () => {
          void handleCameraInterruption(generation);
        };
        track.addEventListener("ended", listener, { once: true });
        return { track, listener };
      });
      video.srcObject = stream;
      await video.play();
      if (generation !== cameraStartGeneration || !stream) return;
      multiplePeopleDetected = false;
      singlePersonFrames = 0;
      detectedFaceBoxes = [];
      lastFaceDetectionAt = -PERSON_CHECK_INTERVAL_MS;
      const actual = stream.getVideoTracks()[0]?.getSettings?.().facingMode;
      if (actual) facingMode = actual;
      video.classList.toggle("mirror", facingMode === "user");
      landmarkCanvas.classList.toggle("mirror", facingMode === "user");
      clearTrackerLandmarks();
      video.classList.add("active");
      placeholder.style.display = "none";
      liveBadge.textContent = "CAMERA ON";
      liveBadge.classList.add("show");
      trackingBadge.classList.add("show");
      startBtn.disabled = true;
      stopBtn.disabled = false;
      switchBtn.textContent =
        facingMode === "environment" ? "Use front camera" : "Use back camera";
      cameraStatus("Camera on - starting tracking", true);
      const trackerError = await trackerReadyPromise;
      if (generation !== cameraStartGeneration || !stream) return;
      if (trackerError) throw trackerError;
      trackerReady = true;
      liveBadge.textContent = "LIVE TRACKING";
      cameraStatus("Camera on", true);
      lastVideoTime = -1;
      lastFrameAt = 0;
      rafId = requestAnimationFrame((time) => frameLoop(time, generation));
      void initOptionalRecognition();
      toastMessage(
        modelReady
          ? "Camera ready. Trained decoder loaded."
          : "Camera preview is ready; hand tracking is active. Sign predictions require a validated model.",
      );
    } catch (e) {
      if (generation !== cameraStartGeneration) return;
      console.error(e);
      resetRecognitionSegment();
      signerVisible = false;
      removeCameraTrackListeners();
      if (stream) stream.getTracks().forEach((t) => t.stop());
      stream = null;
      video.srcObject = null;
      video.classList.remove("active");
      clearTrackerLandmarks();
      liveBadge.textContent = "LIVE TRACKING";
      liveBadge.classList.remove("show");
      trackingBadge.classList.remove("show");
      trackerReady = false;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      showNoSign(cameraError(e));
      setState("CAMERA ERROR", "verifying");
      cameraStatus("Camera unavailable");
      toastMessage(cameraError(e));
    } finally {
      if (generation === cameraStartGeneration) {
        cameraStarting = false;
        startBtn.disabled = !!stream;
        switchBtn.disabled = false;
      }
    }
  }
  async function frameLoop(time, generation) {
    if (!stream || generation !== cameraStartGeneration) return;
    if (
      !processing &&
      video.readyState >= 2 &&
      video.currentTime !== lastVideoTime &&
      time - lastFrameAt >= processEvery
    ) {
      processing = true;
      lastFrameAt = time;
      lastVideoTime = video.currentTime;
      const started = performance.now();
      try {
        const sourceW = video.videoWidth || PIPELINE.config.camera.width,
          sourceH = video.videoHeight || PIPELINE.config.camera.height,
          { width, height } = PIPELINE.trackingFrameSize(
            sourceW,
            sourceH,
            trackingState,
          );
        if (processCanvas.width !== width) processCanvas.width = width;
        if (processCanvas.height !== height) processCanvas.height = height;
        const captureStarted = performance.now();
        processCtx.drawImage(video, 0, 0, width, height);
        recordStageTiming("capture", performance.now() - captureStarted);
        if (
          faceDetection &&
          !faceDetectionBusy &&
          !inferenceBusy &&
          time - lastFaceDetectionAt >= PERSON_CHECK_INTERVAL_MS
        ) {
          const faceScale = Math.min(
              1,
              PERSON_CHECK_DIMENSION / Math.max(sourceW, sourceH),
            ),
            faceWidth = Math.max(1, Math.round(sourceW * faceScale)),
            faceHeight = Math.max(1, Math.round(sourceH * faceScale));
          if (personCheckCanvas.width !== faceWidth)
            personCheckCanvas.width = faceWidth;
          if (personCheckCanvas.height !== faceHeight)
            personCheckCanvas.height = faceHeight;
          personCheckCtx.drawImage(video, 0, 0, faceWidth, faceHeight);
          checkFaceDetection(personCheckCanvas, time, generation);
        }
        if (generation !== cameraStartGeneration) return;
        const results = await tracker.processFrame(processCanvas);
        if (generation === cameraStartGeneration) {
          onHandResults(results);
          const completedAt = performance.now();
          if (lastCompletedFrameAt) {
            const instantaneous = 1000 / (completedAt - lastCompletedFrameAt);
            trackingFps = trackingFps
              ? trackingFps * 0.8 + instantaneous * 0.2
              : instantaneous;
          }
          lastCompletedFrameAt = completedAt;
          if (completedAt - lastMetricsAt >= 500) {
            updateDiagnostics();
            lastMetricsAt = completedAt;
          }
        }
      } catch (e) {
        if (generation === cameraStartGeneration) {
          console.error(e);
          resetRecognitionSegment();
          signerVisible = false;
          showNoSign(
            "Tracking failed. Predictions are paused until fresh landmark frames are available.",
          );
          setState("TRACKING ERROR", "verifying");
        }
      } finally {
        const ms = performance.now() - started;
        if (generation === cameraStartGeneration) adaptTrackingResolution(ms);
        processing = false;
      }
    }
    if (generation === cameraStartGeneration && stream)
      rafId = requestAnimationFrame((nextTime) =>
        frameLoop(nextTime, generation),
      );
  }
  async function stopCamera(show = true) {
    const finalSegments = streamSegmenter.flush(performance.now(), "camera-stop");
    for (const event of finalSegments.events) addRecognizedUnit(event.label, event.confidence);
    cameraStartGeneration++;
    cameraStarting = false;
    trackerReady = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    removeCameraTrackListeners();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
    video.classList.remove("active");
    clearTrackerLandmarks();
    placeholder.style.display = "flex";
    liveBadge.textContent = "LIVE TRACKING";
    liveBadge.classList.remove("show");
    trackingBadge.classList.remove("show", "detected");
    startBtn.disabled = false;
    stopBtn.disabled = true;
    switchBtn.disabled = false;
    resetRecognitionSegment();
    lastTrackingResultAt = 0;
    lastCompletedFrameAt = 0;
    lastMetricsAt = 0;
    trackingFps = 0;
    signerVisible = false;
    guide?.classList.remove("is-hidden");
    updateDiagnostics();
    if (show) {
      cameraStatus("Camera off");
      showNoSign("Walang kasalukuyang senyas.");
    }
  }
  async function switchCamera() {
    const desired = facingMode === "environment" ? "user" : "environment";
    facingMode = desired;
    if (stream) {
      await stopCamera(false);
      await startCamera();
    } else {
      video.classList.toggle("mirror", facingMode === "user");
      landmarkCanvas.classList.toggle("mirror", facingMode === "user");
      switchBtn.textContent =
        facingMode === "environment" ? "Use front camera" : "Use back camera";
    }
  }
  navButtons.forEach((btn) =>
    btn.addEventListener("click", () => {
      navButtons.forEach((b) => b.classList.toggle("active", b === btn));
      sections.forEach((s) =>
        s.classList.toggle("active-section", s.id === btn.dataset.target),
      );
    }),
  );
  startBtn.addEventListener("click", startCamera);
  scoreThresholdInput.addEventListener("input", () => {
    minimumConfidencePercent = Number(scoreThresholdInput.value);
    scoreThresholdValue.textContent = minimumConfidencePercent + "%";
    predictionHistory.clear({ release: false });
    predictionResult = null;
    streamSegmenter.clear();
    lastInferenceAt = 0;
    lastPredictionDecision =
      "Vote history cleared after changing the score threshold.";
    updateTemporalDiagnostics();
  });
  stopBtn.addEventListener("click", () => stopCamera(true));
  switchBtn.addEventListener("click", switchCamera);
  toggleLandmarksBtn.addEventListener("click", () =>
    setLandmarksVisible(!landmarksVisible),
  );
  fullscreenBtn.addEventListener("click", async () => {
    try {
      if (!document.fullscreenElement) await cameraStage.requestFullscreen();
      else await document.exitFullscreen();
    } catch {
      toastMessage("Fullscreen is not available in this browser.");
    }
  });
  deleteBtn.addEventListener("click", () => {
    while (sentence.at(-1) === " ") sentence.pop();
    sentence.pop();
    while (sentence.at(-1) === " ") sentence.pop();
    recognizedSigns.pop();
    sentenceRevision++;
    updateSentence();
    stopSpeech();
  });
  clearBtn.addEventListener("click", () => {
    sentence = [];
    recognizedSigns = [];
    sentenceRevision++;
    updateSentence();
    stopSpeech();
  });
  copyBtn.addEventListener("click", async () => {
    if (!sentence.length) return;
    try {
      await navigator.clipboard.writeText(sentence.join(""));
      toastMessage("Na-kopya ang decoded sentence.");
    } catch {
      const temporary = document.createElement("textarea");
      temporary.value = sentence.join("");
      temporary.setAttribute("readonly", "");
      temporary.style.position = "fixed";
      temporary.style.opacity = "0";
      document.body.appendChild(temporary);
      temporary.select();
      const copied = document.execCommand("copy");
      temporary.remove();
      toastMessage(
        copied
          ? "Na-kopya ang decoded sentence."
          : "Hindi makopya. Pumili at kopyahin ang text.",
      );
    }
  });
  speakBtn.addEventListener("click", () => {
    stopSpeech();
    speakSentence(true);
  });
  voiceLanguage.addEventListener("change", () => {
    stopSpeech();
    updateSpeechSupport();
  });
  window.addEventListener("beforeunload", () => {
    void stopCamera(false);
    stopSpeech();
  });
  window.addEventListener("pagehide", (event) => {
    if (event.persisted) return;
    if (model) {
      model.dispose();
      model = null;
      modelReady = false;
    }
    if (faceDetection) {
      try {
        Promise.resolve(faceDetection.close?.()).catch((error) =>
          console.warn("Could not close face detector.", error),
        );
      } catch (error) {
        console.warn("Could not close face detector.", error);
      }
    }
    void tracker
      .close()
      .catch((error) =>
        console.warn("Could not close landmark tracker.", error),
      );
  });
  window.addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault();
    installPrompt = e;
    installBtn.hidden = false;
  });
  installBtn.addEventListener("click", async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    await installPrompt.userChoice;
    installPrompt = null;
    installBtn.hidden = true;
  });
  if ("speechSynthesis" in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      chooseFilipinoVoice();
      updateSpeechSupport();
    };
  }
  updateSpeechSupport();
  if ("serviceWorker" in navigator)
    window.addEventListener("load", () =>
      navigator.serviceWorker.register("./sw.js").catch(console.warn),
    );
  switchBtn.textContent = "Use front camera";
  updateSentence();
  modelLoading = true;
  modelStatus.textContent = "Starting decoder model checks…";
  showNoSign();
  void loadModel();
})();
