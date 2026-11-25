/* SignAR FSL decoder foundation. The decoder never invents model predictions.
   Model input uses a versioned hand-only or holistic landmark contract. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const video = $("video");
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
    lastRecognitionUpdateAt = 0,
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
    trackingLossReleased = false;
  let recognitionGeneration = 0;
  let speechTimer = null,
    noSignSince = null,
    pendingPauseMs = 0,
    sentence = [],
    recognizedSigns = [],
    sentenceRevision = 0,
    speechRevision = 0;
  let installPrompt = null,
    voices = [];

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
      : stored.motionType || (/^[JZ]$/.test(normalized) ? 'dynamic' : 'static');
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
    speakBtn.textContent = "Read aloud";
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
      speakBtn.textContent = "Read aloud";
      speakBtn.classList.remove("speaking");
      updateSpeechSupport();
    };
    utterance.onerror = (event) => {
      if (speakingRevision !== speechRevision) return;
      speakBtn.disabled = !sentence.length;
      speakBtn.textContent = "Read aloud";
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
    toastMessage("Added " + label + " to your sentence.");
  }

  function showNoSign(message) {
    const cameraActive = !!stream;
    const noSignMessage = message || (cameraActive ? "Keep your hands and upper body in view." : "Start the camera when you are ready.");
    detected.textContent = modelReady
      ? "No sign yet"
      : modelLoading
        ? "Getting ready"
        : "Unavailable";
    confidenceValue.textContent = "-";
    confidenceBar.style.width = "0%";
    translation.textContent = modelReady
      ? cameraActive
        ? noSignMessage
        : "Recognition is ready. Start the camera and sign when you are ready."
      : modelLoading
        ? "Preparing sign recognition..."
        : modelStatus.textContent;
    typePill.textContent = modelReady
      ? cameraActive ? "Listening" : "Ready"
      : modelLoading ? "Preparing" : "Unavailable";
    setState(
      modelReady
        ? cameraActive ? "Listening" : "Ready"
        : modelLoading ? "Getting ready" : "Unavailable",
      modelLoading ? "verifying" : modelReady ? "" : "locked",
    );
    overlay.classList.remove("active");
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
  }
  function updateTrackingStatus(r) {
    const left = r.leftHandLandmarks || [];
    const right = r.rightHandLandmarks || [];
    const face = !!r.faceLandmarks?.length;
    const pose = !!r.poseLandmarks?.length;
    const hands = Number(PIPELINE.hasUsableHand(left)) + Number(PIPELINE.hasUsableHand(right));
    const hasCues = hands > 0 || face || pose || multiplePeopleDetected;
    const text = multiplePeopleDetected
      ? "More than one person in view"
      : hands > 0
        ? "Hands in view"
        : face || pose
          ? "Move your hands into view"
          : "Looking for a sign";
    if (trackingBadge.textContent !== text) trackingBadge.textContent = text;
    trackingBadge.classList.toggle("detected", hasCues);
    guide?.classList.toggle("is-hidden", hasCues);
  }
  async function warmupModel(loadedModel, schema = activeSchema) {
    const input = window.tf.zeros(PIPELINE.sequenceTensorShape(1, schema));
    let output = null;
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
            await new Promise((resolve, reject) => {
              const script = document.createElement("script");
              script.src = src;
              script.crossOrigin = "anonymous";
              script.onload = () =>
                window.tf
                  ? resolve()
                  : reject(
                      new Error("TensorFlow.js loaded without defining tf."),
                    );
              script.onerror = () =>
                reject(new Error("The script request failed."));
              document.head.appendChild(script);
            });
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
        await new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = source;
          script.crossOrigin = "anonymous";
          script.onload = () =>
            window.tf?.wasm?.setWasmPaths
              ? resolve()
              : reject(
                  new Error(
                    "The WASM backend loaded without registering tf.wasm.",
                  ),
                );
          script.onerror = () =>
            reject(new Error("The backend script request failed."));
          document.head.appendChild(script);
        });
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
            await new Promise((resolve, reject) => {
              const script = document.createElement("script");
              script.src = source;
              script.crossOrigin = "anonymous";
              script.onload = () => resolve();
              script.onerror = () =>
                reject(new Error("The backend script request failed."));
              document.head.appendChild(script);
            });
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
);
    return { selected: fastest.backend, results };
  }

  async function loadModelManifest() {
    if (!modelManifestPromise)
      modelManifestPromise = (async () => {
        const response = await fetch("./model/vocabulary.json", {
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
    const response = await fetch(metadataPath, { cache: "no-store" });
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
        modelStatus.textContent = "Preparing sign recognition...";
        showNoSign();
        modelConfig = await loadModelManifest();
        const metadata = await loadModelMetadata(modelConfig);
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
            : declared.motionType || (/^[JZ]$/.test(normalizedLabel) ? 'dynamic' : 'static');
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
          modelStatus.textContent = "Sign recognition is not ready yet. The camera preview is still available.";
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
        await loadTensorFlowLibrary();
        const modelUrl = new URL(modelConfig.modelPath, window.location.href)
          .href;
        modelConfig.backendBenchmark = await benchmarkTensorFlowBackends(
          modelUrl,
          modelConfig,
          activeSchema,
        );
        model = await window.tf.loadLayersModel(modelUrl);
        try {
          validateModelShape(model, modelConfig, activeSchema);
          await warmupModel(model, activeSchema);
        } catch (error) {
          model.dispose();
          model = null;
          throw error;
        }
        modelReady = true;
        modelLoading = false;
        modelStatus.textContent = "Sign recognition is ready.";
        showNoSign();
      } catch (error) {
        modelReady = false;
        modelLoading = false;
        modelLoadPromise = null;
        modelStatus.textContent = "Sign recognition is unavailable. The camera preview is still available.";
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

  function renderStreamingPrediction() {
    if (!modelReady || !model) {
      showNoSign(
        "Landmark tracking works; a validated model for the active holistic feature schema is required for sign recognition.",
      );
      return;
    }
    const prediction = predictionResult;
    const multiPersonNote = multiplePeopleDetected ? " More than one person is in view." : "";
    if (!prediction || !prediction.best) {
      showNoSign(
        (signerVisible
          ? "Waiting for a stable sign."
          : "Keep your hands, face, or upper body in frame when ready.") + multiPersonNote,
      );
      return;
    }
    if (
      prediction.release ||
      prediction.best.label.toUpperCase() === NO_SIGN_LABEL
    ) {
      showNoSign(
        "No sign right now. Keep signing to add to your message." + multiPersonNote,
      );
      return;
    }
    const { best, accepted } = prediction;
    const signMetadata = labelMetadataFor(best.label);
    const segmentActive = streamSegmenter.activeLabel === best.label;
    detected.textContent = best.label;
    confidenceValue.textContent = best.confidence.toFixed(1) + "%";
    confidenceBar.style.width = Math.round(best.confidence) + "%";
    translation.textContent = (accepted
      ? segmentActive ? `${best.label} is in your message.` : `Checking ${best.label} before adding it to your message.` 
      : `Possible sign: ${best.label}. Keep signing clearly to confirm.`) + multiPersonNote;
    typePill.textContent = segmentActive
      ? "In your message"
      : signMetadata.unitType === "fingerspelling"
        ? "Letter"
        : signMetadata.unitType === "phrase"
          ? "Phrase"
          : "Word";
    arSign.textContent = best.label;
    arMeaning.textContent = signMetadata.displayText;
    overlay.classList.toggle("active", accepted);
    setState(
      accepted
        ? segmentActive
          ? "In your message"
          : "Checking sign"
        : "Possible sign",
      accepted ? "verifying" : "",
    );
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
    } catch (error) {
      console.error("Inference error:", error);
      resetRecognitionSegment();
      translation.textContent =
        "Sign recognition hit a problem. Please try again.";
      setState("Recognition issue", "verifying");
    } finally {
      if (input) input.dispose();
      if (output)
        (Array.isArray(output) ? output : [output]).forEach((tensor) =>
          tensor?.dispose?.(),
        );
      inferenceBusy = false;
    }
  }

  function onHandResults(results) {
    const now = performance.now();
    const segmentationTick = streamSegmenter.tick(now);
    for (const event of segmentationTick.events) addRecognizedUnit(event.label, event.confidence);
    const gap = lastTrackingResultAt ? now - lastTrackingResultAt : 0;
    if (gap > PIPELINE.config.sequence.trackingResetMs) {
      resetRecognitionSegment();
    }
    lastTrackingResultAt = now;
    updateTrackingStatus(results);
    const feature = PIPELINE.extractForVersion(
      results,
      activeSchema.version,
      video.videoWidth || PIPELINE.config.camera.width,
      video.videoHeight || PIPELINE.config.camera.height,
    );
    const hasTrackedCue = activeSchema.version === PIPELINE.config.features.version
      ? PIPELINE.hasAnyFeatureRegion(feature)
      : !!feature;
    if (!feature) {
      signerVisible = false;
      const release = streamSegmenter.update(null, 0, now);
      for (const event of release.events) addRecognizedUnit(event.label, event.confidence);
      const trackingLossMs = lastFeatureAt ? now - lastFeatureAt : 0;
      if (trackingLossMs > PIPELINE.config.sequence.trackingResetMs) {
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
      video.classList.add("active");
      placeholder.style.display = "none";
      liveBadge.textContent = "Camera ready";
      liveBadge.classList.add("show");
      trackingBadge.classList.add("show");
      startBtn.disabled = true;
      stopBtn.disabled = false;
      switchBtn.textContent =
        facingMode === "environment" ? "Use front camera" : "Use back camera";
      cameraStatus("Camera on, finding a sign", true);
      const trackerError = await trackerReadyPromise;
      if (generation !== cameraStartGeneration || !stream) return;
      if (trackerError) throw trackerError;
      trackerReady = true;
      liveBadge.textContent = "Camera ready";
      cameraStatus("Camera on", true);
      lastVideoTime = -1;
      lastFrameAt = 0;
      rafId = requestAnimationFrame((time) => frameLoop(time, generation));
      void initOptionalRecognition();
      toastMessage(
        modelReady
          ? "Camera ready. Sign a supported gesture to begin."
          : "Camera preview is ready. Sign recognition is not available right now.",
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
      liveBadge.textContent = "Camera ready";
      liveBadge.classList.remove("show");
      trackingBadge.classList.remove("show");
      trackerReady = false;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      showNoSign(cameraError(e));
      setState("Camera issue", "verifying");
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
        processCtx.drawImage(video, 0, 0, width, height);
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
          if (completedAt - lastRecognitionUpdateAt >= 500) {
            renderStreamingPrediction();
            lastRecognitionUpdateAt = completedAt;
          }
        }
      } catch (e) {
        if (generation === cameraStartGeneration) {
          console.error(e);
          resetRecognitionSegment();
          signerVisible = false;
          showNoSign(
            "We could not follow your sign. Check the framing and try again.",
          );
          setState("Tracking issue", "verifying");
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
    placeholder.style.display = "flex";
    liveBadge.textContent = "Camera ready";
    liveBadge.classList.remove("show");
    trackingBadge.classList.remove("show", "detected");
    startBtn.disabled = false;
    stopBtn.disabled = true;
    switchBtn.disabled = false;
    resetRecognitionSegment();
    lastTrackingResultAt = 0;
    lastRecognitionUpdateAt = 0;
    signerVisible = false;
    guide?.classList.remove("is-hidden");
    if (show) {
      cameraStatus("Camera off");
      showNoSign("Start the camera whenever you are ready.");
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
      switchBtn.textContent =
        facingMode === "environment" ? "Use front camera" : "Use back camera";
    }
  }
  navButtons.forEach((btn) =>
    btn.addEventListener("click", () => {
      navButtons.forEach((b) => {
        const selected = b === btn;
        b.classList.toggle("active", selected);
        b.setAttribute("aria-selected", String(selected));
      });
      sections.forEach((s) =>
        s.classList.toggle("active-section", s.id === btn.dataset.target),
      );
    }),
  );
  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", () => stopCamera(true));
  switchBtn.addEventListener("click", switchCamera);
  fullscreenBtn.addEventListener("click", async () => {
    try {
      if (!document.fullscreenElement) await cameraStage.requestFullscreen();
      else await document.exitFullscreen();
    } catch {
      toastMessage("Fullscreen is not available in this browser.");
    }
  });
  document.addEventListener("fullscreenchange", () => {
    const expanded = document.fullscreenElement === cameraStage;
    fullscreenBtn.textContent = expanded ? "Exit full screen" : "Expand view";
    fullscreenBtn.setAttribute("aria-expanded", String(expanded));
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
      toastMessage("Sentence copied.");
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
          ? "Sentence copied."
          : "Could not copy. Select and copy the sentence manually.",
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
  modelStatus.textContent = "Preparing sign recognition...";
  showNoSign();
  void loadModel();
})();
