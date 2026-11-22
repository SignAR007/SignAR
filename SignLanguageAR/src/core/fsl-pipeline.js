/* Shared hand-only feature and tensor contract for SignAR collection and inference. */
(() => {
  "use strict";

  const FEATURE_VERSION = "signar-fsl-hand-v3";
  const POINT_COUNT = 21;
  const ANGLE_COUNT = 5;
  const DIMENSION = POINT_COUNT * 3 + ANGLE_COUNT;
  const SEQUENCE_LENGTH = 10;
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  const CONFIG = Object.freeze({
    version: "signar-pipeline-v3",
    assets: Object.freeze({
      tasksVision: Object.freeze([
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs",
        "https://unpkg.com/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs",
      ]),
      handModel:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      handWasm: Object.freeze([
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm",
        "https://unpkg.com/@mediapipe/tasks-vision@0.10.21/wasm",
      ]),
      faceDetection: Object.freeze([
        "https://cdn.jsdelivr.net/npm/@mediapipe/face_detection@0.4.1646425229",
        "https://unpkg.com/@mediapipe/face_detection@0.4.1646425229",
      ]),
      tensorflow: Object.freeze([
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
        "https://unpkg.com/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
      ]),
      tensorflowWasm: Object.freeze([
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.js",
        "https://unpkg.com/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.js",
      ]),
    }),
    camera: Object.freeze({
      width: 960,
      height: 540,
      frameRateIdeal: 30,
      frameRateMax: 60,
    }),
    tracking: Object.freeze({
      mode: "video",
      dimensions: Object.freeze([480, 640, 720]),
      initialDimension: 640,
      personCheckDimension: 320,
      personCheckIntervalMs: 1000,
      handOptions: Object.freeze({
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }),
      faceDetectionOptions: Object.freeze({
        model: "short",
        minDetectionConfidence: 0.6,
      }),
      adaptation: Object.freeze({
        slowFrameMs: 50,
        slowFrames: 3,
        fastFrameMs: 34,
        fastFrames: 40,
        intervalSlowMs: 48,
        intervalMediumMs: 40,
        fpsSlow: 24,
        fpsMedium: 30,
        fpsFast: 30,
      }),
    }),
    features: Object.freeze({
      version: FEATURE_VERSION,
      dimension: DIMENSION,
      handPoints: POINT_COUNT,
      coordinatesPerPoint: 3,
      angleCount: ANGLE_COUNT,
      minimumUsableHandPoints: 1,
      coordinateLimit: 3,
      palmReference: "distance between index MCP 5 and pinky MCP 17",
      angles: Object.freeze([
        "thumb joint 2",
        "index PIP 6",
        "middle PIP 10",
        "ring PIP 14",
        "pinky PIP 18",
      ]),
    }),
    sequence: Object.freeze({
      length: SEQUENCE_LENGTH,
      samplingFps: 15,
      inferenceIntervalMs: 80,
      minimumInferenceFrames: 30,
      ringBufferFrames: 30,
      interpolationLimitMs: 400,
      trackingResetMs: 1000,
    }),
    recognition: Object.freeze({
      minimumConfidencePercent: 70,
      minimumMarginPercent: 18,
      stableFrames: 6,
      commitCooldownMs: 300,
      wordSeparatorPauseMs: 1500,
      speechDelayMs: 600,
      emaAlpha: 0.45,
    }),
  });

  const cameraConstraints = (facingMode, exactFacing = true) => {
    const video = {
      width: { ideal: CONFIG.camera.width },
      height: { ideal: CONFIG.camera.height },
      frameRate: {
        ideal: CONFIG.camera.frameRateIdeal,
        max: CONFIG.camera.frameRateMax,
      },
    };
    if (facingMode)
      video.facingMode = exactFacing
        ? { exact: facingMode }
        : { ideal: facingMode };
    return { video, audio: false };
  };

  const isVisiblePoint = (point) =>
    !!point &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    point.x >= -0.5 &&
    point.x <= 1.5 &&
    point.y >= -0.5 &&
    point.y <= 1.5;
  const handPointCount = (hand) =>
    Array.isArray(hand)
      ? hand.slice(0, POINT_COUNT).filter(isVisiblePoint).length
      : 0;
  const hasUsableHand = (hand) =>
    handPointCount(hand) >= CONFIG.features.minimumUsableHandPoints;

  function handArea(hand, width, height) {
    const points = (hand || []).slice(0, POINT_COUNT).filter(isVisiblePoint);
    if (points.length < 2) return 0;
    const xs = points.map((point) => point.x * width);
    const ys = points.map((point) => point.y * height);
    return (
      (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys))
    );
  }

  function chooseHand(
    results,
    width = CONFIG.camera.width,
    height = CONFIG.camera.height,
  ) {
    const choices = [
      { slot: "left", points: results?.leftHandLandmarks || [] },
      { slot: "right", points: results?.rightHandLandmarks || [] },
    ].filter(({ points }) => hasUsableHand(points));
    choices.sort((a, b) => {
      const countDifference =
        handPointCount(b.points) - handPointCount(a.points);
      return (
        countDifference ||
        handArea(b.points, width, height) - handArea(a.points, width, height)
      );
    });
    if (!choices.length) return null;
    return { ...choices[0], validPoints: handPointCount(choices[0].points) };
  }

  function angleAt(a, vertex, c) {
    if (!isVisiblePoint(a) || !isVisiblePoint(vertex) || !isVisiblePoint(c))
      return 0;
    const ax = a.x - vertex.x;
    const ay = a.y - vertex.y;
    const az =
      (Number.isFinite(a.z) ? a.z : 0) -
      (Number.isFinite(vertex.z) ? vertex.z : 0);
    const cx = c.x - vertex.x;
    const cy = c.y - vertex.y;
    const cz =
      (Number.isFinite(c.z) ? c.z : 0) -
      (Number.isFinite(vertex.z) ? vertex.z : 0);
    const aLength = Math.hypot(ax, ay, az);
    const cLength = Math.hypot(cx, cy, cz);
    if (!aLength || !cLength) return 0;
    const cosine = clamp(
      (ax * cx + ay * cy + az * cz) / (aLength * cLength),
      -1,
      1,
    );
    return Math.acos(cosine) / Math.PI;
  }

  function extract(
    results,
    width = CONFIG.camera.width,
    height = CONFIG.camera.height,
  ) {
    const selected = chooseHand(results, width, height);
    if (!selected) return null;
    const hand = selected.points;
    const wrist = hand[0];
    if (!isVisiblePoint(wrist)) return null;
    const indexMcp = hand[5];
    const pinkyMcp = hand[17];
    const palmSize =
      isVisiblePoint(indexMcp) && isVisiblePoint(pinkyMcp)
        ? Math.hypot(
            (indexMcp.x - pinkyMcp.x) * width,
            (indexMcp.y - pinkyMcp.y) * height,
          )
        : 0;
    const handBoundsSize = Math.max(1, handArea(hand, width, height) ** 0.5);
    const scale = Math.max(1, palmSize || handBoundsSize * 0.45);
    const mirror = selected.slot === "left" ? -1 : 1;
    const vector = new Array(DIMENSION);
    let offset = 0;
    for (let index = 0; index < POINT_COUNT; index++) {
      const point = hand[index];
      if (!isVisiblePoint(point)) {
        vector[offset++] = 0;
        vector[offset++] = 0;
        vector[offset++] = 0;
        continue;
      }
      const x = ((point.x - wrist.x) * width * mirror) / scale;
      const y = ((point.y - wrist.y) * height) / scale;
      const z =
        ((Number.isFinite(point.z) ? point.z : 0) * width -
          (Number.isFinite(wrist.z) ? wrist.z : 0) * width) /
        scale;
      vector[offset++] = clamp(
        x,
        -CONFIG.features.coordinateLimit,
        CONFIG.features.coordinateLimit,
      );
      vector[offset++] = clamp(
        y,
        -CONFIG.features.coordinateLimit,
        CONFIG.features.coordinateLimit,
      );
      vector[offset++] = clamp(
        z,
        -CONFIG.features.coordinateLimit,
        CONFIG.features.coordinateLimit,
      );
    }
    for (const [before, joint, after] of [
      [1, 2, 3],
      [5, 6, 7],
      [9, 10, 11],
      [13, 14, 15],
      [17, 18, 19],
    ])
      vector[offset++] = angleAt(hand[before], hand[joint], hand[after]);
    if (offset !== DIMENSION)
      throw new Error("Hand feature schema dimension mismatch.");
    return vector;
  }

  function mergeHandResults(_baseResults, handResults) {
    const results = handResults || {};
    const landmarks = results.landmarks || results.multiHandLandmarks || [];
    const handedness = results.handedness || results.multiHandedness || [];
    let left = [];
    let right = [];
    for (let index = 0; index < landmarks.length; index++) {
      const classification =
        handedness[index]?.[0] ||
        handedness[index]?.classification?.[0] ||
        handedness[index] ||
        {};
      const label = String(
        classification.categoryName || classification.label || "",
      ).toLowerCase();
      // MediaPipe handedness assumes mirrored selfie input; these frames are raw and unmirrored.
      const slot = label.includes("left")
        ? "right"
        : label.includes("right")
          ? "left"
          : !left.length
            ? "right"
            : "left";
      if (slot === "left" && !left.length) left = landmarks[index];
      else if (slot === "right" && !right.length) right = landmarks[index];
    }
    return {
      leftHandLandmarks: left,
      rightHandLandmarks: right,
      poseLandmarks: [],
      faceLandmarks: [],
      signarHandSource: "MediaPipe Tasks HandLandmarker",
      signarFacePresent: false,
      signarPosePresent: false,
    };
  }

  function createTrackingState() {
    return {
      dimensionIndex: CONFIG.tracking.dimensions.indexOf(
        CONFIG.tracking.initialDimension,
      ),
      slowFrames: 0,
      fastFrames: 0,
      processIntervalMs: 1000 / CONFIG.tracking.adaptation.fpsFast,
    };
  }
  function adaptTrackingState(state, elapsed) {
    const options = CONFIG.tracking.adaptation;
    if (elapsed > options.slowFrameMs) {
      state.slowFrames++;
      state.fastFrames = 0;
      if (state.slowFrames >= options.slowFrames) {
        state.dimensionIndex = Math.max(0, state.dimensionIndex - 1);
        state.slowFrames = 0;
      }
    } else if (elapsed < options.fastFrameMs) {
      state.fastFrames++;
      state.slowFrames = 0;
      if (state.fastFrames >= options.fastFrames) {
        state.dimensionIndex = Math.min(
          CONFIG.tracking.dimensions.length - 1,
          state.dimensionIndex + 1,
        );
        state.fastFrames = 0;
      }
    } else {
      state.slowFrames = 0;
      state.fastFrames = 0;
    }
    const fps =
      elapsed > options.intervalSlowMs
        ? options.fpsSlow
        : elapsed > options.intervalMediumMs
          ? options.fpsMedium
          : options.fpsFast;
    state.processIntervalMs = 1000 / fps;
    return state;
  }
  function trackingFrameSize(sourceWidth, sourceHeight, state) {
    const maxDimension =
      CONFIG.tracking.dimensions[state.dimensionIndex] ||
      CONFIG.tracking.initialDimension;
    const scale = Math.min(
      1,
      maxDimension / Math.max(sourceWidth, sourceHeight),
    );
    return {
      width: Math.max(1, Math.round(sourceWidth * scale)),
      height: Math.max(1, Math.round(sourceHeight * scale)),
      scale,
    };
  }
  const sequenceGapToleranceMs = () => CONFIG.sequence.trackingResetMs;

  function validateFrames(frames) {
    if (
      !Array.isArray(frames) ||
      !frames.length ||
      frames.some(
        (frame) =>
          !Array.isArray(frame) ||
          frame.length !== DIMENSION ||
          frame.some((value) => !Number.isFinite(value)),
      )
    )
      throw new Error(
        `Expected finite hand feature frames with ${DIMENSION} values.`,
      );
  }
  function resampleSequence(frames, target = SEQUENCE_LENGTH) {
    validateFrames(frames);
    if (!Number.isInteger(target) || target < 1)
      throw new Error("Target sequence length must be positive.");
    const last = frames.length - 1;
    const output = new Array(target);
    for (let index = 0; index < target; index++) {
      const position = target === 1 ? last : (index * last) / (target - 1);
      const lower = Math.floor(position);
      const upper = Math.min(lower + 1, last);
      const weight = position - lower;
      output[index] = frames[lower].map(
        (value, feature) =>
          value * (1 - weight) + frames[upper][feature] * weight,
      );
    }
    return output;
  }
  function selectRecentSequenceWindow(
    frames,
    target = SEQUENCE_LENGTH,
    windowFrames = CONFIG.sequence.ringBufferFrames,
  ) {
    if (!Number.isInteger(windowFrames) || windowFrames < 1)
      throw new Error("Recent window size must be a positive integer.");
    return resampleSequence(frames.slice(-windowFrames), target);
  }
  function interpolateFeatureFrame(from, to, weight) {
    validateFrames([from, to]);
    if (!Number.isFinite(weight) || weight < 0 || weight > 1)
      throw new Error("Interpolation weight must be between 0 and 1.");
    return from.map(
      (value, index) => value * (1 - weight) + to[index] * weight,
    );
  }
  function sequenceTensorShape(batchSize = null) {
    if (batchSize !== null && (!Number.isInteger(batchSize) || batchSize < 1))
      throw new Error("Batch size must be positive or null.");
    return [batchSize, SEQUENCE_LENGTH, DIMENSION];
  }
  function createBatchTensor(tf, sequences) {
    if (!tf?.tensor3d || !Array.isArray(sequences) || !sequences.length)
      throw new Error("TensorFlow.js needs at least one feature window.");
    const values = new Float32Array(
      sequences.length * SEQUENCE_LENGTH * DIMENSION,
    );
    let offset = 0;
    for (const sequence of sequences) {
      const normalized = resampleSequence(sequence, SEQUENCE_LENGTH);
      for (const frame of normalized) {
        values.set(frame, offset);
        offset += DIMENSION;
      }
    }
    return tf.tensor3d(values, sequenceTensorShape(sequences.length));
  }
  function createSequenceTensor(tf, frames) {
    return createBatchTensor(tf, [frames]);
  }
  function createRecentSequenceTensor(tf, frames) {
    return createBatchTensor(tf, [frames.slice(-SEQUENCE_LENGTH)]);
  }
  function measureMotion(previous, current) {
    if (
      !Array.isArray(previous) ||
      !Array.isArray(current) ||
      previous.length !== DIMENSION ||
      current.length !== DIMENSION
    )
      return 0;
    let total = 0;
    for (let index = 0; index < DIMENSION; index++)
      total += Math.abs(current[index] - previous[index]);
    return total / DIMENSION;
  }

  window.SignARPipeline = Object.freeze({
    config: CONFIG,
    cameraConstraints,
    createTrackingState,
    adaptTrackingState,
    sequenceGapToleranceMs,
    trackingFrameSize,
    isVisiblePoint,
    hasUsableHand,
    chooseHand,
    extract,
    mergeHandResults,
    measureMotion,
    resampleSequence,
    selectRecentSequenceWindow,
    interpolateFeatureFrame,
    sequenceTensorShape,
    createBatchTensor,
    createSequenceTensor,
    createRecentSequenceTensor,
    dimension: DIMENSION,
    sequenceLength: SEQUENCE_LENGTH,
    version: FEATURE_VERSION,
  });
})();
