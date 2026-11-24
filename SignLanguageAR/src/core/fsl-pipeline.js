/* Shared holistic feature and tensor contract for SignAR collection and inference. */
(() => {
  "use strict";

  const FEATURE_VERSION = "signar-fsl-holistic-v4";
  const HAND_V3_VERSION = "signar-fsl-hand-v3";
  const LEGACY_456_VERSION = "signar-fsl-456-v2";
  const MISSING_VALUE = -3;
  const HAND_POINTS = 21;
  const HAND_COORDINATES = HAND_POINTS * 3;
  const HAND_ANGLES = 5;
  const HAND_WRIST_POSITION = 3;
  const HAND_SLOT_DIMENSION = HAND_COORDINATES + HAND_ANGLES + HAND_WRIST_POSITION;
  const POSE_POINTS = 33;
  const POSE_POINT_DIMENSION = 4;
  const FACE_IDS = Object.freeze([
    33, 133, 159, 145, 153, 154, 155, 246,
    263, 362, 386, 374, 380, 381, 382, 466,
    70, 63, 105, 66, 107, 336, 296, 334,
    293, 300, 61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291, 78, 95, 88,
    178, 87, 14, 317, 402, 318, 324, 308,
  ]);
  const FACE_POINTS = FACE_IDS.length;
  const LEFT_HAND_OFFSET = 0;
  const RIGHT_HAND_OFFSET = HAND_SLOT_DIMENSION;
  const POSE_OFFSET = RIGHT_HAND_OFFSET + HAND_SLOT_DIMENSION;
  const FACE_OFFSET = POSE_OFFSET + POSE_POINTS * POSE_POINT_DIMENSION;
  const PRESENCE_OFFSET = FACE_OFFSET + FACE_POINTS * 3;
  const DIMENSION = PRESENCE_OFFSET + 4;
  const SEQUENCE_LENGTH = 30;
  const HAND_V3_DIMENSION = 68;
  const HAND_V3_SEQUENCE_LENGTH = 10;
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const finite = (value) => Number.isFinite(value);
  const visiblePoint = (point) =>
    !!point && finite(point.x) && finite(point.y) &&
    point.x >= -0.5 && point.x <= 1.5 && point.y >= -0.5 && point.y <= 1.5;
  const unwrap = (value) => Array.isArray(value?.[0]) ? value[0] : (value || []);
  const handSchema = Object.freeze({
    version: HAND_V3_VERSION,
    dimension: HAND_V3_DIMENSION,
    sequenceLength: HAND_V3_SEQUENCE_LENGTH,
    windowFrames: 30,
  });
  const holisticSchema = Object.freeze({
    version: FEATURE_VERSION,
    dimension: DIMENSION,
    sequenceLength: SEQUENCE_LENGTH,
    windowFrames: 45,
  });
  const SCHEMAS = Object.freeze({
    [HAND_V3_VERSION]: handSchema,
    [FEATURE_VERSION]: holisticSchema,
  });

  const CONFIG = Object.freeze({
    version: "signar-pipeline-v4",
    assets: Object.freeze({
      tasksVision: Object.freeze([
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs",
        "https://unpkg.com/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs",
      ]),
      holisticModel:
        "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task",
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
      tensorflowWebGpu: Object.freeze([
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.js",
        "https://unpkg.com/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.js",
      ]),
    }),
    camera: Object.freeze({ width: 960, height: 540, frameRateIdeal: 30, frameRateMax: 60 }),
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
      holisticOptions: Object.freeze({
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minHandLandmarksConfidence: 0.5,
      }),
      faceDetectionOptions: Object.freeze({ model: "short", minDetectionConfidence: 0.6 }),
      adaptation: Object.freeze({
        slowFrameMs: 50, slowFrames: 3, fastFrameMs: 34, fastFrames: 40,
        intervalSlowMs: 48, intervalMediumMs: 40, fpsSlow: 24, fpsMedium: 30, fpsFast: 30,
      }),
    }),
    features: Object.freeze({
      version: FEATURE_VERSION,
      dimension: DIMENSION,
      missingValue: MISSING_VALUE,
      coordinateLimit: 3,
      layout: Object.freeze({
        leftHand: Object.freeze({ offset: LEFT_HAND_OFFSET, dimension: HAND_SLOT_DIMENSION }),
        rightHand: Object.freeze({ offset: RIGHT_HAND_OFFSET, dimension: HAND_SLOT_DIMENSION }),
        pose: Object.freeze({ offset: POSE_OFFSET, points: POSE_POINTS, valuesPerPoint: POSE_POINT_DIMENSION }),
        face: Object.freeze({ offset: FACE_OFFSET, points: FACE_POINTS, valuesPerPoint: 3, landmarkIds: FACE_IDS }),
        presence: Object.freeze({ offset: PRESENCE_OFFSET, order: Object.freeze(["leftHand", "rightHand", "pose", "face"]) }),
      }),
      handPoints: HAND_POINTS,
      handSlotDimension: HAND_SLOT_DIMENSION,
      posePoints: POSE_POINTS,
      facePoints: FACE_POINTS,
      minimumUsableHandPoints: 1,
      angles: Object.freeze(["thumb joint 2", "index PIP 6", "middle PIP 10", "ring PIP 14", "pinky PIP 18"]),
    }),
    sequence: Object.freeze({
      length: SEQUENCE_LENGTH,
      samplingFps: 15,
      inferenceIntervalMs: 50,
      minimumInferenceFrames: SEQUENCE_LENGTH,
      ringBufferFrames: holisticSchema.windowFrames,
      interpolationLimitMs: 400,
      trackingResetMs: 1000,
      segmentationPauseMs: 450,
      segmentationTransitionWindows: 3,
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

  function getSchema(version = FEATURE_VERSION) {
    return SCHEMAS[String(version || "")] || null;
  }
  function cameraConstraints(facingMode, exactFacing = true) {
    const video = {
      width: { ideal: CONFIG.camera.width }, height: { ideal: CONFIG.camera.height },
      frameRate: { ideal: CONFIG.camera.frameRateIdeal, max: CONFIG.camera.frameRateMax },
    };
    if (facingMode) video.facingMode = exactFacing ? { exact: facingMode } : { ideal: facingMode };
    return { video, audio: false };
  }
  const handPointCount = (hand) => Array.isArray(hand) ? unwrap(hand).slice(0, HAND_POINTS).filter(visiblePoint).length : 0;
  const hasUsableHand = (hand) => handPointCount(hand) >= CONFIG.features.minimumUsableHandPoints;
  function handArea(hand, width, height) {
    const points = unwrap(hand).slice(0, HAND_POINTS).filter(visiblePoint);
    if (points.length < 2) return 0;
    const xs = points.map((point) => point.x * width), ys = points.map((point) => point.y * height);
    return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
  }
  function chooseHand(results, width = CONFIG.camera.width, height = CONFIG.camera.height) {
    const choices = [
      { slot: "left", points: unwrap(results?.leftHandLandmarks) },
      { slot: "right", points: unwrap(results?.rightHandLandmarks) },
    ].filter(({ points }) => hasUsableHand(points));
    choices.sort((a, b) => handPointCount(b.points) - handPointCount(a.points) || handArea(b.points, width, height) - handArea(a.points, width, height));
    if (!choices.length) return null;
    return { ...choices[0], validPoints: handPointCount(choices[0].points) };
  }
  function angleAt(a, vertex, c) {
    if (!visiblePoint(a) || !visiblePoint(vertex) || !visiblePoint(c)) return null;
    const av = [a.x - vertex.x, a.y - vertex.y, (finite(a.z) ? a.z : 0) - (finite(vertex.z) ? vertex.z : 0)];
    const cv = [c.x - vertex.x, c.y - vertex.y, (finite(c.z) ? c.z : 0) - (finite(vertex.z) ? vertex.z : 0)];
    const denominator = Math.hypot(...av) * Math.hypot(...cv);
    if (!denominator) return null;
    return Math.acos(clamp(av.reduce((sum, value, i) => sum + value * cv[i], 0) / denominator, -1, 1)) / Math.PI;
  }
  function handScale(hand, width, height) {
    const points = hand.slice(0, HAND_POINTS).filter(visiblePoint);
    const indexMcp = hand[5], pinkyMcp = hand[17];
    const palm = visiblePoint(indexMcp) && visiblePoint(pinkyMcp)
      ? Math.hypot((indexMcp.x - pinkyMcp.x) * width, (indexMcp.y - pinkyMcp.y) * height)
      : 0;
    const xs = points.map((point) => point.x * width), ys = points.map((point) => point.y * height);
    const bounds = points.length > 1 ? Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 0.45 : 0;
    return Math.max(1, palm || bounds);
  }
  function poseBasis(pose, width, height) {
    const left = pose[11], right = pose[12];
    if (visiblePoint(left) && visiblePoint(right)) {
      const center = { x: (left.x + right.x) / 2, y: (left.y + right.y) / 2, z: ((left.z || 0) + (right.z || 0)) / 2 };
      const scale = Math.hypot((left.x - right.x) * width, (left.y - right.y) * height);
      if (scale > 1) return { center, scale };
    }
    return { center: { x: 0.5, y: 0.5, z: 0 }, scale: Math.max(1, width * 0.25) };
  }
  function appendHandFeature(vector, offset, handValue, slot, basis, width, height) {
    const hand = unwrap(handValue).slice(0, HAND_POINTS);
    if (!hasUsableHand(hand) || !visiblePoint(hand[0])) return false;
    const wrist = hand[0];
    const scale = handScale(hand, width, height);
    const mirror = slot === "left" ? -1 : 1;
    for (let index = 0; index < HAND_POINTS; index++) {
      const point = hand[index], start = offset + index * 3;
      if (!visiblePoint(point)) continue;
      vector[start] = clamp(((point.x - wrist.x) * width * mirror) / scale, -3, 3);
      vector[start + 1] = clamp(((point.y - wrist.y) * height) / scale, -3, 3);
      vector[start + 2] = clamp((((finite(point.z) ? point.z : 0) - (finite(wrist.z) ? wrist.z : 0)) * width) / scale, -3, 3);
    }
    for (const [index, [before, joint, after]] of [[1, [1, 2, 3]], [2, [5, 6, 7]], [3, [9, 10, 11]], [4, [13, 14, 15]], [5, [17, 18, 19]]]) {
      const value = angleAt(hand[before], hand[joint], hand[after]);
      if (value !== null) vector[offset + HAND_COORDINATES + index - 1] = value;
    }
    const wristOffset = offset + HAND_COORDINATES + HAND_ANGLES;
    vector[wristOffset] = clamp(((wrist.x - basis.center.x) * width * 1) / basis.scale, -3, 3);
    vector[wristOffset + 1] = clamp(((wrist.y - basis.center.y) * height) / basis.scale, -3, 3);
    vector[wristOffset + 2] = clamp((((finite(wrist.z) ? wrist.z : 0) - basis.center.z) * width) / basis.scale, -3, 3);
    return true;
  }
  function appendPoseFeature(vector, poseValue, basis, width, height) {
    const pose = unwrap(poseValue).slice(0, POSE_POINTS);
    let found = false;
    for (let index = 0; index < POSE_POINTS; index++) {
      const point = pose[index], offset = POSE_OFFSET + index * POSE_POINT_DIMENSION;
      if (!visiblePoint(point)) continue;
      const visibility = finite(point.visibility) ? clamp(point.visibility, 0, 1) : 1;
      if (visibility < 0.1) continue;
      vector[offset] = clamp(((point.x - basis.center.x) * width) / basis.scale, -3, 3);
      vector[offset + 1] = clamp(((point.y - basis.center.y) * height) / basis.scale, -3, 3);
      vector[offset + 2] = clamp((((finite(point.z) ? point.z : 0) - basis.center.z) * width) / basis.scale, -3, 3);
      vector[offset + 3] = visibility;
      found = true;
    }
    return found;
  }
  function appendFaceFeature(vector, faceValue, width, height) {
    const face = unwrap(faceValue);
    if (!face.length) return false;
    const anchor = face[1] && visiblePoint(face[1]) ? face[1] : face.find(visiblePoint);
    if (!anchor) return false;
    const left = face[234], right = face[454];
    const faceScale = visiblePoint(left) && visiblePoint(right)
      ? Math.hypot((left.x - right.x) * width, (left.y - right.y) * height)
      : width * 0.2;
    if (!(faceScale > 1)) return false;
    let found = false;
    for (let index = 0; index < FACE_POINTS; index++) {
      const point = face[FACE_IDS[index]];
      if (!visiblePoint(point)) continue;
      const offset = FACE_OFFSET + index * 3;
      vector[offset] = clamp(((point.x - anchor.x) * width) / faceScale, -3, 3);
      vector[offset + 1] = clamp(((point.y - anchor.y) * height) / faceScale, -3, 3);
      vector[offset + 2] = clamp((((finite(point.z) ? point.z : 0) - (finite(anchor.z) ? anchor.z : 0)) * width) / faceScale, -3, 3);
      found = true;
    }
    return found;
  }
  function extractHolistic(results, width = CONFIG.camera.width, height = CONFIG.camera.height) {
    const vector = new Array(DIMENSION).fill(MISSING_VALUE);
    vector.fill(0, PRESENCE_OFFSET);
    const pose = unwrap(results?.poseLandmarks);
    const basis = poseBasis(pose, width, height);
    if (appendHandFeature(vector, LEFT_HAND_OFFSET, results?.leftHandLandmarks, "left", basis, width, height)) vector[PRESENCE_OFFSET] = 1;
    if (appendHandFeature(vector, RIGHT_HAND_OFFSET, results?.rightHandLandmarks, "right", basis, width, height)) vector[PRESENCE_OFFSET + 1] = 1;
    if (appendPoseFeature(vector, pose, basis, width, height)) vector[PRESENCE_OFFSET + 2] = 1;
    if (appendFaceFeature(vector, results?.faceLandmarks, width, height)) vector[PRESENCE_OFFSET + 3] = 1;
    return vector;
  }
  function extractHandV3(results, width = CONFIG.camera.width, height = CONFIG.camera.height) {
    const selected = chooseHand(results, width, height);
    if (!selected) return null;
    const hand = selected.points, wrist = hand[0];
    if (!visiblePoint(wrist)) return null;
    const scale = handScale(hand, width, height);
    const mirror = selected.slot === "left" ? -1 : 1;
    const vector = new Array(HAND_V3_DIMENSION).fill(0);
    for (let index = 0; index < HAND_POINTS; index++) {
      const point = hand[index], offset = index * 3;
      if (!visiblePoint(point)) continue;
      vector[offset] = clamp(((point.x - wrist.x) * width * mirror) / scale, -3, 3);
      vector[offset + 1] = clamp(((point.y - wrist.y) * height) / scale, -3, 3);
      vector[offset + 2] = clamp((((finite(point.z) ? point.z : 0) - (finite(wrist.z) ? wrist.z : 0)) * width) / scale, -3, 3);
    }
    for (const [index, [before, joint, after]] of [[0, [1, 2, 3]], [1, [5, 6, 7]], [2, [9, 10, 11]], [3, [13, 14, 15]], [4, [17, 18, 19]]]) {
      const value = angleAt(hand[before], hand[joint], hand[after]);
      if (value !== null) vector[63 + index] = value;
    }
    return vector;
  }
  function hasAnyFeatureRegion(frame) {
    return Array.isArray(frame) && frame.length === DIMENSION && frame.slice(PRESENCE_OFFSET, PRESENCE_OFFSET + 4).some((value) => value >= 0.5);
  }
  function hasTrackedRegion(results) {
    return hasUsableHand(results?.leftHandLandmarks) || hasUsableHand(results?.rightHandLandmarks) ||
      unwrap(results?.poseLandmarks).some((point) => visiblePoint(point) && (!finite(point.visibility) || point.visibility >= 0.1)) ||
      unwrap(results?.faceLandmarks).some(visiblePoint);
  }
  function mergeHandResults(_baseResults, handResults) {
    const results = handResults || {};
    const landmarks = results.landmarks || results.multiHandLandmarks || [];
    const handedness = results.handedness || results.multiHandedness || [];
    let left = [], right = [];
    for (let index = 0; index < landmarks.length; index++) {
      const classification = handedness[index]?.[0] || handedness[index]?.classification?.[0] || handedness[index] || {};
      const label = String(classification.categoryName || classification.label || "").toLowerCase();
      const slot = label.includes("left") ? "right" : label.includes("right") ? "left" : !left.length ? "right" : "left";
      if (slot === "left" && !left.length) left = landmarks[index];
      else if (slot === "right" && !right.length) right = landmarks[index];
    }
    return { leftHandLandmarks: left, rightHandLandmarks: right, poseLandmarks: [], faceLandmarks: [], signarHandSource: "MediaPipe Tasks Vision HandLandmarker", signarFacePresent: false, signarPosePresent: false };
  }
  function mergeHolisticResults(results) {
    const value = results || {};
    return {
      ...value,
      leftHandLandmarks: unwrap(value.leftHandLandmarks),
      rightHandLandmarks: unwrap(value.rightHandLandmarks),
      poseLandmarks: unwrap(value.poseLandmarks),
      faceLandmarks: unwrap(value.faceLandmarks),
      signarHandSource: "MediaPipe Tasks Vision HolisticLandmarker",
      signarFacePresent: !!unwrap(value.faceLandmarks).length,
      signarPosePresent: !!unwrap(value.poseLandmarks).length,
    };
  }
  function migrateLegacy456Frame(frame) {
    if (!Array.isArray(frame) || frame.length !== 456 || frame.some((value) => !finite(value))) throw new Error("Legacy frames must contain 456 finite features.");
    const output = new Array(DIMENSION).fill(MISSING_VALUE);
    output.fill(0, PRESENCE_OFFSET);
    const pose = frame.slice(0, 132);
    let poseFound = false;
    for (let index = 0; index < POSE_POINTS; index++) {
      const source = index * 4, target = POSE_OFFSET + index * 4;
      const point = pose.slice(source, source + 4);
      if (point.length === 4 && point.slice(0, 3).some((value) => value > -2.75)) {
        output.splice(target, 4, ...point.map((value, axis) => axis === 3 ? clamp(value, 0, 1) : clamp(value, -3, 3)));
        poseFound = true;
      }
    }
    if (poseFound) output[PRESENCE_OFFSET + 2] = 1;
    for (let slot = 0; slot < 2; slot++) {
      const sourceBase = 132 + slot * 126;
      const targetBase = slot === 0 ? LEFT_HAND_OFFSET : RIGHT_HAND_OFFSET;
      let found = false;
      const points = [];
      for (let point = 0; point < HAND_POINTS; point++) {
        const source = sourceBase + point * 6 + 3;
        const xyz = frame.slice(source, source + 3);
        points.push(xyz);
        if (xyz.length === 3 && xyz[0] > -2.75 && xyz.every(finite)) {
          const target = targetBase + point * 3;
          output[target] = clamp(xyz[0] * (slot === 0 ? -1 : 1), -3, 3);
          output[target + 1] = clamp(xyz[1], -3, 3);
          output[target + 2] = clamp(xyz[2], -3, 3);
          found = true;
        }
      }
      if (found) {
        for (const [angleIndex, [a, b, c]] of [[0, [1, 2, 3]], [1, [5, 6, 7]], [2, [9, 10, 11]], [3, [13, 14, 15]], [4, [17, 18, 19]]]) {
          const first = points[a], vertex = points[b], last = points[c];
          if ([first, vertex, last].some((point) => !point || point.length !== 3 || point.some((value) => value <= -2.75))) continue;
          const av = first.map((value, axis) => value - vertex[axis]);
          const cv = last.map((value, axis) => value - vertex[axis]);
          const denominator = Math.hypot(...av) * Math.hypot(...cv);
          if (denominator) output[targetBase + HAND_COORDINATES + angleIndex] = Math.acos(clamp(av.reduce((sum, value, axis) => sum + value * cv[axis], 0) / denominator, -1, 1)) / Math.PI;
        }
        output[targetBase + HAND_COORDINATES + HAND_ANGLES] = 0;
        output[targetBase + HAND_COORDINATES + HAND_ANGLES + 1] = 0;
        output[targetBase + HAND_COORDINATES + HAND_ANGLES + 2] = 0;
        output[PRESENCE_OFFSET + slot] = 1;
      }
    }
    const oldFace = frame.slice(384, 456);
    let faceFound = false;
    for (let index = 0; index < 24; index++) {
      const point = oldFace.slice(index * 3, index * 3 + 3);
      if (point.length !== 3 || point.some((value) => value <= -2.75)) continue;
      output[FACE_OFFSET + index * 3] = clamp(point[0], -3, 3);
      output[FACE_OFFSET + index * 3 + 1] = clamp(point[1], -3, 3);
      output[FACE_OFFSET + index * 3 + 2] = clamp(point[2], -3, 3);
      faceFound = true;
    }
    if (faceFound) output[PRESENCE_OFFSET + 3] = 1;
    return output;
  }
  function migrateHandV3Frame(frame) {
    if (!Array.isArray(frame) || frame.length !== HAND_V3_DIMENSION || frame.some((value) => !finite(value))) throw new Error("Hand-v3 frames must contain 68 finite features.");
    const output = new Array(DIMENSION).fill(MISSING_VALUE);
    output.fill(0, PRESENCE_OFFSET);
    if (frame.every((value) => value === 0)) return output;
    output.splice(RIGHT_HAND_OFFSET, HAND_COORDINATES, ...frame.slice(0, HAND_COORDINATES));
    output.splice(RIGHT_HAND_OFFSET + HAND_COORDINATES, HAND_ANGLES, ...frame.slice(HAND_COORDINATES));
    output[PRESENCE_OFFSET + 1] = 1;
    return output;
  }
  function migrateFrame(frame, sourceVersion) {
    if (sourceVersion === FEATURE_VERSION) {
      validateFrames([frame], DIMENSION);
      return frame.slice();
    }
    if (sourceVersion === HAND_V3_VERSION) return migrateHandV3Frame(frame);
    if (sourceVersion === LEGACY_456_VERSION) return migrateLegacy456Frame(frame);
    throw new Error("Unsupported feature schema: " + String(sourceVersion));
  }
  function createTrackingState() {
    return { dimensionIndex: CONFIG.tracking.dimensions.indexOf(CONFIG.tracking.initialDimension), slowFrames: 0, fastFrames: 0, processIntervalMs: 1000 / CONFIG.tracking.adaptation.fpsFast };
  }
  function adaptTrackingState(state, elapsed) {
    const options = CONFIG.tracking.adaptation;
    if (elapsed > options.slowFrameMs) {
      state.slowFrames++; state.fastFrames = 0;
      if (state.slowFrames >= options.slowFrames) { state.dimensionIndex = Math.max(0, state.dimensionIndex - 1); state.slowFrames = 0; }
    } else if (elapsed < options.fastFrameMs) {
      state.fastFrames++; state.slowFrames = 0;
      if (state.fastFrames >= options.fastFrames) { state.dimensionIndex = Math.min(CONFIG.tracking.dimensions.length - 1, state.dimensionIndex + 1); state.fastFrames = 0; }
    } else { state.slowFrames = 0; state.fastFrames = 0; }
    const fps = elapsed > options.intervalSlowMs ? options.fpsSlow : elapsed > options.intervalMediumMs ? options.fpsMedium : options.fpsFast;
    state.processIntervalMs = 1000 / fps;
    return state;
  }
  function trackingFrameSize(sourceWidth, sourceHeight, state) {
    const maxDimension = CONFIG.tracking.dimensions[state.dimensionIndex] || CONFIG.tracking.initialDimension;
    const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
    return { width: Math.max(1, Math.round(sourceWidth * scale)), height: Math.max(1, Math.round(sourceHeight * scale)), scale };
  }
  const sequenceGapToleranceMs = () => CONFIG.sequence.trackingResetMs;
  function validateFrames(frames, dimension = DIMENSION) {
    if (!Array.isArray(frames) || !frames.length || frames.some((frame) => !Array.isArray(frame) || frame.length !== dimension || frame.some((value) => !finite(value) || value < -3.0001 || value > 3.0001)))
      throw new Error(`Expected finite feature frames with ${dimension} values in the supported range.`);
  }
  function interpolateValue(from, to, weight) {
    if (from === MISSING_VALUE || to === MISSING_VALUE) {
      if (from === MISSING_VALUE && to === MISSING_VALUE) return MISSING_VALUE;
      return weight < 0.5 ? from : to;
    }
    return from * (1 - weight) + to * weight;
  }
  function resampleSequence(frames, target = SEQUENCE_LENGTH) {
    if (!Array.isArray(frames) || !frames.length) throw new Error("Expected at least one feature frame.");
    validateFrames(frames, frames[0].length);
    if (!Number.isInteger(target) || target < 1) throw new Error("Target sequence length must be positive.");
    const last = frames.length - 1, output = new Array(target);
    for (let index = 0; index < target; index++) {
      const position = target === 1 ? last : (index * last) / (target - 1);
      const lower = Math.floor(position), upper = Math.min(lower + 1, last), weight = position - lower;
      output[index] = frames[lower].map((value, feature) => interpolateValue(value, frames[upper][feature], weight));
    }
    return output;
  }
  function selectRecentSequenceWindow(frames, target = SEQUENCE_LENGTH, windowFrames = CONFIG.sequence.ringBufferFrames) {
    if (!Number.isInteger(windowFrames) || windowFrames < 1) throw new Error("Recent window size must be a positive integer.");
    return resampleSequence(frames.slice(-windowFrames), target);
  }
  function interpolateFeatureFrame(from, to, weight) {
    if (!Array.isArray(from) || !Array.isArray(to) || from.length !== to.length) throw new Error("Feature frames must have matching dimensions.");
    validateFrames([from, to], from.length);
    if (!finite(weight) || weight < 0 || weight > 1) throw new Error("Interpolation weight must be between 0 and 1.");
    return from.map((value, index) => interpolateValue(value, to[index], weight));
  }
  function sequenceTensorShape(batchSize = null, schema = FEATURE_VERSION) {
    if (batchSize !== null && (!Number.isInteger(batchSize) || batchSize < 1)) throw new Error("Batch size must be positive or null.");
    const selected = typeof schema === "object" ? schema : getSchema(schema);
    if (!selected) throw new Error("Unsupported model feature schema.");
    return [batchSize, selected.sequenceLength, selected.dimension];
  }
  function createBatchTensor(tf, sequences, schema = FEATURE_VERSION) {
    const selected = typeof schema === "object" ? schema : getSchema(schema);
    if (!selected || !tf?.tensor3d || !Array.isArray(sequences) || !sequences.length) throw new Error("TensorFlow.js needs at least one feature window and a supported schema.");
    const values = new Float32Array(sequences.length * selected.sequenceLength * selected.dimension);
    let offset = 0;
    for (const sequence of sequences) {
      const normalized = resampleSequence(sequence, selected.sequenceLength);
      validateFrames(normalized, selected.dimension);
      for (const frame of normalized) { values.set(frame, offset); offset += selected.dimension; }
    }
    return tf.tensor3d(values, sequenceTensorShape(sequences.length, selected));
  }
  function createSequenceTensor(tf, frames, schema = FEATURE_VERSION) { return createBatchTensor(tf, [frames], schema); }
  function createRecentSequenceTensor(tf, frames, schema = FEATURE_VERSION) {
    const selected = typeof schema === "object" ? schema : getSchema(schema);
    return createBatchTensor(tf, [frames.slice(-selected.sequenceLength)], selected);
  }
  function measureMotion(previous, current) {
    if (!Array.isArray(previous) || !Array.isArray(current) || previous.length !== current.length) return 0;
    let total = 0, count = 0;
    for (let index = 0; index < current.length; index++) {
      if (previous[index] === MISSING_VALUE || current[index] === MISSING_VALUE) continue;
      total += Math.abs(current[index] - previous[index]); count++;
    }
    return count ? total / count : 0;
  }

  window.SignARPipeline = Object.freeze({
    config: CONFIG,
    schemas: SCHEMAS,
    getSchema,
    cameraConstraints,
    createTrackingState,
    adaptTrackingState,
    sequenceGapToleranceMs,
    trackingFrameSize,
    isVisiblePoint: visiblePoint,
    hasUsableHand,
    chooseHand,
    extract: extractHolistic,
    extractForVersion(results, version, width, height) {
      return version === HAND_V3_VERSION ? extractHandV3(results, width, height) : extractHolistic(results, width, height);
    },
    extractHandV3,
    hasAnyFeatureRegion,
    hasTrackedRegion,
    mergeHandResults,
    mergeHolisticResults,
    migrateFrame,
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
    missingValue: MISSING_VALUE,
  });
})();
