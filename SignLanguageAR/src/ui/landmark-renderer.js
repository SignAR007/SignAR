/* Shared camera overlay for collector and interpreter tracking results. */
(() => {
  "use strict";
  const PIPELINE = window.SignARPipeline;
  const HAND_CONNECTIONS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [0, 17],
  ];
  const POSE_CONNECTIONS = [
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [11, 23],
    [12, 24],
    [23, 24],
    [23, 25],
    [25, 27],
    [27, 29],
    [29, 31],
    [24, 26],
    [26, 28],
    [28, 30],
    [30, 32],
  ];
  const FACE_PATHS = [
    [33, 160, 158, 133, 153, 144, 33],
    [362, 385, 387, 263, 373, 380, 362],
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    [70, 63, 105, 66, 107],
    [300, 293, 334, 296, 336],
  ];
  const FACE_CONNECTIONS = FACE_PATHS.flatMap((path) =>
    path.slice(1).map((point, index) => [path[index], point]),
  );
  const POSE_STATUS_IDS =
    PIPELINE.config.features.poseStatusIds ||
    Array.from(new Set(POSE_CONNECTIONS.flat()));
  const FACE_LANDMARK_IDS =
    PIPELINE.config.features.faceLandmarkIds ||
    Array.from(new Set(FACE_CONNECTIONS.flat()));
  const isDrawable = (point) => PIPELINE.isVisiblePoint(point);
  const poseDisplayConfidence =
    PIPELINE.config.tracking.poseDisplayConfidence ?? 0.5;
  const isPoseDrawable = (point) => {
    const confidence = point?.visibility ?? point?.presence;
    return (
      isDrawable(point) &&
      (!Number.isFinite(confidence) || confidence >= poseDisplayConfidence) &&
      point.x >= 0 &&
      point.x <= 1 &&
      point.y >= 0 &&
      point.y <= 1
    );
  };
  const hasTorsoAnchor = (pose) =>
    pose.length > 12 && isPoseDrawable(pose[11]) && isPoseDrawable(pose[12]);

  function drawConnections(
    context,
    points,
    connections,
    mapPoint,
    color,
    lineWidth,
    isDrawablePoint = isDrawable,
  ) {
    context.save();
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const [start, end] of connections) {
      const a = points[start],
        b = points[end];
      if (!isDrawablePoint(a) || !isDrawablePoint(b)) continue;
      const pa = mapPoint(a),
        pb = mapPoint(b);
      context.beginPath();
      context.moveTo(pa.x, pa.y);
      context.lineTo(pb.x, pb.y);
      context.stroke();
    }
    context.restore();
  }

  function drawPoints(
    context,
    points,
    indices,
    mapPoint,
    radius,
    color,
    isDrawablePoint = isDrawable,
  ) {
    context.save();
    context.fillStyle = color;
    for (const index of indices) {
      const point = points[index];
      if (!isDrawablePoint(point)) continue;
      const position = mapPoint(point);
      context.beginPath();
      context.arc(position.x, position.y, radius, 0, Math.PI * 2);
      context.fill();
    }
    context.restore();
  }

  function draw(
    canvas,
    results,
    sourceWidth,
    sourceHeight,
    viewportWidth,
    viewportHeight,
    trackingState,
  ) {
    if (
      !canvas ||
      !results ||
      sourceWidth <= 0 ||
      sourceHeight <= 0 ||
      !viewportWidth ||
      !viewportHeight
    )
      return;
    const { width, height } = PIPELINE.trackingFrameSize(
      sourceWidth,
      sourceHeight,
      trackingState,
    );
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    const scale = Math.max(
      viewportWidth / sourceWidth,
      viewportHeight / sourceHeight,
    );
    const offsetX = (viewportWidth - sourceWidth * scale) / 2;
    const offsetY = (viewportHeight - sourceHeight * scale) / 2;
    const mapPoint = (point) => ({
      x: ((point.x * sourceWidth * scale + offsetX) * width) / viewportWidth,
      y: ((point.y * sourceHeight * scale + offsetY) * height) / viewportHeight,
    });
    const pose = results.poseLandmarks || [],
      visiblePose = hasTorsoAnchor(pose) ? pose : [],
      left = results.leftHandLandmarks || [],
      right = results.rightHandLandmarks || [],
      face = results.faceLandmarks || [];
    const lineWidth = (weight) => Math.max(1.3, (width / 600) * weight),
      radius = Math.max(1.3, width / 850);
    drawConnections(
      context,
      visiblePose,
      POSE_CONNECTIONS,
      mapPoint,
      "#5ebdff",
      lineWidth(2),
      isPoseDrawable,
    );
    drawPoints(
      context,
      visiblePose,
      POSE_STATUS_IDS,
      mapPoint,
      radius,
      "#c8eaff",
      isPoseDrawable,
    );
    drawConnections(
      context,
      left,
      HAND_CONNECTIONS,
      mapPoint,
      "#b8a3ff",
      lineWidth(2.7),
    );
    drawPoints(
      context,
      left,
      left.map((_, index) => index),
      mapPoint,
      radius,
      "#e0d8ff",
    );
    drawConnections(
      context,
      right,
      HAND_CONNECTIONS,
      mapPoint,
      "#43dfad",
      lineWidth(2.7),
    );
    drawPoints(
      context,
      right,
      right.map((_, index) => index),
      mapPoint,
      radius,
      "#a7f3d0",
    );
    drawConnections(
      context,
      face,
      FACE_CONNECTIONS,
      mapPoint,
      "#ffcf70",
      lineWidth(1.1),
    );
    drawPoints(context, face, FACE_LANDMARK_IDS, mapPoint, radius, "#ffe2a6");
  }

  window.SignARLandmarkRenderer = Object.freeze({ draw });
})();
