/* Pinned MediaPipe Tasks Vision HandLandmarker lifecycle shared by collector and decoder. */
(() => {
  "use strict";
  const PIPELINE = window.SignARPipeline;
  const TIMEOUT_MS = 12000;

  function createTracker({ onWarning = () => {} } = {}) {
    let closed = false;
    let frameProcessing = false;
    let initializationPromise = null;
    let landmarker = null;
    let frameCount = 0;
    let averageFrameMs = 0;
    let lastFrameAt = 0;
    let lastVideoTimestamp = -1;
    const trackerStartAt = performance.now();

    async function loadTasksVision() {
      const failures = [];
      for (const source of PIPELINE.config.assets.tasksVision) {
        try {
          return await import(source);
        } catch (error) {
          failures.push(`${source}: ${error.message}`);
        }
      }
      throw new Error(
        `Unable to load pinned MediaPipe Tasks Vision. ${failures.join(" ")}`,
      );
    }

    async function initialize() {
      if (closed) throw new Error("The tracker has been closed.");
      if (landmarker) return landmarker;
      if (!initializationPromise) {
        initializationPromise = (async () => {
          const vision = await loadTasksVision();
          let lastError = null;
          for (const wasmUrl of PIPELINE.config.assets.handWasm) {
            try {
              const files =
                await vision.FilesetResolver.forVisionTasks(wasmUrl);
              try {
                landmarker = await vision.HandLandmarker.createFromOptions(
                  files,
                  {
                    baseOptions: {
                      modelAssetPath: PIPELINE.config.assets.handModel,
                      delegate: "GPU",
                    },
                    runningMode: "VIDEO",
                    ...PIPELINE.config.tracking.handOptions,
                  },
                );
              } catch (gpuError) {
                onWarning(
                  "GPU hand tracking could not initialize; trying CPU.",
                  gpuError,
                );
                landmarker = await vision.HandLandmarker.createFromOptions(
                  files,
                  {
                    baseOptions: {
                      modelAssetPath: PIPELINE.config.assets.handModel,
                      delegate: "CPU",
                    },
                    runningMode: "VIDEO",
                    ...PIPELINE.config.tracking.handOptions,
                  },
                );
              }
              if (landmarker) return landmarker;
            } catch (error) {
              lastError = error;
            }
          }
          throw new Error(
            `Could not initialize MediaPipe hand tracking: ${lastError?.message || "all pinned WASM sources failed"}`,
          );
        })();
      }
      try {
        return await Promise.race([
          initializationPromise,
          new Promise((_, reject) =>
            setTimeout(
              () =>
                reject(
                  new Error(
                    "MediaPipe hand tracking initialization timed out.",
                  ),
                ),
              TIMEOUT_MS,
            ),
          ),
        ]);
      } catch (error) {
        initializationPromise = null;
        throw error;
      }
    }

    async function openCamera(facingMode = "environment") {
      const attempts = [
        PIPELINE.cameraConstraints(facingMode),
        PIPELINE.cameraConstraints(facingMode, false),
        PIPELINE.cameraConstraints(null),
      ];
      let lastError;
      for (const constraints of attempts) {
        try {
          return await navigator.mediaDevices.getUserMedia(constraints);
        } catch (error) {
          lastError = error;
          if (
            error?.name === "NotAllowedError" ||
            error?.name === "SecurityError"
          )
            break;
        }
      }
      throw lastError || new Error("Could not open a camera.");
    }

    async function processFrame(image) {
      if (closed) throw new Error("The tracker has been closed.");
      if (frameProcessing) {
        const error = new Error(
          "The hand tracker can process only one frame at a time.",
        );
        error.name = "InvalidStateError";
        throw error;
      }
      frameProcessing = true;
      try {
        const detector = await initialize();
        const started = performance.now();
        const elapsedVideoMs = Math.max(0, started - trackerStartAt);
        const timestamp = Math.max(elapsedVideoMs, lastVideoTimestamp + 0.001);
        lastVideoTimestamp = timestamp;
        const result = detector.detectForVideo(image, timestamp);
        const merged = PIPELINE.mergeHandResults(null, result);
        const elapsed = performance.now() - started;
        frameCount++;
        averageFrameMs = averageFrameMs
          ? averageFrameMs * 0.8 + elapsed * 0.2
          : elapsed;
        lastFrameAt = performance.now();
        return {
          ...merged,
          signarHandSource: "MediaPipe Tasks Vision HandLandmarker (GPU/CPU)",
          signarLandmarkMs: elapsed,
          signarHandSlots: {
            left: PIPELINE.hasUsableHand(merged.leftHandLandmarks),
            right: PIPELINE.hasUsableHand(merged.rightHandLandmarks),
          },
        };
      } finally {
        frameProcessing = false;
      }
    }

    async function close() {
      if (closed) return;
      closed = true;
      const active = landmarker;
      landmarker = null;
      initializationPromise = null;
      frameProcessing = false;
      try {
        active?.close?.();
      } catch (error) {
        onWarning("Could not close MediaPipe hand tracker.", error);
      }
    }

    return Object.freeze({
      initialize,
      serializeInitialization: (operation) => operation(),
      openCamera,
      processFrame,
      close,
      get frameCount() {
        return frameCount;
      },
      get averageFrameMs() {
        return averageFrameMs;
      },
      get lastFrameAt() {
        return lastFrameAt;
      },
      get closed() {
        return closed;
      },
    });
  }

  window.SignARTracker = Object.freeze({ create: createTracker });
})();
