/* Pinned MediaPipe Tasks Vision HolisticLandmarker lifecycle shared by collector and decoder. */
(() => {
  "use strict";
  const PIPELINE = window.SignARPipeline;
  const INITIALIZATION_TIMEOUT_MS = 90000;
  const ASSET_TIMEOUT_MS = 15000;

  function withTimeout(promise, timeoutMs, description) {
    let timer;
    return Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = window.setTimeout(
          () =>
            reject(
              new Error(
                description + " timed out after " +
                  Math.round(timeoutMs / 1000) + " seconds.",
              ),
            ),
          timeoutMs,
        );
      }),
    ]).finally(() => window.clearTimeout(timer));
  }

  function createTracker({ onWarning = () => {} } = {}) {
    let closed = false;
    let frameProcessing = false;
    let initializationPromise = null;
    let initializationStage = "MediaPipe Tasks Vision module";
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
          initializationStage = "MediaPipe Tasks Vision module from " + source;
          return await withTimeout(
            import(source),
            ASSET_TIMEOUT_MS,
            "Loading " + source,
          );
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
              initializationStage = "MediaPipe WebAssembly runtime from " + wasmUrl;
              const files = await withTimeout(
                vision.FilesetResolver.forVisionTasks(wasmUrl),
                ASSET_TIMEOUT_MS,
                "Loading MediaPipe WebAssembly from " + wasmUrl,
              );
              try {
                initializationStage = "HolisticLandmarker model on GPU";
                landmarker = await vision.HolisticLandmarker.createFromOptions(
                  files,
                  {
                    baseOptions: {
                      modelAssetPath: PIPELINE.config.assets.holisticModel,
                      delegate: "GPU",
                    },
                    runningMode: "VIDEO",
                    ...PIPELINE.config.tracking.holisticOptions,
                  },
                );
              } catch (gpuError) {
                onWarning(
                  "GPU holistic tracking could not initialize; trying CPU.",
                  gpuError,
                );
                initializationStage = "HolisticLandmarker model on CPU";
                landmarker = await vision.HolisticLandmarker.createFromOptions(
                  files,
                  {
                    baseOptions: {
                      modelAssetPath: PIPELINE.config.assets.holisticModel,
                      delegate: "CPU",
                    },
                    runningMode: "VIDEO",
                    ...PIPELINE.config.tracking.holisticOptions,
                  },
                );
              }
              if (landmarker) return landmarker;
            } catch (error) {
              lastError = error;
            }
          }
          throw new Error(
            `Could not initialize MediaPipe holistic tracking: ${lastError?.message || "all pinned WASM sources failed"}`,
          );
        })();
      }
      let timedOut = false;
      let timeoutId;
      try {
        return await Promise.race([
          initializationPromise,
          new Promise((_, reject) => {
            timeoutId = window.setTimeout(() => {
              timedOut = true;
              reject(
                new Error(
                  "MediaPipe holistic tracking initialization timed out after " +
                    Math.round(INITIALIZATION_TIMEOUT_MS / 1000) +
                    " seconds while loading " +
                    initializationStage +
                    ". Check the network requests to jsDelivr, unpkg, and storage.googleapis.com.",
                ),
              );
            }, INITIALIZATION_TIMEOUT_MS);
          }),
        ]);
      } catch (error) {
        // Keep a still-running initialization so another click does not start a
        // second model download. Clear it only after the actual initialization fails.
        if (!timedOut) initializationPromise = null;
        throw error;
      } finally {
        window.clearTimeout(timeoutId);
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
          "The holistic tracker can process only one frame at a time.",
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
        const merged = PIPELINE.mergeHolisticResults(result);
        const elapsed = performance.now() - started;
        frameCount++;
        averageFrameMs = averageFrameMs
          ? averageFrameMs * 0.8 + elapsed * 0.2
          : elapsed;
        lastFrameAt = performance.now();
        return {
          ...merged,
          signarHandSource: "MediaPipe Tasks Vision HolisticLandmarker (GPU/CPU)",
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
        onWarning("Could not close MediaPipe holistic tracker.", error);
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
