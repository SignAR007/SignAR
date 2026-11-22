const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

global.window = {};
const root = path.join(__dirname, '..', '..', 'SignLanguageAR', 'src');
vm.runInThisContext(fs.readFileSync(path.join(root, 'core', 'fsl-pipeline.js'), 'utf8'));

const optionCalls = {};
const hand = Array.from({ length: 21 }, (_, index) => ({
  x: 0.2 + index / 1000,
  y: 0.3 + index / 1000,
  z: 0
}));

window.Holistic = class {
  setOptions(options) { optionCalls.holistic = options; }
  onResults(callback) { this.callback = callback; }
  async initialize() {}
  async send() {
    this.callback({ leftHandLandmarks: [], rightHandLandmarks: [] });
  }
};

window.Hands = class {
  setOptions(options) { optionCalls.hands = options; }
  onResults(callback) { this.callback = callback; }
  async initialize() {}
  async send() {
    this.callback({
      multiHandLandmarks: [hand, hand.map(point => ({ ...point, x: 1 - point.x }))],
      multiHandedness: [
        { classification: [{ label: 'Left' }] },
        { classification: [{ label: 'Right' }] }
      ]
    });
  }
};

vm.runInThisContext(fs.readFileSync(path.join(root, 'tracker', 'mediapipe-tracker.js'), 'utf8'));

(async () => {
  const warnings = [];
  const tracker = window.SignARTracker.create({
    onWarning: (...warning) => warnings.push(warning)
  });
  await tracker.initialize();
  const result = await tracker.processFrame({}, { waitForHands: true });

  assert.deepEqual(optionCalls.holistic, window.SignARPipeline.config.tracking.holisticOptions);
  assert.deepEqual(optionCalls.hands, window.SignARPipeline.config.tracking.handsOptions);
  assert.equal(result.rightHandLandmarks, hand, 'unmirrored input should correct the mirrored-input Hands classification');
  assert.deepEqual(result.leftHandLandmarks, hand.map(point => ({ ...point, x: 1 - point.x })));
  assert.deepEqual(warnings, []);

  const savedHolistic=window.Holistic;
  const pose=Array.from({length:33},()=>({x:.5,y:.5,visibility:0}));
  pose[15]={x:.25,y:.5,visibility:1};
  pose[16]={x:.75,y:.5,visibility:1};
  const anatomicalLeft=Array.from({length:21},(_,index)=>({x:.25+index*.0001,y:.5,z:0}));
  const anatomicalRight=Array.from({length:21},(_,index)=>({x:.75+index*.0001,y:.5,z:0}));
  window.Holistic=class {
    setOptions() {}
    onResults(callback) { this.callback=callback; }
    async initialize() {}
    async send() {
      this.callback({poseLandmarks:pose,leftHandLandmarks:anatomicalRight,rightHandLandmarks:anatomicalLeft});
    }
  };
  const poseAlignedTracker=window.SignARTracker.create();
  const alignedResult=await poseAlignedTracker.processFrame({width:100,height:100});
  assert.equal(alignedResult.leftHandLandmarks,anatomicalLeft,'Holistic hand lists should be associated with the matching pose wrist');
  assert.equal(alignedResult.rightHandLandmarks,anatomicalRight,'Holistic hand lists should be associated with the matching pose wrist');
  window.Holistic=savedHolistic;

  const requests = [];
  const stream = { id: 'camera-stream' };
  const mediaDevices = {
    async getUserMedia(constraints) {
      requests.push(constraints);
      if (requests.length === 1) throw { name: 'OverconstrainedError' };
      return stream;
    }
  };
  assert.equal(await tracker.openCamera('environment', mediaDevices), stream);
  assert.deepEqual(requests, [
    window.SignARPipeline.cameraConstraints('environment'),
    window.SignARPipeline.cameraConstraints('environment', false)
  ]);

  const fallbackRequests = [];
  const fallbackStream = { id: 'fallback-camera-stream' };
  const fallbackMediaDevices = {
    async getUserMedia(constraints) {
      fallbackRequests.push(constraints);
      if (fallbackRequests.length < 3) throw { name: 'OverconstrainedError' };
      return fallbackStream;
    }
  };
  assert.equal(await tracker.openCamera('user', fallbackMediaDevices), fallbackStream);
  assert.deepEqual(fallbackRequests, [
    window.SignARPipeline.cameraConstraints('user'),
    window.SignARPipeline.cameraConstraints('user', false),
    { video: true, audio: false }
  ], 'camera acquisition should relax constraints in stages');

  const permissionError = { name: 'NotAllowedError', message: 'Permission denied.' };
  let permissionAttempts = 0;
  await assert.rejects(
    tracker.openCamera('environment', {
      async getUserMedia() {
        permissionAttempts++;
        throw permissionError;
      }
    }),
    error => error === permissionError
  );
  assert.equal(permissionAttempts, 1, 'permission failures should not trigger misleading fallback requests');

  const originalHolistic = window.Holistic;
  const originalDocument = global.document;
  const failedScripts = [];
  window.Holistic = undefined;
  global.document = {
    createElement() {
      return { remove() { this.removed = true; } };
    },
    head: {
      appendChild(script) {
        failedScripts.push(script);
        script.onerror();
      }
    }
  };
  await assert.rejects(window.SignARTracker.create().initialize(), /Unable to load MediaPipe holistic/);
  assert.equal(failedScripts.length, window.SignARPipeline.config.assets.holistic.length, 'library loading should try each configured CDN');
  assert.ok(failedScripts.every(script => script.removed), 'failed script elements should be cleaned up');
  window.Holistic = originalHolistic;
  if (originalDocument === undefined) delete global.document;
  else global.document = originalDocument;

  let handSendAttempts = 0;
  window.Hands = class {
    setOptions() {}
    onResults(callback) { this.callback = callback; }
    async initialize() {}
    async send() {
      handSendAttempts++;
      if (handSendAttempts === 1) throw new Error('Temporary Hands failure');
      this.callback({
        multiHandLandmarks: [hand],
        multiHandedness: [{ classification: [{ label: 'Left' }] }]
      });
    }
  };
  const retryWarnings = [];
  const retryTracker = window.SignARTracker.create({
    onWarning: (...warning) => retryWarnings.push(warning)
  });
  const firstFallback = await retryTracker.processFrame({}, { waitForHands: true });
  assert.deepEqual(firstFallback.leftHandLandmarks, [], 'failed fallback should preserve Holistic output');
  assert.equal(retryTracker.handsUnavailable, false, 'one transient Hands failure should not disable fallback for the session');
  const recoveredFallback = await retryTracker.processFrame({}, { waitForHands: true });
  assert.equal(recoveredFallback.rightHandLandmarks, hand, 'the fallback should recover with corrected handedness on the next frame');
  assert.equal(retryTracker.handsUnavailable, false);
  assert.equal(retryWarnings.length, 1);
  assert.match(retryWarnings[0][0], /retry on the next frame/);

  const holisticInstances = [];
  window.Holistic = class {
    constructor() { holisticInstances.push(this); }
    setOptions() {}
    onResults(callback) { this.callback = callback; }
    async initialize() {}
    async send() {
      if (holisticInstances.length === 1) throw new Error('First tracker instance failed');
    }
  };
  const staleTracker = window.SignARTracker.create({ onWarning: () => {} });
  await assert.rejects(staleTracker.processFrame({}), /First tracker instance failed/);
  const nextFrame = staleTracker.processFrame({});
  await new Promise(resolve => setImmediate(resolve));
  let nextFrameSettled = false;
  const observedNextFrame = nextFrame.then(results => {
    nextFrameSettled = true;
    return results;
  });
  holisticInstances[0].callback({ leftHandLandmarks: hand, rightHandLandmarks: [] });
  await Promise.resolve();
  assert.equal(nextFrameSettled, false, 'late callbacks from a closed tracker must not resolve a newer frame');
  holisticInstances[1].callback({ leftHandLandmarks: hand, rightHandLandmarks: [] });
  await observedNextFrame;

  let releaseFrame;
  window.Holistic = class {
    setOptions() {}
    onResults(callback) { this.callback = callback; }
    async initialize() {}
    async send() {
      await new Promise(resolve => { releaseFrame = resolve; });
      this.callback({ leftHandLandmarks: hand, rightHandLandmarks: [] });
    }
  };
  const serializedTracker = window.SignARTracker.create();
  await serializedTracker.initialize();
  const activeFrame = serializedTracker.processFrame({});
  await new Promise(resolve => setImmediate(resolve));
  await assert.rejects(
    serializedTracker.processFrame({}),
    error => error.name === 'InvalidStateError'
  );
  releaseFrame();
  await activeFrame;

  const initializationTrackers = [window.SignARTracker.create(), window.SignARTracker.create()];
  let activeInitializations = 0;
  let maximumConcurrentInitializations = 0;
  const initializeSerially = async () => {
    activeInitializations++;
    maximumConcurrentInitializations = Math.max(maximumConcurrentInitializations, activeInitializations);
    await new Promise(resolve => setTimeout(resolve, 5));
    activeInitializations--;
  };
  await Promise.all([
    initializationTrackers[0].serializeInitialization(initializeSerially),
    initializationTrackers[1].serializeInitialization(initializeSerially)
  ]);
  assert.equal(maximumConcurrentInitializations, 1, 'MediaPipe solution initialization must be globally serialized');
  await assert.rejects(
    initializationTrackers[0].serializeInitialization(async () => { throw new Error('Initialization failed'); }),
    /Initialization failed/
  );
  await initializationTrackers[1].serializeInitialization(initializeSerially);
  assert.equal(maximumConcurrentInitializations, 1, 'a failed initialization must not block later solutions');

  let integratedInitializations = 0;
  let maximumIntegratedInitializations = 0;
  const initializeIntegratedSolution = async () => {
    integratedInitializations++;
    maximumIntegratedInitializations = Math.max(maximumIntegratedInitializations, integratedInitializations);
    await new Promise(resolve => setTimeout(resolve, 5));
    integratedInitializations--;
  };
  window.Holistic = class {
    setOptions() {}
    onResults() {}
    initialize() { return initializeIntegratedSolution(); }
  };
  const integratedTracker = window.SignARTracker.create();
  await Promise.all([
    integratedTracker.initialize(),
    integratedTracker.serializeInitialization(initializeIntegratedSolution)
  ]);
  assert.equal(maximumIntegratedInitializations, 1, 'Holistic and optional MediaPipe solutions must share the same initialization queue');
  await integratedTracker.close();
  assert.equal(integratedTracker.closed, true);
  await assert.rejects(
    integratedTracker.processFrame({}),
    /tracker has been closed/
  );
  console.log('Shared MediaPipe tracker tests passed.');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
