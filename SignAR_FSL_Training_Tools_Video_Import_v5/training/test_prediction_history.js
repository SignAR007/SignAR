const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

global.window = {};
const projectRoot = path.join(__dirname, '..', '..', 'SignLanguageAR');
const pipelinePath = path.join(projectRoot, 'src', 'core', 'fsl-pipeline.js');
const historyPath = path.join(projectRoot, 'src', 'interpreter', 'prediction-history.js');
const decoderPath = path.join(projectRoot, 'src', 'interpreter', 'app.js');
vm.runInThisContext(fs.readFileSync(pipelinePath, 'utf8'));
vm.runInThisContext(fs.readFileSync(historyPath, 'utf8'));

const policy = global.window.SignARPredictionHistory;
const pipeline = global.window.SignARPipeline;
const threshold = pipeline.config.recognition.minimumConfidencePercent;
const marginThreshold = pipeline.config.recognition.minimumMarginPercent;
const labels = Array.from('ZYXWVUTSRQPONMLKJIHGFEDCBA');
const decoderSource = fs.readFileSync(decoderPath, 'utf8');
const zIndex = labels.indexOf('Z');
assert.notEqual(zIndex, -1, 'the decoder label fixture should include Z');

function scoresFor(label, confidence) {
  const index = labels.indexOf(label);
  const remaining = (1 - confidence) / (labels.length - 1);
  return Float32Array.from(labels, (_, i) => i === index ? confidence : remaining);
}

const idleHistory = policy.create();
const uncertain = idleHistory.update(scoresFor('Z', 0.20), labels, threshold, marginThreshold);
assert.equal(uncertain.status, 'uncertain', 'a low-confidence closed-set top class must remain neutral');
assert.equal(uncertain.best.label, 'Z', 'low confidence can still have a top class in this positive-only vocabulary');
assert.equal(idleHistory.recent[0].accepted, false, 'low-confidence input must not count as a vote');
assert.equal(idleHistory.recent.length, 1, 'rejected windows remain visible in the bounded diagnostic history');
assert.equal(pipeline.hasUsableHand([]), false, 'an idle frame with no tracked hand must not enter sign inference');
assert.equal(pipeline.extract({ leftHandLandmarks: [], rightHandLandmarks: [] }), null,
  'an idle frame without usable body, face, or hand landmarks must not create a feature vector');
assert.match(decoderSource,/if\s*\(\s*!handsVisible\s*\)\s*\{\s*resetRecognitionSegment\(\);[\s\S]*?return;/,
  'the live decoder must clear its sequence and return NO SIGN before inference when no hand is tracked');
assert.match(decoderSource,/Loaded model asset:\s*["']\s*\+\s*modelConfig\.modelPath/,
  'the interpreter must expose which installed model produced the live predictions');

const signingHistory = policy.create();
const zWindow = scoresFor('Z', 0.99);
const firstWindow = signingHistory.update(zWindow, labels, threshold);
assert.equal(firstWindow.status, 'verifying', 'one model window is not enough to show a sign');
const secondWindow = signingHistory.update(zWindow, labels, threshold);
assert.equal(secondWindow.status, 'verifying', 'two highly overlapping windows are not enough to display a sign');
const heldStaticWindow = signingHistory.update(zWindow, labels, threshold);
assert.equal(heldStaticWindow.status, 'stable', 'a held static sign should stabilize without requiring motion');
assert.equal(heldStaticWindow.label, 'Z', 'consistent high-confidence Z windows should decode as Z');
assert.equal(heldStaticWindow.streak, 3, 'a displayed sign must have three qualifying votes in the latest five windows');

const noSignLabels = [...labels, 'NO SIGN'];
const noSignScores = Float32Array.from(noSignLabels, label => label === 'NO SIGN' ? 0.99 : 0.01 / labels.length);
const noSignHistory = policy.create();
noSignHistory.update(noSignScores, noSignLabels, threshold);
noSignHistory.update(noSignScores, noSignLabels, threshold);
const stableNoSign = noSignHistory.update(noSignScores, noSignLabels, threshold);
assert.equal(stableNoSign.status, 'stable', 'a trained background class should stabilize through the same evidence policy');
assert.equal(stableNoSign.label, 'NO SIGN', 'the decoder should receive the reserved background class label');

const staticSequenceRestartMs = policy.noPredictionRestartMs({
  sequenceLength: 30,
  sampleIntervalMs: 1000 / 15,
  failedSegmentRestartMs: 1400,
  maxInferenceIntervalMs: 180,
  requiredPredictions: policy.voteWindowSize
});
assert.ok(staticSequenceRestartMs > 30 * (1000 / 15),
  'a static segment must stay alive after its first full window becomes available');
assert.ok(30 * (1000 / 15) + (policy.voteWindowSize - 1) * 180 < staticSequenceRestartMs,
  'a static sign must have time to collect the five-window vote at the slowest supported inference cadence');
const sequenceFillMs = (30 - 1) * (1000 / 15);
assert.ok(sequenceFillMs + (policy.voteWindowSize - 1) * 180 < staticSequenceRestartMs,
  'the full model window must remain available long enough to fill the five-window vote');

const staleFrames = Array.from({ length: 35 }, (_, frame) => Array(456).fill(frame));
const recentTensor = pipeline.createRecentSequenceTensor({
  tensor3d(values, shape) { return { values, shape }; }
}, staleFrames);
assert.deepEqual(recentTensor.shape, [1, 30, 456], 'live inference must preserve the 30 × 456 TF.js contract');
assert.equal(recentTensor.values[0], 5, 'inference must not reuse frames that have fallen out of the recent window');
assert.equal(recentTensor.values[(30 - 1) * 456], 34, 'inference must use the newest sequence frame');

const commitState = {
  segmentActive: true,
  segmentCommitted: false,
  predictionStreak: policy.requiredStablePredictions,
  predictionLength: policy.voteWindowSize,
  signerVisible: true,
  quietMs: 800,
  quietToEndMs: 750
};
assert.equal(policy.shouldCommitPrediction(commitState), true,
  'a stable static sign can commit after the normal quiet period without requiring movement');
commitState.segmentCommitted = true;
assert.equal(policy.shouldCommitPrediction(commitState), false,
  'repeated stable predictions in one segment must not announce the same sign more than once');
commitState.segmentCommitted = false;
assert.equal(policy.shouldCommitPrediction(commitState), true,
  'a new segment may legitimately announce the same sign again');

const clearedHistory = policy.create();
clearedHistory.update(zWindow, labels, threshold);
clearedHistory.update(zWindow, labels, threshold);
assert.equal(clearedHistory.length, 2);
const afterIdle = clearedHistory.update(scoresFor('Z', 0.20), labels, threshold, marginThreshold);
assert.equal(afterIdle.status, 'uncertain');
assert.equal(clearedHistory.length, 3, 'rejected windows remain visible but do not count as votes');
const restoredVote = clearedHistory.update(zWindow, labels, threshold);
assert.equal(restoredVote.status, 'stable', 'three qualified votes in four windows should pass the 3-of-5 policy');
assert.equal(restoredVote.voteCount, 3);

const resetHistory = policy.create();
resetHistory.update(zWindow, labels, threshold);
resetHistory.update(zWindow, labels, threshold);
resetHistory.clear();
assert.equal(resetHistory.length, 0, 'a tracker or camera reset must discard buffered prediction evidence');
assert.equal(resetHistory.update(zWindow, labels, threshold).status, 'verifying',
  'a restarted segment must not inherit stale votes from the previous sequence');

const switchHistory = policy.create();
const yWindow = scoresFor('Y', 0.99);
assert.equal(switchHistory.update(zWindow, labels, threshold).streak, 1);
assert.equal(switchHistory.update(zWindow, labels, threshold).streak, 2);
const firstY = switchHistory.update(yWindow, labels, threshold);
assert.equal(firstY.status, 'verifying', 'a new class should compete within the rolling vote history');
assert.equal(firstY.label, 'Y');
assert.equal(firstY.streak, 1, 'one Y window should contribute one qualified vote');
assert.equal(switchHistory.update(yWindow, labels, threshold).status, 'verifying');
const firstZAgain = switchHistory.update(zWindow, labels, threshold);
assert.equal(firstZAgain.status, 'stable', 'three Z votes in the latest five windows should win');
assert.equal(firstZAgain.label, 'Z');
assert.equal(firstZAgain.streak, 3);
const yMajority = switchHistory.update(yWindow, labels, threshold);
assert.equal(yMajority.status, 'stable', 'the latest five windows should be able to switch to a Y majority');
assert.equal(yMajority.label, 'Y');

const otherLabelHistory = policy.create();
const aWindow = scoresFor('A', 0.99);
otherLabelHistory.update(aWindow, labels, threshold);
otherLabelHistory.update(aWindow, labels, threshold);
assert.equal(otherLabelHistory.update(aWindow, labels, threshold).label, 'A',
  'consecutive evidence should work for labels other than Z');

const closeScores = Float32Array.from(labels, (label) =>
  label === 'Z' ? 0.55 : label === 'Y' ? 0.45 : 0
);
const closeHistory = policy.create();
const closePrediction = closeHistory.update(closeScores, labels, 50, 12);
assert.equal(closePrediction.status, 'uncertain', 'a small top-two margin must reject a high enough top score');
assert.equal(closePrediction.scorePassed, true);
assert.equal(closePrediction.marginPassed, false);
assert.equal(closePrediction.voteCount, 0, 'low-margin outputs must not enter the vote count');

const repeatedHistory = policy.create();
const repeatedUpdates = Array.from({ length: 6 }, () => repeatedHistory.update(zWindow, labels, threshold));
assert.deepEqual(repeatedUpdates.slice(0, 2).map(result => result.status), ['verifying', 'verifying']);
assert.ok(repeatedUpdates.slice(2).every(result => result.status === 'stable' && result.label === 'Z'),
  'repeated qualifying predictions should keep the displayed label stable without counting older labels');
assert.equal(repeatedHistory.length, 5, 'prediction history should stay bounded');
assert.throws(() => policy.create().update([NaN], ['Z'], threshold), /invalid class probabilities/i,
  'invalid model values should fail explicitly rather than create a prediction');

console.log('Prediction history tests passed.');
