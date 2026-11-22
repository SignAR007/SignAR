/* Node.js tests for browser training data preparation; no TensorFlow.js dependency required. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

global.window = {};
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'core', 'fsl-pipeline.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'src', 'collector', 'browser-trainer.js'), 'utf8'));

const trainer = global.window.SignARBrowserTrainer;
const collectorSource = fs.readFileSync(path.join(__dirname, '..', 'src', 'collector', 'collect.js'), 'utf8');
const pythonTrainerSource = fs.readFileSync(path.join(__dirname, 'train_model.py'), 'utf8');
assert.match(collectorSource, /new Map\(\[\['NO SIGN', 'NO SIGN'\]\]\)/,
  'the collector should keep the reserved background label available even before any examples are saved');
assert.match(collectorSource, /'X-Training-Job-ID':jobId/, 'the browser should identify each local training job for reconnects');
assert.match(collectorSource, /progress\.jobId!==jobId[\s\S]*?progress\.completed&&progress\.result/, 'the browser should poll the matching job until its result is available');
assert.match(collectorSource, /startResult\?\.completed===true\)completedResult=startResult/, 'the browser should accept a completed response from an older synchronous local trainer');
assert.match(collectorSource, /progress\.active&&!progress\.jobId[\s\S]*?ActiveLegacyTrainingError/, 'an active older-server run should not be mislabeled as stopped or restarted');
assert.match(collectorSource, /progress\.output[\s\S]*?progress\.error\|\|progress\.stage/, 'training failures should show the server error and captured trainer output');
assert.match(pythonTrainerSource, /Dense\(\s*64, activation="relu", kernel_regularizer=regularizer, name="frame_shape_encoder"/, 'Python training should use a regularized static handshape encoder');
assert.match(pythonTrainerSource, /GlobalAveragePooling1D\(name="static_handshape_pool"\)/, 'Python training should pool the static shape branch');
assert.match(pythonTrainerSource, /Conv1D\(48, 5, padding="same", activation="relu",\s+kernel_regularizer=regularizer/, 'Python training should regularize its order-preserving temporal encoder');
assert.match(pythonTrainerSource, /ModelCheckpoint\(/, 'Python training should checkpoint the best model');
assert.match(pythonTrainerSource, /"sequence_length": SEQ_LEN/, 'Python training should write the dynamic interpreter input metadata');
assert.match(pythonTrainerSource, /Concatenate\(name="shape_and_motion"\)/, 'Python training should combine shape and motion encoders');
const frames = [new Array(456).fill(0), new Array(456).fill(1)];
const datasetDocument = trainer.createDatasetDocument([
  { id: 'existing', label: 'KAMUSTA KA', createdAt: 0, features:frames },
  { id: 'new-a', label: 'A', createdAt: 1, features:frames },
  { id: 'new-b', label: 'I love you', createdAt: 2, features:frames },
  { id: 'duplicate', label: 'a', createdAt: 3, features:frames }
], ['KAMUSTA KA', 'UNUSED OLD LABEL']);
assert.deepEqual(datasetDocument.labels, ['KAMUSTA KA', 'A', 'I LOVE YOU'], 'export every observed label without stale unused vocabulary');
assert.equal(datasetDocument.samples.length, 4, 'preserve all saved samples in the export');
assert.equal(datasetDocument.featuresResampled, true, 'serialize the shared fixed-length sequence output');
const backgroundDataset = trainer.createDatasetDocument([
  { id: 'sign-a', label: 'A', createdAt: 4, features: frames },
  { id: 'background', label: 'no sign', createdAt: 5, features: frames }
]);
assert.deepEqual(backgroundDataset.labels, ['A', 'NO SIGN'], 'export NO SIGN only when labeled samples actually exist');
const staticImageFrame = new Array(456).fill(0.25);
const staticImageDataset = trainer.createDatasetDocument([{
  id: 'static-image', label: 'STATIC', signerId: 'signer-01', createdAt: 4,
  sourceType: 'imported-image', frameCount: 1, features: [staticImageFrame]
}]);
assert.equal(staticImageDataset.samples[0].sourceType, 'imported-image', 'preserve image provenance for static samples');
assert.equal(staticImageDataset.samples[0].frameCount, 1, 'keep the original one-frame sample count');
assert.equal(staticImageDataset.samples[0].features.length, 30, 'turn a still image into a fixed-length static model sequence');
assert.ok(staticImageDataset.samples[0].features.every(frame => frame.every(value => value === 0.25)));
const samples = [
  { id: 'one-a', label: 'word a', signerId: 'signer-01', features: frames },
  { id: 'two-a', label: 'WORD A', signerId: 'signer-02', features: frames },
  { id: 'one-b', label: 'word b', signerId: 'signer-01', features: frames },
  { id: 'two-b', label: 'WORD B', signerId: 'signer-02', features: frames },
  { id: 'bad', label: 'WORD A', features: [[0, 1]] }
];

const prepared = trainer.prepareDataset(samples, ['WORD B', 'WORD A']);
assert.deepEqual(prepared.labels, ['WORD B', 'WORD A'], 'preserve preferred label order for observed classes');
assert.equal(prepared.samples.length, 4, 'skip invalid feature sequences');
assert.equal(prepared.samples[0].sequence.length, 30, 'resample each clip to the model window');
assert.equal(prepared.samples[0].sequence[0].length, 456, 'preserve the feature width');
assert.equal(prepared.invalidSamples.length, 1, 'report invalid samples');

const split = trainer.splitDataset(prepared.samples, prepared.labels);
assert.equal(split.train.length + split.validation.length, prepared.samples.length, 'retain all valid samples');
assert.equal(split.train.length, 2, 'keep one training example for each class');
assert.equal(split.validation.length, 2, 'reserve held-out examples when available');
assert.deepEqual(trainer.splitDataset(prepared.samples, prepared.labels), split, 'produce a repeatable sample-level split');

assert.throws(
  () => trainer.prepareDataset([{ id: 'empty', label: 'WORD A', features: [] }], []),
  /No saved samples have a valid label/
);
console.log('Browser trainer data-preparation tests passed.');

(async () => {
  let inputDisposed = false;
  let predictionDisposed = false;
  global.window.tf = {
    tensor3d(values, shape) {
      assert.equal(values.length, 30 * 456, 'resample a test sign to the model input size');
      assert.deepEqual(shape, [1, 30, 456], 'include a single sequence batch');
      return { dispose() { inputDisposed = true; } };
    }
  };
  const predictions = await trainer.predict({
    predict() {
      return {
        async data() { return new Float32Array([0.2, 0.8]); },
        dispose() { predictionDisposed = true; }
      };
    }
  }, frames, ['WORD A', 'WORD B']);
  assert.deepEqual(predictions.map(result => result.label), ['WORD B', 'WORD A'], 'rank predictions by model score');
  assert.ok(Math.abs(predictions[0].score - 0.8) < 1e-6);
  assert.ok(Math.abs(predictions[1].score - 0.2) < 1e-6);
  assert.equal(inputDisposed, true, 'dispose the prepared input tensor');
  assert.equal(predictionDisposed, true, 'dispose the prediction tensor');
  const singleLabelPrediction = await trainer.predict({
    predict() {
      return {
        async data() { return new Float32Array([0.91]); },
        dispose() {}
      };
    }
  }, frames, ['WORD A']);
  assert.deepEqual(singleLabelPrediction.map(result => result.label), ['WORD A'], 'allow prediction with one vocabulary label');
  assert.ok(Math.abs(singleLabelPrediction[0].score - 0.91) < 1e-6);
  console.log('Browser trainer prediction tests passed.');

  let compileMetrics = null;
  let trainingPredictionCount = 0;
  let modelDefinition = null;
  const layerSpecs = [];
  const tensor = { dispose() {} };
  const layer = type => options => {
    layerSpecs.push({ type, options });
    return { apply(input) { return { type, input }; } };
  };
  const trainingModel = {
    compile(options) { compileMetrics = options.metrics; },
    async fit(_x, _y, options) {
      await options.callbacks[0].onEpochEnd(0, {
        loss: 0.4, acc: 0.75, val_loss: 0.5, val_acc: 0.5
      });
    },
    predict() {
      trainingPredictionCount++;
      return {
        async data() { return new Float32Array([0.8, 0.2, 0.2, 0.8]); },
        dispose() {}
      };
    }
  };
  global.window.tf = {
    input(options) { return { shape: options.shape, name: options.name }; },
    layers: {
      gru: layer('gru'),
      dense: layer('dense'),
      dropout: layer('dropout'),
      globalAveragePooling1d: layer('globalAveragePooling1d'),
      concatenate: layer('concatenate')
    },
    train: { adam: () => ({}) },
    model(options) { modelDefinition = options; return trainingModel; },
    async ready() {},
    tensor3d() { return tensor; },
    tensor1d() { return tensor; },
    async nextFrame() {}
  };
  const trained = await trainer.train(samples.slice(0, 4), ['WORD A', 'WORD B'], {
    epochs: 1
  });
  assert.deepEqual(compileMetrics, ['accuracy'], 'compute accuracy as part of model.fit');
  assert.equal(trainingPredictionCount, 1, 'only evaluate validation once after training, not again each epoch');
  assert.equal(trained.metrics.accuracy, 0.75, 'use fit training accuracy logs');
  assert.equal(trained.metrics.validationAccuracy, 0.5, 'use fit validation accuracy logs');
  assert.equal(trained.evaluation.accuracy, 1, 'preserve final validation evaluation');
  assert.deepEqual(layerSpecs.map(spec => spec.type), [
    'dense', 'globalAveragePooling1d', 'gru', 'gru', 'concatenate', 'dense', 'dropout', 'dense'
  ], 'use the same static-shape and temporal encoder layers as the Python trainer');
  assert.deepEqual(layerSpecs.map(spec => spec.options.name), [
    'frame_shape_encoder', 'static_handshape_pool', 'temporal_gru_1', 'temporal_gru_2',
    'shape_and_motion', 'representation', 'classifier_dropout', 'fsl_class_probabilities'
  ]);
  assert.deepEqual(modelDefinition.inputs.shape, [30, 456], 'share the model input shape with the interpreter');
  assert.equal(modelDefinition.name, 'signar_static_dynamic_classifier');
  assert.equal(layerSpecs[2].options.recurrentInitializer, 'orthogonal', 'match the Python GRU recurrent initializer');
  console.log('Browser trainer avoids redundant per-epoch predictions.');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
