(() => {
  'use strict';

  const PIPELINE = window.SignARPipeline;
  const SEQUENCE_LENGTH = PIPELINE.config.sequence.length;
  const FEATURE_DIMENSION = PIPELINE.config.features.dimension;

  function canonicalizeFrame(frame) {
    if (!Array.isArray(frame)) throw new Error('Feature frame must be an array.');
    const sourceVersion = frame.length === FEATURE_DIMENSION
      ? PIPELINE.config.features.version
      : frame.length === 68
        ? 'signar-fsl-hand-v3'
        : frame.length === 456
          ? 'signar-fsl-456-v2'
          : '';
    if (!sourceVersion) throw new Error(`Expected ${FEATURE_DIMENSION} holistic-v4 features or a supported earlier feature frame.`);
    return PIPELINE.migrateFrame(frame, sourceVersion);
  }

  function canonicalizeFrames(frames) {
    if (!Array.isArray(frames) || !frames.length) throw new Error('A sample needs feature frames.');
    return frames.map(canonicalizeFrame);
  }

  function metadataForLabel(label, sample = {}) {
    const normalized = String(label || '').trim().toUpperCase();
    const inferredUnitType = normalized === 'NO SIGN'
      ? 'background'
      : /^[A-Z]$/.test(normalized) ? 'fingerspelling' : 'word';
    return {
      unitType: normalized === 'NO SIGN' ? 'background' : (sample.unitType || inferredUnitType),
      motionType: normalized === 'NO SIGN' ? 'static' : (sample.motionType || (/^[JZ]$/.test(normalized) ? 'dynamic' : 'static')),
      displayText: String(sample.displayText || '').trim()
    };
  }

  function prepareDataset(samples, preferredLabels, extractor = PIPELINE) {
    if (!extractor?.resampleSequence) throw new Error('The shared SignAR feature resampler is unavailable.');
    const usable = [];
    const invalidSamples = [];

    for (const sample of samples) {
      const label = String(sample?.label ?? '').trim().toUpperCase();
      if (!label) {
        invalidSamples.push({ id: sample?.id, reason: 'Missing sign label' });
        continue;
      }
      try {
        const sequence = extractor.resampleSequence(canonicalizeFrames(sample.features), SEQUENCE_LENGTH);
        if (sequence.some(frame => frame.length !== FEATURE_DIMENSION || frame.some(value => !Number.isFinite(value)))) {
          throw new Error('Invalid holistic-v4 feature sequence');
        }
        if (label !== 'NO SIGN' && !sequence.some(frame => extractor.hasAnyFeatureRegion(frame))) {
          throw new Error('The sign sample contains no tracked hand, face, or body cues.');
        }
        usable.push({
          id: sample.id,
          label,
          signerId: String(sample.signerId ?? '').trim(),
          sequence
        });
      } catch (error) {
        invalidSamples.push({ id: sample?.id, reason: error.message || 'Invalid landmark sequence' });
      }
    }

    if (!usable.length) throw new Error('No saved samples have a valid holistic sign sequence. Record or import a sign first.');

    const observed = new Set(usable.map(sample => sample.label));
    const labels = [...new Set([
      ...(Array.isArray(preferredLabels) ? preferredLabels : []).map(label => String(label).trim().toUpperCase()),
      ...usable.map(sample => sample.label)
    ])].filter(label => label && observed.has(label));
    const labelIndex = new Map(labels.map((label, index) => [label, index]));

    return {
      samples: usable.map(sample => ({ ...sample, classIndex: labelIndex.get(sample.label) })),
      labels,
      invalidSamples
    };
  }

  function createDatasetDocument(samples, preferredLabels = []) {
    const observedLabels = [...new Set(samples
      .map(sample => String(sample?.label ?? '').trim().toUpperCase())
      .filter(Boolean))];
    const observed = new Set(observedLabels);
    const labels = [...new Set([
      ...(Array.isArray(preferredLabels) ? preferredLabels : [])
        .map(label => String(label ?? '').trim().toUpperCase()),
      ...observedLabels
    ])].filter(label => label && observed.has(label));
    const labelMetadata = Object.fromEntries(labels.map(label => {
      const matching = samples.filter(sample => String(sample?.label ?? '').trim().toUpperCase() === label);
      const source = matching.find(sample => sample?.displayText) || matching[0] || {};
      return [label, metadataForLabel(label, source)];
    }));

    return {
      schemaVersion: 4,
      project: 'SignAR FSL Sign Recognition',
      featureVersion: PIPELINE.config.features.version,
      featureDimension: FEATURE_DIMENSION,
      samplingFps: PIPELINE.config.sequence.samplingFps,
      sequenceLength: SEQUENCE_LENGTH,
      featuresResampled: true,
      rejectLabel: 'NO SIGN',
      labels,
      labelMetadata,
      exportedAt: new Date().toISOString(),
      samples: samples.map(sample => ({
        id: sample.id,
        signerId: sample.signerId,
        label: sample.label,
        ...metadataForLabel(sample.label, sample),
        createdAt: new Date(sample.createdAt).toISOString(),
        verificationNotes: sample.verificationNotes || '',
        verified: !!sample.verified,
        consentRecorded: !!sample.consentRecorded,
        sourceType: sample.sourceType || 'camera',
        frameCount: sample.frameCount,
        durationMs: sample.durationMs,
        features: PIPELINE.resampleSequence(canonicalizeFrames(sample.features), SEQUENCE_LENGTH)
      }))
    };
  }

  function splitDataset(samples, labels, seed = 42) {
    let state = seed >>> 0;
    const random = () => {
      state = (1664525 * state + 1013904223) >>> 0;
      return state / 4294967296;
    };
    const train = [];
    const validation = [];

    for (const label of labels) {
      const group = samples.filter(sample => sample.label === label);
      for (let i = group.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [group[i], group[j]] = [group[j], group[i]];
      }
      const validationCount = group.length > 1 ? Math.max(1, Math.floor(group.length * 0.2)) : 0;
      train.push(...group.slice(0, group.length - validationCount));
      validation.push(...group.slice(group.length - validationCount));
    }

    return { train, validation };
  }

  function tensorData(tf, samples, singleLabel) {
    const x = PIPELINE.createBatchTensor(tf, samples.map(sample => sample.sequence));
    const y = singleLabel
      ? tf.tensor2d(new Float32Array(samples.length).fill(1), [samples.length, 1])
      : tf.tensor1d(Float32Array.from(samples, sample => sample.classIndex));
    return { x, y };
  }

  async function predict(model, frames, labels) {
    const tf = window.tf;
    if (!tf?.tensor3d) throw new Error('TensorFlow.js could not prepare the test sequence.');
    if (!Array.isArray(labels) || labels.length < 1) throw new Error('At least one sign label is needed for prediction.');
    const input = PIPELINE.createSequenceTensor(tf, frames);
    let predictions = null;
    try {
      predictions = model.predict(input);
      if (Array.isArray(predictions)) {
        predictions.forEach(tensor => tensor.dispose());
        predictions = null;
        throw new Error('The saved model returned an unexpected set of predictions.');
      }
      const scores = Array.from(await predictions.data());
      if (scores.length !== labels.length || scores.some(score => !Number.isFinite(score))) {
        throw new Error('The model returned invalid prediction scores.');
      }
      return labels.map((label, index) => ({ label, score: scores[index] })).sort((a, b) => b.score - a.score);
    } finally {
      if (predictions) predictions.dispose();
      input.dispose();
    }
  }

  function buildModel(tf, labels) {
    const sequence = tf.input({
      shape: PIPELINE.sequenceTensorShape(null).slice(1),
      name: 'landmark_sequence'
    });
    const shape = tf.layers.dense({units: 48, activation: 'relu', name: 'hand_shape_encoder'}).apply(sequence);
    const pooledShape = tf.layers.globalAveragePooling1d({name: 'shape_sequence_pool'}).apply(shape);
    const movement = tf.layers.gru({
      units: 24, dropout: 0.10, resetAfter: false,
      recurrentInitializer: 'orthogonal', name: 'temporal_sign_encoder'
    }).apply(sequence);
    const combined = tf.layers.concatenate({name: 'shape_and_motion'}).apply([pooledShape, movement]);
    const logits = tf.layers.dense({units: labels.length, name: 'sign_logits'}).apply(combined);
    const probabilities = tf.layers.activation({activation: 'softmax', name: 'sign_class_probabilities'}).apply(logits);
    const model = tf.model({
      inputs: sequence,
      outputs: probabilities,
      name: 'signar_isolated_sign_sequence_classifier'
    });
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });
    return model;
  }

  function evaluateValidation(probabilities, samples, labels) {
    if (!samples.length || labels.length < 2) return null;
    const confusionMatrix = labels.map(() => labels.map(() => 0));
    let correct = 0;
    samples.forEach((sample, sampleIndex) => {
      let predicted = 0;
      for (let labelIndex = 1; labelIndex < labels.length; labelIndex++) {
        if (probabilities[sampleIndex * labels.length + labelIndex] >
            probabilities[sampleIndex * labels.length + predicted]) predicted = labelIndex;
      }
      confusionMatrix[sample.classIndex][predicted]++;
      if (predicted === sample.classIndex) correct++;
    });
    return {
      accuracy: correct / samples.length,
      sampleCount: samples.length,
      confusionMatrix
    };
  }

  async function train(samples, preferredLabels, options = {}) {
    const tf = window.tf;
    if (!tf?.layers?.gru) throw new Error('TensorFlow.js could not load. Check your internet connection and refresh this page.');
    const epochs = Number(options.epochs);
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 200) throw new Error('Choose between 1 and 200 training epochs.');

    const dataset = prepareDataset(samples, preferredLabels);
    const partitions = splitDataset(dataset.samples, dataset.labels);
    if (!partitions.train.length) throw new Error('At least one usable sample is needed for training.');
    const singleLabel = false;

    await tf.ready();
    const trainTensors = tensorData(tf, partitions.train, false);
    const validationTensors = partitions.validation.length
      ? tensorData(tf, partitions.validation, false)
      : null;
    const model = buildModel(tf, dataset.labels);
    let trainingSucceeded = false;
    let finalMetrics = { accuracy: null, loss: null, validationAccuracy: null, validationLoss: null };

    try {
      const fitOptions = {
        epochs,
        batchSize: Math.min(16, partitions.train.length),
        shuffle: true,
        verbose: 0,
        callbacks: [{
        onEpochEnd: async (epoch, logs) => {
            const accuracy=Number.isFinite(logs?.accuracy)?logs.accuracy:Number.isFinite(logs?.acc)?logs.acc:null;
            if(accuracy===null)throw new Error('TensorFlow.js did not report training accuracy.');
            const validationMetric=Number.isFinite(logs?.val_accuracy)?logs.val_accuracy:Number.isFinite(logs?.val_acc)?logs.val_acc:null;
            if(validationTensors&&validationMetric===null)throw new Error('TensorFlow.js did not report validation accuracy.');
            finalMetrics = {
              accuracy,
              loss: Number.isFinite(logs?.loss) ? logs.loss : null,
              validationAccuracy:validationTensors?validationMetric:null,
              validationLoss: Number.isFinite(logs?.val_loss) ? logs.val_loss : null
            };
            if (options.onEpoch) {
              await options.onEpoch({ epoch: epoch + 1, epochs, singleLabel, ...finalMetrics });
            }
          }
        }]
      };
      if (validationTensors) fitOptions.validationData = [validationTensors.x, validationTensors.y];
      await model.fit(trainTensors.x, trainTensors.y, fitOptions);

      let evaluation = null;
      if (validationTensors) {
        const predictions = model.predict(validationTensors.x);
        try {
          evaluation = evaluateValidation(
            await predictions.data(), partitions.validation, dataset.labels
          );
        } finally {
          predictions.dispose();
        }
      }

      trainingSucceeded = true;
      return {
        model,
        labels: dataset.labels,
        invalidSamples: dataset.invalidSamples,
        sampleCount: dataset.samples.length,
        trainCount: partitions.train.length,
        validationCount: partitions.validation.length,
        singleLabel: false,
        metrics: finalMetrics,
        evaluation
      };
    } finally {
      trainTensors.x.dispose();
      trainTensors.y.dispose();
      if (validationTensors) {
        validationTensors.x.dispose();
        validationTensors.y.dispose();
      }
      if (!trainingSucceeded) model.dispose();
    }
  }

  window.SignARBrowserTrainer = Object.freeze({ canonicalizeFrame, createDatasetDocument, prepareDataset, splitDataset, predict, train });
})();
