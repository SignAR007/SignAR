(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const video = $('camera');
  const canvas = $('landmarkCanvas');
  const stage = $('stage');
  const ctx = canvas.getContext('2d');
  const startCameraBtn = $('startCamera');
  const switchCameraBtn = $('switchCamera');
  const stopCameraBtn = $('stopCamera');
  const startRecordBtn = $('startRecording');
  const stopRecordBtn = $('stopRecording');
  const signerInput = $('signerId');
  const signerOptions = $('signerOptions');
  const labelSelect = $('labelSelect');
  const unitTypeSelect = $('unitType');
  const motionTypeSelect = $('motionType');
  const displayTextInput = $('displayText');
  const reviewerOptions = $('reviewerOptions');
  const verifyCheck = $('verified');
  const consentCheck = $('consent');
  const notesInput = $('verificationNotes');
  const autoCheck = $('autoCapture');
  const trackStatus = $('trackingStatus');
  const status = $('storageStatus');
  const stageLabel = $('stageLabel');
  const guide = $('guide');
  const recordState = $('recordState');
  const frameCounter = $('frameCounter');
  const frameDiagnostics = $('frameDiagnostics');
  const captureHelp = $('captureHelp');
  const sampleList = $('sampleList');
  const sampleSearch = $('sampleSearch');
  const sampleLabelFilter = $('sampleLabelFilter');
  const sampleSignerFilter = $('sampleSignerFilter');
  const sampleSort = $('sampleSort');
  const sampleFilterSummary = $('sampleFilterSummary');
  const selectFilteredSamples = $('selectFilteredSamples');
  const selectedSampleCount = $('selectedSampleCount');
  const downloadSelectedBtn = $('downloadSelected');
  const downloadProgress = $('downloadProgress');
  const downloadProgressLabel = $('downloadProgressLabel');
  const downloadProgressBar = $('downloadProgressBar');
  const deleteSelectedBtn = $('deleteSelected');
  const analyzeSamplesBtn = $('analyzeSamples');
  const applySampleCleanupBtn = $('applySampleCleanup');
  const sampleHealthSummary = $('sampleHealthSummary');
  const sampleCleanupDetails = $('sampleCleanupDetails');
  const sampleCleanupList = $('sampleCleanupList');
  const toast = $('toast');
  const importMediaInput = $('importMediaFile');
  const importProgress = $('importProgress');
  const importBatchStatus = $('importBatchStatus');
  const sampleSummary = $('sampleSummary');
  const exportJsonBtn = $('exportJson');
  const deleteAllBtn = $('deleteAll');
  const trainModelBtn = $('trainModel');
  const trainingMessage = $('trainingMessage');
  const trainingProgressWrap = $('trainingProgressWrap');
  const trainingProgress = $('trainingProgress');
  const trainingProgressLabel = $('trainingProgressLabel');
  const trainingEpochCount = $('trainingEpochCount');
  const trainingCoverage = $('trainingCoverage');
  const testAccuracy = $('testAccuracy');
  const testSampleCount = $('testSampleCount');
  const evaluationScopeMetric = $('evaluationScope');
  const deploymentEligibleMetric = $('deploymentEligible');
  const trainingEvaluationDetails = $('trainingEvaluationDetails');
  const trainingClassMetrics = $('trainingClassMetrics');
  const trainingEvaluationNote = $('trainingEvaluationNote');
  const deployModelBtn = $('deployModel');
  const deploymentMessage = $('deploymentMessage');

  const PIPELINE = window.SignARPipeline;
  const DB_NAME = 'SignAR-FSL-Private-Training-v2';
  const DB_VERSION = 1;
  const IMPORT_HAND_SIDE_VERSION = 1;
  const LEGACY_POSE_FEATURE_VALUES = 132;
  const LEGACY_HAND_FEATURE_VALUES = 126;
  const LEGACY_FEATURE_DIMENSION = 456;
  const SAMPLE_MS = 1000 / PIPELINE.config.sequence.samplingFps;
  const MOTION_THRESHOLD = 0.015;
  const AUTO_CAPTURE_PAUSE_MS = 1500;
  const MIN_FRAMES = 5;
  const MIN_IMPORTED_FRAMES = 1;
  const MAX_IMPORTED_FRAMES = 90;
  const FEATURE_DIMENSION = PIPELINE.config.features.dimension;

  let dbPromise = null;
  const tracker = window.SignARTracker.create({onWarning:(message,error)=>console.warn(message,error)});
  let stream = null;
  let cameraStartGeneration = 0;
  let cameraStarting = false;
  let trackerReady = false;
  let facingMode = 'environment';
  let rafId = null;
  let processBusy = false;
  let processIdlePromise = Promise.resolve();
  let lastFrameTime = -1;
  let lastProcessAt = 0;
  const trackingState = PIPELINE.createTrackingState();
  let processEveryMs = trackingState.processIntervalMs;
  let processCanvas = document.createElement('canvas');
  let processContext = processCanvas.getContext('2d');
  let previousFeatures = null;
  let previousHandsVisible = false;
  let capture = null;
  let mediaRecorder = null;
  let mediaChunks = [];
  let captureMeta = null;
  let captureFrames = [];
  let lastSampleAt = 0;
  let captureStartedAt = 0;
  let lastMotionAt = 0;
  let saving = false;
  let vocabulary = [];
  let savedSampleCount = 0;
  let savedLabelCount = 0;
  let savedSampleSignature = null;
  let sampleRecords = [];
  const selectedSampleIds = new Set();
  let filteredSampleIds = [];
  let training = false;
  let trainingCompleted = false;
  let candidateDeployable = false;
  let trainingHistoryAvailable = false;
  let deploying = false;
  let localTrainerAvailable = false;
  let localTrainerChecked = false;
  let importSession = null;
  let importing = false;
  let importBatchRunning = false;
  let deletingSamples = false;
  let downloadingSamples = false;
  let analyzingSamples = false;
  let applyingSampleCleanup = false;
  let sampleCleanupPlan = null;
  let importScheduled = false;
  let importedMediaUrl = null;
  let trackerResultsSeen = 0;
  let trackerFeaturesSeen = 0;
  let trackerFramesSent = 0;
  let trackerCurrentCounts = {left:0,right:0,selected:0};
  let motionTypeManuallySet = false;

  const safeText = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  function inferredUnitType(label) {
    const normalized = String(label || '').trim().toUpperCase();
    if (normalized === 'NO SIGN') return 'background';
    return /^[A-Z]$/.test(normalized) ? 'fingerspelling' : 'word';
  }
  function sampleLabelMetadata(label, source = {}) {
    const normalized = String(label || '').trim().toUpperCase();
    return {
      unitType: normalized === 'NO SIGN' ? 'background' : (source.unitType || inferredUnitType(label)),
      motionType: normalized === 'NO SIGN' ? 'static' : (source.motionType || (/^[JZ]$/.test(normalized) ? 'dynamic' : 'static')),
      displayText: String(source.displayText || '').trim()
    };
  }
  function syncLabelMetadataFields(useLabelDefaults = false, preserveMotion = false) {
    const label = labelSelect.value.trim();
    const background = label.toUpperCase() === 'NO SIGN';
    const previousUnitType = unitTypeSelect.value;
    unitTypeSelect.disabled = background || capture || importing || importBatchRunning;
    motionTypeSelect.disabled = background || capture || importing || importBatchRunning;
    displayTextInput.disabled = background || capture || importing || importBatchRunning;
    if (background) {
      unitTypeSelect.value = 'background';
      displayTextInput.value = '';
    } else {
      if (useLabelDefaults || previousUnitType === 'background' || !previousUnitType) {
        unitTypeSelect.value = inferredUnitType(label);
        if (!preserveMotion && !motionTypeManuallySet)
          motionTypeSelect.value = /^[JZ]$/.test(label.toUpperCase()) ? 'dynamic' : 'static';
      }
    }
  }
  function currentLabelMetadata(label = labelSelect.value) {
    return sampleLabelMetadata(label, {
      unitType: unitTypeSelect.value,
      motionType: motionTypeSelect.value,
      displayText: displayTextInput.value
    });
  }
  const measureDistance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);
  function adaptTrackingResolution(elapsed) {PIPELINE.adaptTrackingState(trackingState,elapsed);processEveryMs=trackingState.processIntervalMs;}

  function updateLabelOptions(samples = []) {
    const labels = new Map([['NO SIGN', 'NO SIGN']]);
    [...vocabulary,...samples.map(sample => sample.label)].forEach(value => {
      const label = String(value ?? '').trim();
      const key = label.toUpperCase();
      if (label && !labels.has(key)) labels.set(key,label);
    });
    $('labelOptions').innerHTML = [...labels.values()]
      .map(label => '<option value="' + safeText(label) + '">')
      .join('');
  }

  function updateSignerOptions(samples = []) {
    const aliases = [...new Set(samples.map(sample => String(sample.signerId ?? '').trim()).filter(Boolean))]
      .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
    signerOptions.innerHTML = aliases
      .map(alias => '<option value="' + safeText(alias) + '">')
      .join('');
  }

  function updateReviewerOptions(samples = []) {
    const reviewers = [...new Set(samples.map(sample => String(sample.verificationNotes ?? '').trim()).filter(Boolean))]
      .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
    reviewerOptions.innerHTML = reviewers
      .map(reviewer => '<option value="' + safeText(reviewer) + '">')
      .join('');
  }

  function notify(message) {
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(notify.timer);
    notify.timer = setTimeout(() => toast.classList.remove('show'), 3200);
  }

  function isSetupReady() {
    return !!signerInput.value.trim() && !!labelSelect.value.trim() &&
      notesInput.value.trim().length >= 5 && verifyCheck.checked && consentCheck.checked;
  }

  function importLabelForFile(file, fallbackLabel = labelSelect.value.trim()) {
    const signer = signerInput.value.trim();
    const extension = file.name.match(/\.[^.]+$/);
    let stem = extension ? file.name.slice(0, -extension[0].length) : file.name;
    const idSuffix = /_(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|sample-\d+-[a-z0-9]+|\d+)$/i;
    stem = stem.replace(idSuffix, '');
    const letterName = stem.match(/^([a-z])$/i);
    if (letterName) return letterName[1].toUpperCase();

    const knownSigners = [signer, ...Array.from(signerOptions.options || []).map(option => option.value)]
      .filter(Boolean)
      .sort((a, b) => b.length - a.length);
    const matchingSignerPrefix = knownSigners
      .map(alias => stem.match(new RegExp('^' + alias.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '_', 'i')))
      .find(Boolean);
    const firstSeparator = stem.indexOf('_');
    const possibleSigner = firstSeparator < 0 ? '' : stem.slice(0, firstSeparator);
    const hasGenericSignerPrefix = !matchingSignerPrefix && /^signer[-_].+|[-_]\d+$/i.test(possibleSigner);
    const label = matchingSignerPrefix
      ? stem.slice(matchingSignerPrefix[0].length)
      : hasGenericSignerPrefix
        ? stem.slice(firstSeparator + 1)
        : stem;
    return label.replace(/_/g, ' ').trim().slice(0, 40) || fallbackLabel;
  }

  function isImportSetupReady(files) {
    return !!signerInput.value.trim() && files.every(file => !!importLabelForFile(file)) &&
      notesInput.value.trim().length >= 5 && verifyCheck.checked && consentCheck.checked;
  }

  function updateBulkSelectionControls() {
    const selectedFiltered = filteredSampleIds.filter(id => selectedSampleIds.has(id)).length;
    const hiddenSelected = selectedSampleIds.size - selectedFiltered;
    const blocked = importing || importBatchRunning || training || deploying || capture || saving || deletingSamples || downloadingSamples || analyzingSamples || applyingSampleCleanup;
    const downloadableCount = sampleRecords.filter(sample => selectedSampleIds.has(String(sample.id)) && sampleSourceBlob(sample)).length;
    selectFilteredSamples.checked = filteredSampleIds.length > 0 && selectedFiltered === filteredSampleIds.length;
    selectFilteredSamples.indeterminate = selectedFiltered > 0 && selectedFiltered < filteredSampleIds.length;
    selectFilteredSamples.disabled = !filteredSampleIds.length || blocked;
    selectedSampleCount.textContent = selectedSampleIds.size + ' selected' +
      (hiddenSelected ? ' (' + hiddenSelected + ' hidden by filters)' : '');
    downloadSelectedBtn.disabled = !downloadableCount || blocked;
    deleteSelectedBtn.disabled = !selectedSampleIds.size || blocked;
  }

  function updateWorkflow() {
    const setupReady = isSetupReady();
    const hasSamples = savedSampleCount > 0;

    if (!capture && !saving && !importing && !training && !hasSamples && !captureHelp.textContent.startsWith('Sample too short')) {
      captureHelp.textContent = !setupReady
        ? 'Complete the sign details, reference, verification, and consent before recording.'
        : stream && !trackerReady
          ? 'Camera preview is on; hand tracking is starting. Recording will be available shortly.'
          : !stream
          ? 'Your sign details are ready. Start the camera to record, or import media from the panel above.'
          : autoCheck.checked
            ? 'Hands-free capture is ready. Make sure the signer is ready; movement starts recording and a pause saves the sample.'
            : 'Your camera and sign details are ready. Press Start recording, then Stop and save when the sign is complete.';
    }

    if (importMediaInput.files?.length && !importing && !importBatchRunning &&
        (importProgress.textContent.startsWith('File selected') || importProgress.textContent.startsWith('Media selected') || importProgress.textContent.startsWith('Ready to analyze'))) {
        const selectedFiles = Array.from(importMediaInput.files);
        importProgress.textContent = training
          ? 'Selected media; it will import automatically when training finishes.'
          : isImportSetupReady(selectedFiles)
            ? 'Ready to analyze ' + selectedFiles.length + ' selected file' + (selectedFiles.length === 1 ? '' : 's') + ' locally; import starts automatically.'
            : 'Media selected. Complete the signer, reference, verification, and consent; each label will be derived from its filename.';
    }

    startRecordBtn.disabled = !stream || !trackerReady || capture || saving || importing || importBatchRunning || training || deploying || deletingSamples || analyzingSamples || applyingSampleCleanup;
    importMediaInput.disabled = capture || saving || importing || importBatchRunning || training || deploying || deletingSamples || analyzingSamples || applyingSampleCleanup;
    exportJsonBtn.disabled = !hasSamples || importing || importBatchRunning || deletingSamples || analyzingSamples || applyingSampleCleanup;
    deleteAllBtn.disabled = !hasSamples || importing || importBatchRunning || training || deploying || capture || saving || deletingSamples || analyzingSamples || applyingSampleCleanup;
    trainModelBtn.disabled = savedLabelCount < 1 || training || deploying || capture || saving || importing || importBatchRunning || deletingSamples || analyzingSamples || applyingSampleCleanup || !localTrainerChecked || !localTrainerAvailable;
    trainModelBtn.textContent = training ? 'Training…' : trainingCompleted || trainingHistoryAvailable ? 'Train Again' : 'Start Training';
    deployModelBtn.disabled = !localTrainerAvailable || !candidateDeployable || training || deploying || capture || saving || importing || importBatchRunning || deletingSamples || analyzingSamples || applyingSampleCleanup;
    sampleList.querySelectorAll('[data-delete], [data-save]').forEach(button => {
      button.disabled = importing || importBatchRunning || training || deploying || capture || saving || deletingSamples || downloadingSamples || analyzingSamples || applyingSampleCleanup;
    });
    sampleList.querySelectorAll('[data-sample-select]').forEach(input => {
      input.disabled = importing || importBatchRunning || training || deploying || capture || saving || deletingSamples || downloadingSamples || analyzingSamples || applyingSampleCleanup;
    });
    analyzeSamplesBtn.disabled = !sampleRecords.length || analyzingSamples || applyingSampleCleanup || importing || importBatchRunning || training || deploying || capture || saving || deletingSamples || downloadingSamples;
    applySampleCleanupBtn.disabled = !sampleCleanupPlan || analyzingSamples || applyingSampleCleanup || importing || importBatchRunning || training || deploying || capture || saving || deletingSamples || downloadingSamples;
    updateBulkSelectionControls();
    startCameraBtn.disabled = importing || importBatchRunning || training || deploying || deletingSamples || analyzingSamples || applyingSampleCleanup || cameraStarting || !!stream;
    switchCameraBtn.disabled = importing || importBatchRunning || training || deploying || deletingSamples || analyzingSamples || applyingSampleCleanup || cameraStarting || capture || saving;
    switchCameraBtn.textContent = facingMode === 'environment' ? 'Use front camera' : 'Use back camera';
    stopCameraBtn.disabled = importing || importBatchRunning || training || deploying || deletingSamples || analyzingSamples || applyingSampleCleanup || !stream;

    if (importMediaInput.files?.length && isImportSetupReady(Array.from(importMediaInput.files)) && !capture && !saving && !importing && !importBatchRunning && !training && !importScheduled) {
      importScheduled = true;
      setTimeout(() => {
        importScheduled = false;
        if (importMediaInput.files?.length && isImportSetupReady(Array.from(importMediaInput.files)) && !capture && !saving && !importing && !importBatchRunning && !training) {
         void importSelectedMedia();
        }
      }, 0);
    }
  }

  function openDb() {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve,reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains('samples')) database.createObjectStore('samples',{keyPath:'id'});
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('IndexedDB is unavailable'));
    });
    return dbPromise;
  }

  async function dbRequest(mode, operation) {
    const database = await openDb();
    return new Promise((resolve,reject) => {
      const transaction = database.transaction('samples',mode);
      const store = transaction.objectStore('samples');
      const request = operation(store);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error || new Error('Local database operation failed'));
    });
  }
  const getSamples = async () => (await dbRequest('readonly',store => store.getAll())).sort((a,b) => b.createdAt-a.createdAt);
  const saveSample = sample => dbRequest('readwrite',store => store.put(sample));
  const deleteSample = id => dbRequest('readwrite',store => store.delete(id));
  const clearSamples = () => dbRequest('readwrite',store => store.clear());
  function swapImportedFeatureHandSlots(frame) {
    const rightHandStart=LEGACY_POSE_FEATURE_VALUES+LEGACY_HAND_FEATURE_VALUES;
    return [
      ...frame.slice(0,LEGACY_POSE_FEATURE_VALUES),
      ...frame.slice(rightHandStart,rightHandStart+LEGACY_HAND_FEATURE_VALUES),
      ...frame.slice(LEGACY_POSE_FEATURE_VALUES,rightHandStart),
      ...frame.slice(rightHandStart+LEGACY_HAND_FEATURE_VALUES)
    ];
  }
  function migrateImportedSampleRecordHandSlots(sample) {
    if(!['imported-video','imported-image'].includes(sample.sourceType)||sample.importHandSideVersion===IMPORT_HAND_SIDE_VERSION)return null;
    if(!Array.isArray(sample.features)||!sample.features.length||sample.features.some(frame=>!Array.isArray(frame)||frame.length!==LEGACY_FEATURE_DIMENSION||frame.some(value=>!Number.isFinite(value)))){
      throw new Error('Sample '+String(sample.id)+' has invalid landmark features; no imported samples were changed.');
    }
    return{
      ...sample,
      features:sample.features.map(swapImportedFeatureHandSlots),
      importHandSideVersion:IMPORT_HAND_SIDE_VERSION
    };
  }
  function isValidFeatureFrame(frame) {
    return Array.isArray(frame) && frame.length === FEATURE_DIMENSION && frame.every(value =>
      Number.isFinite(value) && value >= -3.00001 && value <= 3.00001);
  }
  function analyzeSampleFrames(sample,canonicalizeFrame) {
    const sourceFrames=Array.isArray(sample?.features)?sample.features:[];
    const isNoSign=String(sample?.label||'').trim().toUpperCase()==='NO SIGN';
    const validFrames=[];
    let malformedFrameCount=0,noHandFrameCount=0;
    for(const frame of sourceFrames){
      try{
        const normalized=canonicalizeFrame(frame);
        if(!isValidFeatureFrame(normalized)){malformedFrameCount++;continue;}
        if(!isNoSign&&!PIPELINE.hasAnyFeatureRegion(normalized))noHandFrameCount++;
        validFrames.push(normalized);
      }catch(error){malformedFrameCount++;}
    }
    return {
      validFrames,
      invalidFrameCount:sourceFrames.length-validFrames.length,
      malformedFrameCount,
      noHandFrameCount,
      reason:!validFrames.length
        ? sourceFrames.length?(noHandFrameCount?'No usable hand, face, or body frames':'No stored frames passed feature validation'):'No stored feature frames'
        : !isNoSign&&!validFrames.some(PIPELINE.hasAnyFeatureRegion)
          ? 'No usable hand, face, or body frames'
          : ''
    };
  }
  async function migrateLegacyImportedHandSlots() {
    const samples=await getSamples();
    const updates=samples.map(migrateImportedSampleRecordHandSlots).filter(Boolean);
    if(!updates.length)return 0;
    const database=await openDb();
    return new Promise((resolve,reject)=>{
      let transaction;
      try{
        transaction=database.transaction('samples','readwrite');
        transaction.oncomplete=()=>resolve(updates.length);
        transaction.onerror=()=>reject(transaction.error||new Error('Updating existing imported hand landmarks failed.'));
        transaction.onabort=()=>reject(transaction.error||new Error('Updating existing imported hand landmarks was aborted.'));
        const store=transaction.objectStore('samples');
        updates.forEach(sample=>store.put(sample));
      }catch(error){reject(error);}
    });
  }
  async function deleteSamples(ids) {
    if (!ids.length) return;
    const database = await openDb();
    return new Promise((resolve,reject) => {
      let transaction;
      try {
        transaction = database.transaction('samples','readwrite');
        const store = transaction.objectStore('samples');
        ids.forEach(id => store.delete(id));
      } catch (error) {
        reject(error);
        return;
      }
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error || new Error('Could not delete selected samples.'));
      transaction.onabort = () => reject(transaction.error || new Error('Deleting selected samples was aborted.'));
    });
  }

  function updateRecordState(active) {
    recordState.classList.toggle('recording',active);
    recordState.children[1].textContent = active ? (captureMeta?.auto ? 'Automatic capture running' : 'Recording sample…') : 'Not recording';
    frameCounter.textContent = captureFrames.length + ' frames';
    stopRecordBtn.disabled = !active;
    signerInput.disabled = active;
    labelSelect.disabled = active;
    verifyCheck.disabled = active;
    consentCheck.disabled = active;
    notesInput.disabled = active;
    syncLabelMetadataFields();
    updateWorkflow();
  }

  function validateCaptureForm() {
    if (!stream) { notify('Start the camera first.'); return false; }
    if (!trackerReady) { notify('Wait for hand tracking to finish starting.'); return false; }
    if (!signerInput.value.trim()) { notify('Enter a signer alias such as signer-01.'); return false; }
    if (!labelSelect.value.trim()) { notify('Choose a sign label or NO SIGN.'); return false; }
    if (!verifyCheck.checked) {
      notify(labelSelect.value.trim().toUpperCase() === 'NO SIGN'
        ? 'Have a qualified reviewer confirm that no intended sign is being performed.'
        : 'Verify the sign form with an FSL reference and qualified signer first.');
      return false;
    }
    if (!consentCheck.checked) { notify('Get informed permission before recording a signer.'); return false; }
    if (notesInput.value.trim().length < 5) { notify('Add a short FSL reference/checker note before recording.'); return false; }
    if (!window.MediaRecorder) { notify('This browser does not support MediaRecorder.'); return false; }
    return true;
  }

  function mediaRecorderType() {
    if (!window.MediaRecorder?.isTypeSupported) return null;
    const candidates=['video/webm;codecs=vp8','video/mp4;codecs=avc1.42E01E','video/webm;codecs=vp9'];
    return candidates.find(t => MediaRecorder.isTypeSupported(t)&&video.canPlayType(t)) || null;
  }

  function startCapture(isAuto = false) {
    if (capture || saving || !validateCaptureForm()) return;
    const mimeType = mediaRecorderType();
    if (!mimeType) {
      notify('This browser does not report a camera-recording format it can play. Import a WebM (VP8) or MP4 (H.264) video instead.');
      return;
    }
    try {
      mediaChunks = [];
      captureFrames = [];
      lastSampleAt = 0;
      captureStartedAt = performance.now();
      lastMotionAt = captureStartedAt;
      captureMeta = {
        id: (crypto.randomUUID?.() || ('sample-' + Date.now() + '-' + Math.random().toString(36).slice(2,9))),
        signerId: signerInput.value.trim().replace(/[^a-zA-Z0-9_-]/g,'_').slice(0,40),
        label: labelSelect.value,
        ...currentLabelMetadata(labelSelect.value),
        verificationNotes: notesInput.value.trim(),
        verified: true,
        consentRecorded: true,
        auto: isAuto,
        createdAt: Date.now()
      };
      mediaRecorder = new MediaRecorder(stream, {mimeType});
      mediaRecorder.ondataavailable = event => { if (event.data?.size) mediaChunks.push(event.data); };
      mediaRecorder.onerror = event => notify('Recorder error: ' + (event.error?.message || 'unknown error'));
      mediaRecorder.onstop = finishCapture;
      mediaRecorder.start(250);
      capture = true;
      if(frameDiagnostics) frameDiagnostics.textContent = 'Recording is active. Capturing coordinated hand, face, and body landmarks; the count below reports valid feature frames / analyzed frames.';
      frameCounter.textContent = '0 valid / waiting for results';
      stageLabel.textContent = isAuto ? 'AUTO RECORDING · ' + captureMeta.label : 'RECORDING · ' + captureMeta.label;
      const noSignSample = captureMeta.label.trim().toUpperCase() === 'NO SIGN';
      captureHelp.textContent = isAuto
        ? noSignSample ? 'Recording a neutral background example. Relax or move naturally, then pause to save.' : 'Motion detected. Relax and pause to save the sample automatically.'
        : noSignSample ? 'Record neutral posture, non-sign movement, and varied resting positions. Frames without detected cues are included.' : 'Perform the complete verified sign naturally, including its hand, face, and body movement. Press Stop and save when it is complete.';
      updateRecordState(true);
    } catch (error) {
      mediaRecorder = null; capture = null;
      notify('Unable to start recording: ' + (error.message || error.name));
    }
  }

  function stopCapture() {
    if (!capture || !mediaRecorder || mediaRecorder.state === 'inactive') return;
    capture = false;
    saving = true;
    updateRecordState(false);
    stageLabel.textContent = 'Saving sample…';
    try { mediaRecorder.stop(); }
    catch (error) { saving = false; notify('Could not stop recorder: ' + error.message); updateRecordState(false); }
  }

  async function finishCapture() {
    const meta = captureMeta;
    const frames = captureFrames.slice();
    const type = mediaRecorder?.mimeType || mediaChunks[0]?.type || 'video/webm';
    const videoBlob = new Blob(mediaChunks,{type});
    mediaRecorder = null; captureMeta = null; mediaChunks = []; captureFrames = []; capture = false;
    updateRecordState(false);
    if (!meta) { saving = false; return; }
    if (frames.length < MIN_FRAMES || videoBlob.size === 0) {
      saving = false;
      stageLabel.textContent = 'Sample not saved';
      captureHelp.textContent = 'Sample too short. Record at least ' + MIN_FRAMES + ' landmark frames and try again.';
      notify('Sample not saved: too short or no raw video was captured.');
      updateRecordState(false);
      return;
    }
    if (String(meta.label).trim().toUpperCase() !== 'NO SIGN' && !frames.some(PIPELINE.hasAnyFeatureRegion)) {
      saving = false;
      stageLabel.textContent = 'Sample not saved';
      captureHelp.textContent = 'No usable hand, face, or body cues were captured. Check framing and record the complete sign again.';
      notify('Sample not saved: the tracker found no usable holistic sign cues.');
      updateRecordState(false);
      return;
    }
    const sample = {...meta,sourceType:'camera',features:frames,frameCount:frames.length,videoBlob,videoType:type,videoExtension:type.includes('mp4')?'mp4':'webm',durationMs:Math.round(performance.now()-captureStartedAt)};
    try {
      await saveSample(sample);
      saving = false;
      updateWorkflow();
      stageLabel.textContent = 'Saved · ' + sample.label;
      captureHelp.textContent = 'Saved locally. Change the label or signer alias before recording the next sample.';
      notify('Saved ' + sample.label + ' — ' + frames.length + ' landmark frames.');
      await refreshSamples();
    } catch (error) {
      stageLabel.textContent = 'Local storage error';
      notify('Could not save sample. Local browser storage may be full: ' + error.message);
    } finally {
      saving = false;
      if (!capture) updateRecordState(false);
    }
  }

  function getLandmarkCounts(results) {
    const left = results?.leftHandLandmarks || [];
    const right = results?.rightHandLandmarks || [];
    const selected = PIPELINE.chooseHand(results);
    return {
      left: left.filter(PIPELINE.isVisiblePoint).length,
      right: right.filter(PIPELINE.isVisiblePoint).length,
      selected: selected?.validPoints || 0,
    };
  }

  function updateFrameDiagnostics(results, feature) {
    trackerResultsSeen++;
    trackerCurrentCounts = getLandmarkCounts(results || {});
    if (feature && PIPELINE.hasAnyFeatureRegion(feature)) trackerFeaturesSeen++;
    if (!frameDiagnostics) return;
    const counts = trackerCurrentCounts;
    frameDiagnostics.dataset.state = feature && PIPELINE.hasAnyFeatureRegion(feature) ? 'ok' : 'warning';
    frameDiagnostics.textContent = `Holistic tracking: ${trackerResultsSeen} frames · ${trackerFeaturesSeen} frames with hand, face, or body cues · left ${counts.left}, right ${counts.right} hand points.`;
  }
  function paintLandmarks(r, sw, sh) {
    const activeTrackingState=importSession?.trackingState||trackingState;
    const rect=video.getBoundingClientRect();
    window.SignARLandmarkRenderer.draw(canvas,r,sw,sh,rect.width,rect.height,activeTrackingState);
  }

  async function processTrackingFrame(image, sourceWidth=video.videoWidth||PIPELINE.config.camera.width, sourceHeight=video.videoHeight||PIPELINE.config.camera.height, waitForHands=false, generation=null) {
    if (generation===null&&processBusy) await processIdlePromise;
    if (generation!==null&&generation!==cameraStartGeneration) return null;
    const results = await tracker.processFrame(image,{waitForHands});
    if (generation!==null&&generation!==cameraStartGeneration) return null;
    processLandmarkResults(results,sourceWidth,sourceHeight);
    return results;
  }

  function processLandmarkResults(results, sourceWidth=video.videoWidth||PIPELINE.config.camera.width, sourceHeight=video.videoHeight||PIPELINE.config.camera.height) {
    paintLandmarks(results, sourceWidth, sourceHeight);
    const feature = PIPELINE.extract(results, sourceWidth, sourceHeight);
    const hasFeatures = PIPELINE.hasAnyFeatureRegion(feature);
    updateFrameDiagnostics(results, feature);
    guide.classList.toggle('hidden', hasFeatures);
    const now = performance.now();

    if (importSession) {
      importSession.processedFrames++;
      if (PIPELINE.hasUsableHand(results.leftHandLandmarks)) importSession.leftHandFrames++;
      if (PIPELINE.hasUsableHand(results.rightHandLandmarks)) importSession.rightHandFrames++;
      importSession.frames.push(feature);
      if (importSession.processedFrames % 5 === 0) frameCounter.textContent = importSession.frames.length + ' holistic feature frames';
      trackStatus.textContent = hasFeatures ? 'Imported holistic signer cues' : 'No hand, face, or body cues in this frame';
      return;
    }

    trackStatus.textContent = hasFeatures ? 'Holistic landmarks usable' : 'No holistic cues in latest frame';
    if (!hasFeatures) {
      previousFeatures = null;
      previousHandsVisible = false;
      if (capture && now - lastSampleAt >= SAMPLE_MS) {
        captureFrames.push(feature);
        lastSampleAt = now;
      }
      if (capture) frameCounter.textContent = `${captureFrames.length} feature frames / ${trackerResultsSeen} analyzed`;
      return;
    }

    const handsVisible = hasFeatures;
    const handAppeared = handsVisible && !previousHandsVisible;
    const energy = PIPELINE.measureMotion(previousFeatures, feature) || 0;
    previousFeatures = feature;
    previousHandsVisible = handsVisible;
    if (energy >= MOTION_THRESHOLD || handAppeared) {
      lastMotionAt = now;
      if (autoCheck.checked && verifyCheck.checked && consentCheck.checked && labelSelect.value && signerInput.value.trim() && notesInput.value.trim().length >= 5 && !capture && !saving)
        startCapture(true);
    }
    if (capture && now - lastSampleAt >= SAMPLE_MS) {
      captureFrames.push(feature);
      lastSampleAt = now;
      frameCounter.textContent = `${captureFrames.length} feature frames / ${trackerResultsSeen} analyzed`;
    }
    if (autoCheck.checked && capture && now - lastMotionAt > AUTO_CAPTURE_PAUSE_MS && captureFrames.length >= MIN_FRAMES && now - captureStartedAt > 1000)
      stopCapture();
  }
  function importMediaKind(file) {
    if (!file) return null;
    const type=(file.type||'').toLowerCase(),name=file.name||'';
    if (type.startsWith('video/') || /\.(mp4|webm|mov|m4v|ogv)$/i.test(name)) return 'video';
    if (['image/jpeg','image/png','image/webp','image/bmp'].includes(type) || /\.(jpe?g|png|webp|bmp)$/i.test(name)) return 'image';
    return null;
  }

  function emptyImportFileError(file) {
    if (file?.size !== 0) return null;
    return 'The selected file "'+file.name+'" is empty. Check that its download or export completed, then select it again.';
  }

  function importedVideoFrameCount(duration) {
    const desiredFrames = Math.ceil(duration * PIPELINE.config.sequence.samplingFps);
    return Math.max(MIN_IMPORTED_FRAMES, Math.min(MAX_IMPORTED_FRAMES, desiredFrames));
  }

  function loadImportImage(url) {
    return new Promise((resolve,reject)=>{
      const image=new Image();
      image.onload=()=>resolve(image);
      image.onerror=()=>reject(new Error('This browser cannot decode the selected image. Use PNG, JPEG, WebP, or BMP.'));
      image.src=url;
    });
  }

  function validImportForm(file, label) {
    if (!file) { notify('Choose a video or still image first.'); return false; }
    const emptyFileError=emptyImportFileError(file);
    if(emptyFileError){importProgress.textContent=emptyFileError;notify(emptyFileError);return false;}
    if (!importMediaKind(file)) {
      notify('Choose a video (MP4/WebM) or still image (PNG/JPEG/WebP/BMP).'); return false;
    }
    if (!signerInput.value.trim()) { notify('Enter a signer alias such as signer-01.'); return false; }
    if (!label) { notify('Choose the FSL label performed in this file or use a supported filename.'); return false; }
    if (!verifyCheck.checked) { notify('Verify this exact sign form with a reliable FSL reference and qualified signer first.'); return false; }
    if (!consentCheck.checked) { notify('Confirm that the signer gave permission for local storage and analysis.'); return false; }
    if (notesInput.value.trim().length < 5) { notify('Add a reference/checker note of at least five characters.'); return false; }
    if (capture || saving || importing) { notify('Finish the current recording or import first.'); return false; }
    return true;
  }

  function waitForMetadata() {
    if (video.readyState >= 1) return Promise.resolve();
    return new Promise((resolve,reject) => {
      const timer=setTimeout(()=>{cleanup();reject(new Error('Timed out reading this video. Try WebM (VP8) or MP4 (H.264).'));},12000);
      const loaded=()=>{cleanup();resolve();};
      const failed=()=>{cleanup();reject(new Error(videoDecodeErrorMessage()));};
      function cleanup(){clearTimeout(timer);video.removeEventListener('loadedmetadata',loaded);video.removeEventListener('error',failed);}
      video.addEventListener('loadedmetadata',loaded,{once:true});video.addEventListener('error',failed,{once:true});
    });
  }

  function videoDecodeErrorMessage() {
    const error=video.error;
    const reasons={
      1:'video playback was aborted',
      2:'the video could not be read',
      3:'the video data could not be decoded',
      4:'the video format or codec is not supported by this browser'
    };
    const reason=error?.code&&reasons[error.code]?reasons[error.code]:'the browser could not decode the video';
    const details=error?.message?' ('+error.message+')':'';
    return 'Unable to import this video: '+reason+details+'. Try exporting it as WebM (VP8) or MP4 (H.264).';
  }

  function waitForVideoFrame() {
    const hasDecodedFrame=()=>video.readyState>=2&&video.videoWidth>0&&video.videoHeight>0;
    if(hasDecodedFrame())return Promise.resolve();
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{cleanup();reject(new Error('The browser could not decode a video frame. Try WebM (VP8) or MP4 (H.264).'));},8000);
      const ready=()=>{if(!hasDecodedFrame())return;cleanup();resolve();};
      const failed=()=>{cleanup();reject(new Error(videoDecodeErrorMessage()));};
      function cleanup(){clearTimeout(timer);video.removeEventListener('loadeddata',ready);video.removeEventListener('canplay',ready);video.removeEventListener('error',failed);}
      video.addEventListener('loadeddata',ready);video.addEventListener('canplay',ready);video.addEventListener('error',failed);
      ready();
    });
  }

  function readableVideoDuration() {
    if(Number.isFinite(video.duration)&&video.duration>0)return video.duration;
    if(video.seekable.length){
      const end=video.seekable.end(video.seekable.length-1);
      if(Number.isFinite(end)&&end>0)return end;
    }
    return 0;
  }

  function seekImportedVideo(time) {
    const target=Math.max(0,Math.min(time,Math.max(0,readableVideoDuration()-0.01)));
    if (Math.abs(video.currentTime-target)<0.002) return waitForVideoFrame();
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{cleanup();reject(new Error('The browser could not seek through this video. Try WebM (VP8) or MP4 (H.264).'));},8000);
      const done=()=>{cleanup();waitForVideoFrame().then(resolve,reject);};
      const failed=()=>{cleanup();reject(new Error('A video frame could not be decoded.'))};
      function cleanup(){clearTimeout(timer);video.removeEventListener('seeked',done);video.removeEventListener('error',failed);}
      video.addEventListener('seeked',done,{once:true});video.addEventListener('error',failed,{once:true});
      try{video.currentTime=target;}catch(error){cleanup();reject(error);}
    });
  }

  async function processImportedFrame(source,sourceWidth,sourceHeight,waitForHands) {
    const state=importSession?.trackingState||trackingState;
    const frameStarted=performance.now(),{width,height}=PIPELINE.trackingFrameSize(sourceWidth,sourceHeight,state);
    if(processCanvas.width!==width)processCanvas.width=width;if(processCanvas.height!==height)processCanvas.height=height;
    processContext.drawImage(source,0,0,width,height);
    await processTrackingFrame(processCanvas,sourceWidth,sourceHeight,waitForHands);
    if(importSession){
      const currentDimensionIndex=state.dimensionIndex;
      PIPELINE.adaptTrackingState(state,performance.now()-frameStarted);
      state.dimensionIndex=Math.min(state.dimensionIndex,currentDimensionIndex);
    }else{
      adaptTrackingResolution(performance.now()-frameStarted);
    }
  }

  async function importUnknownDurationVideo(sourceWidth,sourceHeight) {
    let sampledFrames=0,lastSampleTime=video.currentTime,lastPlaybackTime=video.currentTime;
    let lastProgressAt=performance.now();
    const sampleFrame=async waitForHands=>{
      if(sampledFrames>=MAX_IMPORTED_FRAMES)return;
      await processImportedFrame(video,sourceWidth,sourceHeight,waitForHands);
      sampledFrames++;
      if(sampledFrames%5===0){
        importProgress.textContent='FSL sign label: '+importSession.label+' · analyzing video; sampled '+sampledFrames+' frames. Valid landmark frames: '+importSession.frames.length+'.';
        status.textContent='Import in progress';
      }
      if(sampledFrames===MAX_IMPORTED_FRAMES){
        importProgress.textContent='FSL sign label: '+importSession.label+' · reached the '+MAX_IMPORTED_FRAMES+'-frame analysis limit; continuing to read the video.';
      }
    };
    await sampleFrame(true);
    try {
      await video.play();
    } catch(error) {
      throw new Error('The browser could not play this video to read its duration: '+(error.message||error.name||'playback was blocked'));
    }
    try {
      while(!video.ended){
        if(!importing)throw new Error('Import cancelled.');
        await new Promise(resolve=>setTimeout(resolve,16));
        if(video.error)throw new Error(videoDecodeErrorMessage());
        const playbackTime=video.currentTime;
        if(playbackTime>lastPlaybackTime+0.001){lastPlaybackTime=playbackTime;lastProgressAt=performance.now();}
        else if(video.paused&&!video.ended)throw new Error('Video playback stopped before reaching the end.');
        else if(performance.now()-lastProgressAt>10000)throw new Error('Video playback stalled before reaching the end.');
        if(video.ended)break;
        if(playbackTime-lastSampleTime<SAMPLE_MS/1000)continue;
        await sampleFrame(false);
        lastSampleTime=playbackTime;
      }
      if(video.error)throw new Error(videoDecodeErrorMessage());
      if(video.currentTime-lastSampleTime>0.001&&sampledFrames<MAX_IMPORTED_FRAMES)await sampleFrame(false);
      return readableVideoDuration()||video.currentTime||0;
    } finally {
      video.pause();
    }
  }

  async function importMediaSample(file, fallbackLabel) {
    const label = importLabelForFile(file, fallbackLabel);
    labelSelect.value = label;
    syncLabelMetadataFields(true, true);
    if(!validImportForm(file, label))return false;
    const kind=importMediaKind(file);
    if(currentLabelMetadata(label).motionType==='dynamic'&&kind==='image'){
      importProgress.textContent='Dynamic signs need a video showing the full movement. Select a video or mark this sign as static.';
      notify(importProgress.textContent);
      return false;
    }
    importing=true;
    [signerInput,labelSelect,verifyCheck,consentCheck,notesInput,importMediaInput].forEach(input=>input.disabled=true);
    syncLabelMetadataFields();
    updateWorkflow();
    startCameraBtn.disabled=true;stopCameraBtn.disabled=true;startRecordBtn.disabled=true;stopRecordBtn.disabled=true;
    importProgress.textContent='Importing '+file.name+' · FSL sign label: '+label+'. Preparing local '+kind+'…';status.textContent='Importing '+kind+'…';
    try {
      if (stream) await stopCamera();
      while (processBusy) await new Promise(resolve=>setTimeout(resolve,25));
      await tracker.initialize();
      stage.classList.remove('mirror');
      video.pause();video.srcObject=null;video.removeAttribute('src');video.removeAttribute('poster');video.controls=kind==='video';video.muted=true;video.loop=false;
      importedMediaUrl=URL.createObjectURL(file);
      let image=null,duration=0;
      if(kind==='image'){
        video.poster=importedMediaUrl;video.classList.add('active');
        image=await loadImportImage(importedMediaUrl);
        if(!image.naturalWidth||!image.naturalHeight)throw new Error('The selected image has no readable dimensions.');
      }else{
        video.preload='metadata';video.src=importedMediaUrl;video.classList.add('active');video.load();
        await waitForMetadata();
        if(video.error)throw new Error(videoDecodeErrorMessage());
        await waitForVideoFrame();video.pause();
        duration=readableVideoDuration();
        if(duration)await seekImportedVideo(0);
      }
      const sourceWidth=kind==='image'?image.naturalWidth:(video.videoWidth||PIPELINE.config.camera.width);
      const sourceHeight=kind==='image'?image.naturalHeight:(video.videoHeight||PIPELINE.config.camera.height);
      const steps=kind==='image'?1:duration?importedVideoFrameCount(duration):0;
      const mediaType=(file.type||'').toLowerCase();
      const namedExtension=(file.name.match(/\.([a-z0-9]+)$/i)||[])[1]?.toLowerCase();
      const imageExtensions={'image/jpeg':'jpg','image/png':'png','image/webp':'webp','image/bmp':'bmp'};
      const videoExtensions={'video/mp4':'mp4','video/quicktime':'mov','video/webm':'webm','video/ogg':'ogv'};
      const videoMimeByExtension={mp4:'video/mp4',m4v:'video/mp4',mov:'video/quicktime',webm:'video/webm',ogv:'video/ogg'};
      const extension=kind==='image'
        ? (/\.(jpe?g|png|webp|bmp)$/i.test(file.name)?namedExtension:imageExtensions[mediaType]||'png')
        : (/\.(mp4|webm|mov|m4v|ogv)$/i.test(file.name)?namedExtension:videoExtensions[mediaType.split(';')[0]]||namedExtension||'webm');
      const mime=kind==='image'
        ? (imageExtensions[mediaType]?mediaType:'image/'+(extension==='jpg'?'jpeg':extension))
        : (mediaType.startsWith('video/')?mediaType:videoMimeByExtension[extension]||'video/webm');
      const importState=PIPELINE.createTrackingState();
      importState.dimensionIndex=Math.min(importState.dimensionIndex,trackingState.dimensionIndex);
      const session={label,trackingState:importState,frames:[],processedFrames:0,leftHandFrames:0,rightHandFrames:0};importSession=session;
      stageLabel.textContent='IMPORTING '+kind.toUpperCase()+' · '+label;
      trackStatus.textContent='Analyzing '+(kind==='image'?'still image':'recorded video');
      frameCounter.textContent='0 valid frames';
      guide.classList.remove('hidden');
      if(kind==='video'&&!duration){
        duration=await importUnknownDurationVideo(sourceWidth,sourceHeight);
      }else{
        for(let index=0;index<steps;index++){
          if(!importing)throw new Error('Import cancelled.');
          if(kind==='video'){
            const time=Math.min(index*duration/steps,Math.max(0,duration-0.01));
            await seekImportedVideo(time);
          }
          await processImportedFrame(kind==='image'?image:video,sourceWidth,sourceHeight,steps===1);
          const completed=index+1,percent=Math.min(100,Math.round(completed/steps*100));
          if(completed%5===0||completed===steps){importProgress.textContent='FSL sign label: '+label+' · analyzing frame '+completed+' of '+steps+' ('+percent+'%). Valid landmark frames: '+session.frames.length+'.';status.textContent='Import '+percent+'%';}
          await new Promise(resolve=>setTimeout(resolve,0));
        }
      }
      const frames=session.frames;
      importSession=null;
      frameCounter.textContent=frames.length+' valid frames';
      trackStatus.textContent=frames.some(PIPELINE.hasAnyFeatureRegion)?'Imported holistic signer cues':'No sign cues detected in this media';
      if(frames.length<MIN_IMPORTED_FRAMES){
        throw new Error('Found '+frames.length+' usable holistic feature frames; at least '+MIN_IMPORTED_FRAMES+' are needed. MediaPipe analyzed '+session.processedFrames+' frames; left-hand results appeared in '+session.leftHandFrames+' and right-hand results in '+session.rightHandFrames+'. Check lighting, framing, and the MediaPipe CDN if no hand, face, or body cues were found.');
      }
      const sampleLabel=label;
      if(!sampleLabel)throw new Error('The FSL sign label is empty; this sample was not saved.');
      if(sampleLabel.toUpperCase()!=='NO SIGN'&&!frames.some(PIPELINE.hasAnyFeatureRegion))
        throw new Error('No usable hand, face, or body cues were tracked in this sign. Check the framing and import a clip that shows the complete sign.');
      importProgress.textContent='Saving '+file.name+' with FSL sign label: '+sampleLabel+'…';
      const sample={
        id:(crypto.randomUUID?.()||('sample-'+Date.now()+'-'+Math.random().toString(36).slice(2,9))),
        signerId:signerInput.value.trim().replace(/[^a-zA-Z0-9_-]/g,'_').slice(0,40),
        label:sampleLabel,...currentLabelMetadata(sampleLabel),verificationNotes:notesInput.value.trim(),verified:true,consentRecorded:true,
        createdAt:Date.now(),sourceType:'imported-'+kind,importHandSideVersion:IMPORT_HAND_SIDE_VERSION,features:frames,frameCount:frames.length,
        sourceBlob:file.slice(0,file.size,mime),sourceMimeType:mime,sourceExtension:extension,durationMs:Math.round(duration*1000)
      };
      await saveSample(sample);
      if(!importBatchRunning)await refreshSamples();
      stageLabel.textContent='Imported '+kind+' · '+sample.label;
      importProgress.textContent='Saved '+file.name+' as '+sample.label+' ('+frames.length+' usable landmark frames and source file).';
      if(!importBatchRunning)notify('Imported '+sample.label+' and saved the source file and '+frames.length+' landmark frames locally.');
      return true;
    } catch(error) {
      console.error(error);
      const fileDetails=file?' ('+file.name+', '+(file.type||'unknown MIME type')+', '+file.size+' bytes)':'';
      importProgress.textContent='Import failed'+fileDetails+': '+(error.message||error.name||'Unknown error');
      notify('Media import failed: '+(error.message||error.name||'Unknown error'));
      return false;
    } finally {
      importSession=null;importing=false;
      video.pause();video.removeAttribute('src');video.removeAttribute('poster');video.controls=false;video.load();video.classList.remove('active');
      stage.classList.remove('mirror');
      if(importedMediaUrl){URL.revokeObjectURL(importedMediaUrl);importedMediaUrl=null;}
      ctx.clearRect(0,0,canvas.width,canvas.height);guide.classList.remove('hidden');stageLabel.textContent='Camera is off';
      [signerInput,labelSelect,verifyCheck,consentCheck,notesInput,importMediaInput].forEach(input=>input.disabled=importBatchRunning);
      syncLabelMetadataFields();
      syncLabelMetadataFields();
      startCameraBtn.disabled=importBatchRunning;stopCameraBtn.disabled=true;
      status.textContent='Local database ready';trackStatus.textContent='Not started';
      updateWorkflow();
    }
  }

  async function importSelectedMedia() {
    const files = Array.from(importMediaInput.files || []);
    if (!files.length) return;
    const unsupported = files.filter(file => !importMediaKind(file));
    if (unsupported.length) {
      const names = unsupported.map(file => file.name).join(', ');
      importProgress.textContent = 'Unsupported file type' + (unsupported.length === 1 ? '' : 's') + ': ' + names + '. Choose video (MP4/WebM) or still image (PNG/JPEG/WebP/BMP).';
      notify('Remove unsupported files from the selection before importing.');
      return;
    }
    if (!isImportSetupReady(files)) {
      notify('Complete the signer, reference, verification, and consent before importing.');
      return;
    }
    const fallbackLabel = labelSelect.value.trim();
    importBatchRunning = true;
    const failures = [];
    let importedCount = 0;
    let refreshError = null;
    importBatchStatus.hidden = false;
    importBatchStatus.textContent = 'Saved 0 of ' + files.length + ' files.';
    try {
      for (let index = 0; index < files.length; index++) {
        const file = files[index];
        importProgress.textContent = 'Importing file ' + (index + 1) + ' of ' + files.length + ': ' + file.name;
        importBatchStatus.textContent = 'Saved ' + importedCount + ' of ' + files.length + ' files · processing file ' + (index + 1) + ' of ' + files.length + '.';
        if (await importMediaSample(file, fallbackLabel)) importedCount++;
        else failures.push(file.name + ': ' + importProgress.textContent);
        importBatchStatus.textContent = 'Saved ' + importedCount + ' of ' + files.length + ' files.';
      }
    } finally {
      importBatchRunning = false;
      importMediaInput.value = '';
      [signerInput,labelSelect,verifyCheck,consentCheck,notesInput,importMediaInput].forEach(input=>input.disabled=false);
      if (importedCount) {
        try {
          await refreshSamples();
        } catch (error) {
          refreshError = error;
          console.error('Could not refresh the saved sample list after batch import.', error);
          status.textContent = 'Local storage error';
        }
      }
      importProgress.textContent = 'Batch finished: ' + importedCount + ' of ' + files.length + ' file' + (files.length === 1 ? '' : 's') + ' saved.' +
        (failures.length ? '\nFailed files: ' + failures.join(' | ') : '') +
        (refreshError ? '\nSamples were saved, but the list could not be refreshed: ' + (refreshError.message || refreshError.name || 'Unknown error') : '');
      importBatchStatus.textContent = 'Saved ' + importedCount + ' of ' + files.length + ' files' +
        (failures.length ? ' · ' + failures.length + ' failed' : '') + '.';
      updateWorkflow();
    }
    notify(refreshError
      ? 'Batch saved, but the sample list could not be refreshed: ' + (refreshError.message || refreshError.name || 'Unknown error')
      : 'Batch import finished: ' + importedCount + ' of ' + files.length + ' files saved.');
  }

  async function startCamera() {
    if(importing){notify('Wait for the imported video to finish.');return;}
    if(cameraStarting||stream)return;
    if(!window.isSecureContext){notify('Use HTTPS or http://localhost to access the camera.');return;}
    if(!navigator.mediaDevices?.getUserMedia){notify('This browser does not provide camera access.');return;}
    cameraStarting=true;trackerReady=false;const generation=++cameraStartGeneration;startCameraBtn.disabled=true;updateWorkflow();
    try {
      status.textContent='Opening camera...';
      const trackerInitialization=tracker.initialize().then(()=>null,error=>error);
      const openedStream=await tracker.openCamera(facingMode);
      if(generation!==cameraStartGeneration){openedStream.getTracks().forEach(track=>track.stop());return;}
      stream=openedStream;video.srcObject=stream;await video.play();if(generation!==cameraStartGeneration||!stream)return;video.classList.add('active');
      trackerResultsSeen=0;trackerFeaturesSeen=0;trackerFramesSent=0;trackerCurrentCounts={pose:0,left:0,right:0,face:0};
      if(frameDiagnostics){frameDiagnostics.dataset.state='warning';frameDiagnostics.textContent='Camera preview is on; preparing MediaPipe tracking.';}
      const actual=stream.getVideoTracks()[0]?.getSettings?.().facingMode;if(actual)facingMode=actual;
      stage.classList.toggle('mirror',facingMode==='user');
      startCameraBtn.disabled=true;stopCameraBtn.disabled=false;updateWorkflow();trackStatus.textContent='Starting hand tracking';stageLabel.textContent='Starting landmark tracking';guide.classList.remove('hidden');status.textContent='Camera preview on - starting tracking';
      const trackerError=await trackerInitialization;
      if(generation!==cameraStartGeneration||!stream)return;
      if(trackerError)throw trackerError;
      trackerReady=true;updateWorkflow();trackStatus.textContent='Camera ready';stageLabel.textContent='Frame the signing hand';status.textContent='Local database ready';
      rafId=requestAnimationFrame(time=>loop(time,generation));notify('Camera ready. Landmark capture will be stored locally.');
    }catch(error){
      if(generation!==cameraStartGeneration)return;
      console.error(error);if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;video.classList.remove('active');stage.classList.remove('mirror');trackerReady=false;startCameraBtn.disabled=false;stopCameraBtn.disabled=true;status.textContent='Camera unavailable';notify(error.message||error.name||'Unable to start camera');updateWorkflow();
    }finally{
      if(generation===cameraStartGeneration){cameraStarting=false;if(!stream)startCameraBtn.disabled=false;updateWorkflow();}
    }
  }

  async function switchCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    if (stream) {
      await stopCamera();
      await startCamera();
    } else {
      updateWorkflow();
    }
  }

  async function loop(time,generation) {
    if(!stream||generation!==cameraStartGeneration)return;
    if(!processBusy&&video.readyState>=2&&video.currentTime!==lastFrameTime&&time-lastProcessAt>=processEveryMs){
      processBusy=true;let releaseProcessIdle;processIdlePromise=new Promise(resolve=>{releaseProcessIdle=resolve});lastFrameTime=video.currentTime;lastProcessAt=time;const started=performance.now();
      try{const sw=video.videoWidth||PIPELINE.config.camera.width,sh=video.videoHeight||PIPELINE.config.camera.height,{width,height}=PIPELINE.trackingFrameSize(sw,sh,trackingState);if(processCanvas.width!==width)processCanvas.width=width;if(processCanvas.height!==height)processCanvas.height=height;processContext.drawImage(video,0,0,width,height);await processTrackingFrame(processCanvas,sw,sh,false,generation);if(generation===cameraStartGeneration){trackerFramesSent++;if(trackerFramesSent>=10&&trackerResultsSeen===0&&frameDiagnostics){frameDiagnostics.dataset.state='warning';frameDiagnostics.textContent=`Camera frames sent to MediaPipe: ${trackerFramesSent}, but no result callbacks have arrived. Check DevTools Console and internet access to the MediaPipe CDN, then refresh.`;}}}
      catch(error){if(generation===cameraStartGeneration){console.error(error);trackStatus.textContent='Tracking error';if(frameDiagnostics){frameDiagnostics.dataset.state='warning';frameDiagnostics.textContent='MediaPipe error: '+(error.message||error.name||'unknown')+'. Check internet/CDN and browser console.';}}}
      finally{const elapsed=performance.now()-started;if(generation===cameraStartGeneration)adaptTrackingResolution(elapsed);processBusy=false;releaseProcessIdle();}
    }
    if(generation===cameraStartGeneration&&stream)rafId=requestAnimationFrame(nextTime=>loop(nextTime,generation));
  }

  async function stopCamera() {
    cameraStartGeneration++;cameraStarting=false;trackerReady=false;
    if(capture)stopCapture();
    if(rafId)cancelAnimationFrame(rafId);rafId=null;
    if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;video.pause();video.srcObject=null;video.removeAttribute('src');video.controls=false;video.load();video.classList.remove('active');stage.classList.remove('mirror');
    ctx.clearRect(0,0,canvas.width,canvas.height);startCameraBtn.disabled=false;stopCameraBtn.disabled=true;trackStatus.textContent='Not started';stageLabel.textContent='Camera is off';guide.classList.remove('hidden');previousFeatures=null;previousHandsVisible=false;status.textContent='Local database ready';updateWorkflow();
  }

  function downloadBlob(blob,name) {
    const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),60000);
  }
  function sampleSourceBlob(sample) {
    return sample?.sourceBlob||sample?.videoBlob||null;
  }
  const zipCrcTable = Uint32Array.from({length:256},(_,value)=>{
    let crc=value;
    for(let bit=0;bit<8;bit++)crc=crc&1?0xedb88320^(crc>>>1):crc>>>1;
    return crc>>>0;
  });
  async function zipCrc32(blob,onProgress) {
    let crc=0xffffffff;
    const chunkSize=4*1024*1024;
    for(let offset=0;offset<blob.size;offset+=chunkSize){
      const end=Math.min(offset+chunkSize,blob.size),bytes=new Uint8Array(await blob.slice(offset,end).arrayBuffer());
      for(const byte of bytes)crc=zipCrcTable[(crc^byte)&0xff]^(crc>>>8);
      onProgress(end);
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    return (crc^0xffffffff)>>>0;
  }
  function zipHeader(size) {
    const bytes=new Uint8Array(size);
    return {bytes,view:new DataView(bytes.buffer)};
  }
  function zipDosDateTime(date) {
    const year=Math.max(1980,date.getFullYear());
    return {
      time:(date.getHours()<<11)|(date.getMinutes()<<5)|(date.getSeconds()>>1),
      date:((year-1980)<<9)|((date.getMonth()+1)<<5)|date.getDate()
    };
  }
  async function createZipBlob(files,onProgress=()=>{}) {
    const encoder=new TextEncoder(),localParts=[],centralParts=[];
    let offset=0;
    const totalSize=files.reduce((total,file)=>total+file.blob.size,0);
    const headerSize=22+files.reduce((total,file)=>total+76+encoder.encode(file.name).length,0);
    if(files.length>0xffff||totalSize+headerSize>0xffffffff)throw new Error('The selected files exceed the 4 GiB ZIP limit. Select fewer samples and download them in smaller batches.');
    const dos=zipDosDateTime(new Date());
    let processedSize=0;
    for(let fileIndex=0;fileIndex<files.length;fileIndex++){
      const file=files[fileIndex];
      const name=encoder.encode(file.name),dataBlob=file.blob;
      const crc=await zipCrc32(dataBlob,bytesProcessed=>{
        const completeSize=processedSize+bytesProcessed;
        downloadProgressBar.value=totalSize?Math.min(99,completeSize/totalSize*100):0;
        downloadProgressLabel.textContent='Preparing archive: file '+(fileIndex+1)+' of '+files.length+' · '+(completeSize/1048576).toFixed(1)+' of '+(totalSize/1048576).toFixed(1)+' MB';
        onProgress(completeSize,totalSize,fileIndex,files.length);
      });
      const local=zipHeader(30+name.length);
      local.view.setUint32(0,0x04034b50,true);local.view.setUint16(4,20,true);local.view.setUint16(6,0x0800,true);
      local.view.setUint16(8,0,true);local.view.setUint16(10,dos.time,true);local.view.setUint16(12,dos.date,true);
      local.view.setUint32(14,crc,true);local.view.setUint32(18,dataBlob.size,true);local.view.setUint32(22,dataBlob.size,true);
      local.view.setUint16(26,name.length,true);local.view.setUint16(28,0,true);local.bytes.set(name,30);
      localParts.push(local.bytes,dataBlob);
      const central=zipHeader(46+name.length);
      central.view.setUint32(0,0x02014b50,true);central.view.setUint16(4,20,true);central.view.setUint16(6,20,true);
      central.view.setUint16(8,0x0800,true);central.view.setUint16(10,0,true);central.view.setUint16(12,dos.time,true);
      central.view.setUint16(14,dos.date,true);central.view.setUint32(16,crc,true);
      central.view.setUint32(20,dataBlob.size,true);central.view.setUint32(24,dataBlob.size,true);
      central.view.setUint16(28,name.length,true);central.view.setUint16(30,0,true);central.view.setUint16(32,0,true);
      central.view.setUint16(34,0,true);central.view.setUint16(36,0,true);central.view.setUint32(38,0,true);
      central.view.setUint32(42,offset,true);central.bytes.set(name,46);centralParts.push(central.bytes);
      offset+=local.bytes.length+dataBlob.size;
      processedSize+=dataBlob.size;
    }
    const centralSize=centralParts.reduce((total,part)=>total+part.length,0),end=zipHeader(22);
    end.view.setUint32(0,0x06054b50,true);end.view.setUint16(4,0,true);end.view.setUint16(6,0,true);
    end.view.setUint16(8,files.length,true);end.view.setUint16(10,files.length,true);
    end.view.setUint32(12,centralSize,true);end.view.setUint32(16,offset,true);end.view.setUint16(20,0,true);
    return new Blob([...localParts,...centralParts,end.bytes],{type:'application/zip'});
  }
  function safeArchiveName(value) {
    return String(value||'sample').replace(/[\u0000-\u001f<>:"/\\|?*]/g,'_').replace(/[. ]+$/g,'').trim()||'sample';
  }
  function csvValue(value) {
    return '"'+String(value??'').replace(/"/g,'""')+'"';
  }
  async function downloadSelectedSamples() {
    if(downloadingSamples)return;
    const samples=sampleRecords.filter(sample=>selectedSampleIds.has(String(sample.id))&&sampleSourceBlob(sample));
    if(!samples.length){notify('Select at least one sample with a source file to download.');return;}
    downloadingSamples=true;
    downloadProgress.hidden=false;
    downloadProgressBar.value=0;
    downloadProgressLabel.textContent='Preparing '+samples.length+' selected samples…';
    updateWorkflow();
    try{
      const usedNames=new Set(['manifest.csv']),manifest=[['file_name','label','signer_id','sample_id','created_at']];
      const files=samples.map(sample=>{
        const extension=String(sample.sourceExtension||sample.videoExtension||(sample.sourceType==='imported-image'?'png':'webm')).replace(/^\.+/,'').replace(/[^a-z0-9]/gi,'')||'bin';
        const base=safeArchiveName(sample.signerId)+'_'+safeArchiveName(sample.label)+'_'+safeArchiveName(sample.id);
        let name=base+'.'+extension,index=2;
        while(usedNames.has(name.toLocaleLowerCase())){name=base+'_'+index+'.'+extension;index++;}
        usedNames.add(name.toLocaleLowerCase());
        manifest.push([name,sample.label,sample.signerId,sample.id,new Date(sample.createdAt||0).toISOString()]);
        return {name,blob:sampleSourceBlob(sample)};
      });
      const csv='\ufeff'+manifest.map(row=>row.map(csvValue).join(',')).join('\r\n')+'\r\n';
      files.push({name:'manifest.csv',blob:new Blob([csv],{type:'text/csv;charset=utf-8'})});
      await new Promise(resolve=>requestAnimationFrame(resolve));
      const archive=await createZipBlob(files);
      downloadBlob(archive,'signar-selected-samples-'+new Date().toISOString().slice(0,10)+'.zip');
      downloadProgressBar.value=100;
      downloadProgressLabel.textContent='Download ready: '+samples.length+' samples · '+(archive.size/1048576).toFixed(1)+' MB';
      notify(samples.length+' selected sample'+(samples.length===1?'':'s')+' downloaded.');
    }catch(error){
      console.error('Could not create selected sample archive:',error);
      downloadProgressLabel.textContent='Download failed: '+(error.message||error.name||'Unknown error');
      notify('Could not create the ZIP archive: '+(error.message||error.name||'Unknown error'));
    }finally{
      downloadingSamples=false;
      updateWorkflow();
    }
  }
  async function exportJson() {
    const samples=await getSamples();if(!samples.length){notify('No samples to export yet.');return;}
    const dataset=window.SignARBrowserTrainer.createDatasetDocument(samples,vocabulary);
    const blob=new Blob([JSON.stringify(dataset,null,2)],{type:'application/json'});downloadBlob(blob,'signar-fsl-dataset-'+new Date().toISOString().slice(0,10)+'.json');notify('Landmark JSON exported.');
  }
  function formatPercent(value) {
    return Number.isFinite(value) ? (value * 100).toFixed(1) + '%' : '—';
  }

  function trainingCoverageSummary(samples) {
    const labels = [...new Set(samples.map(sample => String(sample.label || '').trim().toUpperCase()).filter(Boolean))];
    const signers = [...new Set(samples.map(sample => String(sample.signerId || '').trim()).filter(Boolean))].sort();
    const liveCounts = Object.fromEntries(signers.map(signer => [
      signer,
      samples.filter(sample => String(sample.signerId || '').trim() === signer && sample.sourceType === 'camera').length
    ]));
    const labelSignerCounts = Object.fromEntries(labels.map(label => [
      label,
      new Set(samples.filter(sample => String(sample.label || '').trim().toUpperCase() === label).map(sample => String(sample.signerId || '').trim()).filter(Boolean)).size
    ]));
    const signLabels = labels.filter(label => label !== 'NO SIGN');
    const liveLabelCounts = Object.fromEntries(signLabels.map(label => [
      label,
      samples.filter(sample => String(sample.label || '').trim().toUpperCase() === label && sample.sourceType === 'camera').length
    ]));
    const noSignSamples = samples.filter(sample => String(sample.label || '').trim().toUpperCase() === 'NO SIGN').length;
    const noSignShare = samples.length ? noSignSamples / samples.length : 0;
    const missingCoverage = labels.filter(label => (labelSignerCounts[label] || 0) < 5);
    const missingLiveLabels = signLabels.filter(label => liveLabelCounts[label] === 0);
    const liveLabelMessage = missingLiveLabels.length
      ? ' Live-camera performance is unverified for ' + missingLiveLabels.slice(0, 8).join(', ') + (missingLiveLabels.length > 8 ? ', plus ' + (missingLiveLabels.length - 8) + ' more' : '') + '.'
      : ' Every sign label has live-camera examples in this dataset.';
    const unitCounts = samples.reduce((counts, sample) => {
      const type = sampleLabelMetadata(sample.label, sample).unitType;
      counts[type] = (counts[type] || 0) + 1;
      return counts;
    }, {});
    const unitSummary = Object.entries(unitCounts).map(([type, count]) => type + ': ' + count).join(', ') || 'none';
    return samples.length + ' samples | ' + labels.length + ' sign labels (' + unitSummary + ') | ' + signers.length + ' signers. ' +
      'Optional live-camera clips per signer: ' + (signers.map(signer => signer + ' ' + liveCounts[signer]).join(', ') || 'none') + '. ' +
      'Optional live-camera sign coverage: ' + (signLabels.length - missingLiveLabels.length) + '/' + signLabels.length + ' labels.' + liveLabelMessage + ' Camera samples do not affect training or transfer eligibility. ' +
      'NO SIGN: ' + noSignSamples + ' samples (' + (noSignShare * 100).toFixed(1) + '%; validation profile needs at least 10 and 1-25%). ' +
      (missingCoverage.length ? 'Collect every label from five signers; low coverage: ' + missingCoverage.slice(0, 5).join(', ') + '.' : 'Every label has five-signer coverage.');
  }
  function evaluationMetricRows(evaluation) {
    if (!Array.isArray(evaluation.labels) || !evaluation.classificationReport) return [];
    const rows = evaluation.labels.map(label => {
      const metrics = evaluation.classificationReport[label];
      if (!metrics || !Number.isFinite(metrics.precision) || !Number.isFinite(metrics.recall) ||
          !Number.isFinite(metrics['f1-score']) || !Number.isFinite(metrics.support)) return null;
      return {
        label,
        precision: metrics.precision,
        recall: metrics.recall,
        f1: metrics['f1-score'],
        support: metrics.support
      };
    });
    if (rows.some(row => !row)) return [];
    return rows.sort((a, b) => a.recall - b.recall || a.label.localeCompare(b.label));
  }

  function clearTrainingEvaluation() {
    trainingEvaluationDetails.hidden = true;
    trainingEvaluationDetails.open = false;
    trainingClassMetrics.innerHTML = '';
    trainingEvaluationNote.textContent = '';
  }

  function renderTrainingEvaluation(evaluation) {
    const rows = evaluationMetricRows(evaluation);
    const testSamples = Number(evaluation.sampleCounts?.test);
    trainingEvaluationDetails.hidden = false;
    trainingEvaluationDetails.open = false;
    if (!rows.length || !Number.isFinite(testSamples) || testSamples < 1) {
      trainingEvaluationDetails.querySelector('summary').textContent = 'Per-sign held-out results unavailable';
      trainingClassMetrics.innerHTML = '';
      trainingEvaluationNote.textContent = Number.isFinite(testSamples) && testSamples > 0
        ? 'Held-out test clips were reported, but complete per-sign metrics were missing from the trainer response.'
        : 'The trainer did not have held-out test clips, so no per-sign metrics were computed.';
      return;
    }
    trainingEvaluationDetails.querySelector('summary').textContent = 'Per-sign held-out results';
    trainingClassMetrics.innerHTML = rows.map(row =>
      '<tr><th scope="row">' + safeText(row.label) + '</th><td>' + formatPercent(row.precision) +
      '</td><td>' + formatPercent(row.recall) + '</td><td>' + formatPercent(row.f1) +
      '</td><td>' + safeText(String(Math.round(row.support))) + '</td></tr>'
    ).join('');
    trainingEvaluationNote.textContent = 'Sorted by lowest recall first. Metrics cover ' + testSamples +
      ' held-out clips; support is the number of test clips for each sign. These results are not live model scores.';
  }

  async function checkLocalTrainer() {
    let trainerStatus = null;
    try {
      const response = await fetch('/api/training/status', {cache:'no-store'});
      trainerStatus = await response.json();
      localTrainerAvailable = response.ok && trainerStatus.available === true;
      candidateDeployable = response.ok && trainerStatus.candidateDeployable === true;
      trainingHistoryAvailable = response.ok && trainerStatus.hasTrainingHistory === true;
    } catch (error) {
      localTrainerAvailable = false;
      candidateDeployable = false;
      trainingHistoryAvailable = false;
    }
    localTrainerChecked = true;
    if (!localTrainerAvailable) {
      trainingMessage.dataset.state = 'error';
      trainingMessage.textContent = 'Local trainer unavailable. Start start_training_studio.bat from the toolkit folder, then reopen this page.';
      deploymentMessage.dataset.state = 'error';
      deploymentMessage.textContent = 'Model transfer requires the local training studio.';
    } else {
      trainingMessage.dataset.state = '';
      trainingMessage.textContent = savedSampleCount
        ? trainingHistoryAvailable
          ? 'Local trainer ready. Add reviewed samples, then select Train Again to retrain and reevaluate the full eligible dataset.'
          : 'Local trainer ready. Add reviewed samples, then select Start Training to build and evaluate the first candidate.'
        : 'Save reviewed samples, then select Start Training. Every run trains and evaluates the complete eligible dataset.';
      deploymentMessage.dataset.state = candidateDeployable ? 'success' : '';
      deploymentMessage.textContent = trainerStatus.candidateMessage || 'Train and validate a candidate before transferring it to the interpreter.';
    }
    updateWorkflow();
  }

  async function trainModel() {
    if (training || importing || capture || saving || deletingSamples || !savedSampleCount || !localTrainerAvailable) return;
    if (savedLabelCount < 1) {
      trainingMessage.dataset.state = '';
      trainingMessage.textContent = 'Save at least one labeled sample before training.';
      return;
    }
    training = true;
    trainingCompleted = false;
    candidateDeployable = false;
    deploymentMessage.dataset.state = '';
    deploymentMessage.textContent = 'A new candidate is being trained; transfer will be enabled after validation.';
    trainingProgressWrap.dataset.state = 'running';
    trainingProgressWrap.setAttribute('aria-busy','true');
    trainingProgress.value = 0;
    trainingProgressLabel.textContent = 'Preparing and starting the local trainer';
    trainingEpochCount.textContent = 'Starting...';
    testAccuracy.textContent = '—';
    testSampleCount.textContent = '—';
    evaluationScopeMetric.textContent = '—';
    deploymentEligibleMetric.textContent = '—';
    clearTrainingEvaluation();
    trainingMessage.dataset.state = '';
    trainingMessage.textContent = 'Preparing a full model training run and signer-held-out evaluation using the eligible dataset. Python packages are checked and installed only when missing; keep this page open.';
    updateWorkflow();

    try {
      if (stream) await stopCamera();
      const samples = await getSamples();
      if (!samples.length) throw new Error('Save at least one labeled sample before training.');
      const dataset = window.SignARBrowserTrainer.createDatasetDocument(samples, vocabulary);
      trainingProgressLabel.textContent = 'train.ps1 is running';
      const trainingStartedAt=performance.now();
      const jobId=window.crypto?.randomUUID?.()||('job-'+Date.now()+'-'+Math.random().toString(36).slice(2));
      const elapsedText=()=>{
        const elapsed=Math.floor((performance.now()-trainingStartedAt)/1000);
        return 'Elapsed '+Math.floor(elapsed/60)+':'+String(elapsed%60).padStart(2,'0');
      };
      const readProgress=async()=>{
        const response=await fetch('/api/training/progress',{cache:'no-store'});
        const progress=await response.json();
        if(!response.ok)throw new Error(progress.error||'The local progress service returned an error.');
        return progress;
      };
      const showProgress=progress=>{
        const percent=Number(progress.percent);
        if(Number.isFinite(percent)){
          trainingProgress.value=Math.max(trainingProgress.value,Math.min(99,Math.max(0,percent)));
        }
        trainingProgressLabel.textContent=progress.stage||'Training is running';
        const epochText=progress.totalEpochs
          ? 'Epoch '+progress.epoch+' of '+progress.totalEpochs+' · '
          : '';
        const batchText=progress.batchesPerEpoch
          ? 'Batch '+progress.batch+' of '+progress.batchesPerEpoch+' · '
          : '';
        trainingEpochCount.textContent=Math.round(trainingProgress.value)+'% · '+epochText+batchText+elapsedText();
      };
      let startResponse;
      let completedResult=null;
      try{
        startResponse=await fetch('/api/training/run',{
          method:'POST',
          headers:{'Content-Type':'application/json','X-Training-Job-ID':jobId},
          body:JSON.stringify(dataset)
        });
      }catch(error){
        console.warn('Training start response was interrupted; checking the local job:',error);
      }
      if(startResponse){
        let startResult;
        try{startResult=await startResponse.json();}
        catch(error){
          if(!startResponse.ok)throw new Error('The local trainer returned an unreadable error response. Check the studio window before retrying.');
          console.warn('Training start response was unreadable; checking the local job:',error);
        }
        if(!startResponse.ok)throw new Error((startResult&&startResult.error)||'The local trainer could not start.');
        if(startResult?.completed===true)completedResult=startResult;
        else if(startResult&&startResult.started!==true)console.warn('The local trainer did not confirm the start response; checking the job ID.');
      }

      let unavailableCount=0;
      while(!completedResult){
        let progress;
        try{
          progress=await readProgress();
          unavailableCount=0;
        }catch(error){
          unavailableCount++;
          console.warn('Could not read local trainer progress:',error);
          if(unavailableCount>=30){
            throw new Error('The connection to the local training studio was lost. Training may still be running; check the studio window before retrying.');
          }
          trainingProgressLabel.textContent='Reconnecting to the local trainer...';
          trainingEpochCount.textContent=elapsedText();
          await new Promise(resolve=>setTimeout(resolve,1000));
          continue;
        }
        if(progress.jobId!==jobId){
          if(progress.active&&!progress.jobId){
            const error=new Error('The local studio is running a training job ('+(progress.stage||'progress unavailable')+') but is using an older server without job tracking. Training may still be running. Do not retry or restart until it finishes; then close and relaunch start_training_studio.bat.');
            error.name='ActiveLegacyTrainingError';
            throw error;
          }
          throw new Error('The local studio cannot find this training job. Restart start_training_studio.bat if the toolkit was updated, and check its window before retrying.');
        }
        if(progress.failed){
          const detail=progress.output?'\n\n'+progress.output:'';
          throw new Error((progress.error||progress.stage||'The local PowerShell trainer failed.')+detail);
        }
        if(progress.completed&&progress.result){
          completedResult=progress.result;
          break;
        }
        if(!progress.active){
          throw new Error('The local training job stopped without a result. Check the studio window before retrying.');
        }
        showProgress(progress);
        await new Promise(resolve=>setTimeout(resolve,1000));
      }
      const result=completedResult;

      const evaluation = result.evaluation || {};
      const trainingUpdate = evaluation.trainingUpdate || {};
      candidateDeployable = result.deploymentEligible === true;
      trainingHistoryAvailable = true;
      trainingCompleted = true;
      sampleSummary.textContent = savedSampleCount + ' sample' + (savedSampleCount === 1 ? '' : 's') +
        ' saved on this device. Select Train Again after adding reviewed samples to retrain and reevaluate the full eligible dataset.';
      trainingProgressWrap.removeAttribute('data-state');
      trainingProgressWrap.setAttribute('aria-busy','false');
      trainingProgress.value = 100;
      trainingProgressLabel.textContent = 'Training finished';
      trainingEpochCount.textContent = result.samples + ' samples sent to train.ps1';
      testAccuracy.textContent = formatPercent(evaluation.testAccuracy);
      testSampleCount.textContent = evaluation.sampleCounts?.test ?? '—';
      evaluationScopeMetric.textContent = evaluation.evaluationScope === 'signer-independent'
        ? 'Signer-independent'
        : evaluation.evaluationScope === 'sample-level' ? 'Sample-level'
        : evaluation.evaluationScope === 'not-evaluated' ? 'Not evaluated' : 'Unknown';
      deploymentEligibleMetric.textContent = result.deploymentEligible ? 'Yes' : 'No';
      renderTrainingEvaluation(evaluation);
      const recallFailures = Array.isArray(evaluation.classRecallFailures) ? evaluation.classRecallFailures : [];
      const recallNote = recallFailures.length
        ? 'Held-out recall is below 50% for: ' + recallFailures.join(', ') +
          '. Record more varied real examples for these labels and train again.'
        : '';
      deploymentMessage.dataset.state = candidateDeployable ? 'success' : '';
      deploymentMessage.textContent = candidateDeployable
        ? 'Automated checks passed. Review the sign labels and evaluation before transferring; the current manifest will be backed up.'
        : recallNote || result.deploymentMessage || 'This candidate did not pass deployment checks and cannot be transferred.';
      trainingMessage.dataset.state = 'success';
      const scopeNote = evaluation.evaluationScope === 'sample-level'
        ? 'This run used a sample-level split; performance on unseen signers has not been measured. '
        : evaluation.evaluationScope === 'not-evaluated'
          ? 'No held-out evaluation was possible; test accuracy was not measured. '
          : '';
      const updateNote = trainingUpdate.mode === 'incremental'
        ? 'Quick update trained ' + trainingUpdate.newSamples + ' new sample' + (trainingUpdate.newSamples === 1 ? '' : 's') +
          ' plus ' + trainingUpdate.replaySamples + ' replay sample' + (trainingUpdate.replaySamples === 1 ? '' : 's') + '. '
        : trainingUpdate.mode === 'unchanged'
          ? 'No new samples were found, so the existing candidate was kept. '
          : 'This run trained the full saved dataset. ' +
            (trainingUpdate.reason && trainingUpdate.reason !== 'Incremental updates were not requested.'
              ? trainingUpdate.reason + ' ' : '');
      trainingMessage.textContent = updateNote + 'Landmark JSON: ' + result.datasetExport + '. Candidate files: ' + result.candidateDirectory + '. ' + scopeNote +
        (result.deploymentEligible
          ? 'The candidate passed the configured data and metric checks; review the signs and evaluation before deployment.'
          : recallNote || 'This candidate is not deployment eligible. Review evaluation.json and improve the dataset before deployment.');
      notify('The local trainer finished and saved the landmark JSON.');
    } catch (error) {
      console.error(error);
      trainingCompleted = false;
      candidateDeployable = false;
      deploymentMessage.dataset.state = 'error';
      deploymentMessage.textContent = 'Transfer is disabled because training did not complete successfully.';
      trainingMessage.dataset.state = 'error';
      trainingMessage.textContent = 'Training could not finish: ' + (error.message || error.name || 'Unknown error');
      trainingProgressWrap.removeAttribute('data-state');
      trainingProgressWrap.setAttribute('aria-busy','false');
      const activeLegacyJob=error.name==='ActiveLegacyTrainingError';
      trainingProgressLabel.textContent=activeLegacyJob?'Studio update required':'Training failed';
      trainingEpochCount.textContent=activeLegacyJob?'Training may still be running':'Trainer stopped';
      if(!activeLegacyJob)trainingProgress.value=0;
      notify('Training failed: ' + (error.message || error.name || 'Unknown error'));
    } finally {
      training = false;
      updateWorkflow();
    }
  }
  async function deployCandidate() {
    if(deploying||training||deletingSamples||!localTrainerAvailable||!candidateDeployable)return;
    if(!confirm('Have you reviewed the model evaluation and sign labels? This transfers the candidate to SignLanguageAR/model and activates it in the interpreter. The current manifest will be backed up. Reload the interpreter page after transfer.'))return;
    deploying=true;
    deploymentMessage.dataset.state='';
    deploymentMessage.textContent='Validating and transferring the candidate model...';
    updateWorkflow();
    try{
      const response=await fetch('/api/model/deploy',{method:'POST'});
      let result;
      try{result=await response.json();}
      catch(error){throw new Error('The local studio returned an unreadable transfer response.');}
      if(!response.ok)throw new Error(result.error||'The model transfer failed.');
      if(result.deployed!==true)throw new Error('The local studio did not confirm model activation.');
      deploymentMessage.dataset.state='success';
      deploymentMessage.textContent='Activated '+result.modelPath+' in '+result.destination+'. Previous manifest backup: '+(result.backupDirectory||'none')+'. Reload the interpreter page to load the new model.';
      notify('Candidate model transferred. Reload the interpreter page to use it.');
    }catch(error){
      deploymentMessage.dataset.state='error';
      deploymentMessage.textContent='Model transfer failed: '+(error.message||error.name||'Unknown error');
    }finally{
      deploying=false;
      updateWorkflow();
    }
  }
  function normalizeSavedLabel(value) {
    return String(value||'').trim().toLocaleUpperCase();
  }

  function updateSampleFilterOptions(samples) {
    const updateOptions=(select,values,placeholder)=>{
      const selected=select.value;
      select.innerHTML='<option value="">'+placeholder+'</option>'+values.map(value=>'<option value="'+safeText(value)+'">'+safeText(value)+'</option>').join('');
      select.value=values.includes(selected)?selected:'';
    };
    const labels=[...new Map(samples.map(sample=>{
      const label=String(sample.label||'').trim();
      return [normalizeSavedLabel(label),label];
    }).filter(([key,label])=>key&&label)).values()]
      .sort((a,b)=>a.localeCompare(b,undefined,{sensitivity:'base',numeric:true}));
    const signers=[...new Set(samples.map(sample=>String(sample.signerId||'').trim()).filter(Boolean))]
      .sort((a,b)=>a.localeCompare(b,undefined,{sensitivity:'base',numeric:true}));
    updateOptions(sampleLabelFilter,labels,'All labels');
    updateOptions(sampleSignerFilter,signers,'All signers');
  }

  function renderSampleList(samples) {
    const query=sampleSearch.value.trim().toLocaleLowerCase();
    const labelFilter=sampleLabelFilter.value;
    const signerFilter=sampleSignerFilter.value;
    const sortMode=sampleSort.value;
    const matching=samples.filter(sample=>{
      const label=String(sample.label||''),signer=String(sample.signerId||'').trim();
      return (!labelFilter||normalizeSavedLabel(label)===normalizeSavedLabel(labelFilter))&&(!signerFilter||signer===signerFilter)&&
        (!query||(label+' '+signer).toLocaleLowerCase().includes(query));
    });
    const compareText=(a,b)=>String(a||'').localeCompare(String(b||''),undefined,{sensitivity:'base',numeric:true});
    matching.sort((a,b)=>{
      if(sortMode==='label')return compareText(a.label,b.label)||compareText(a.signerId,b.signerId)||Number(b.createdAt||0)-Number(a.createdAt||0);
      if(sortMode==='signer')return compareText(a.signerId,b.signerId)||compareText(a.label,b.label)||Number(b.createdAt||0)-Number(a.createdAt||0);
      if(sortMode==='oldest')return Number(a.createdAt||0)-Number(b.createdAt||0)||compareText(a.id,b.id);
      return Number(b.createdAt||0)-Number(a.createdAt||0)||compareText(a.id,b.id);
    });
    filteredSampleIds=matching.map(sample=>String(sample.id));
    sampleFilterSummary.textContent=matching.length===samples.length
      ? samples.length+' sample'+(samples.length===1?'':'s')
      : matching.length+' of '+samples.length+' samples';
    updateBulkSelectionControls();
    if(!matching.length){
      sampleList.innerHTML='<p class="empty">'+(samples.length?'No samples match these search and filter settings.':'No samples yet. Saved samples will appear here.')+'</p>';
      return;
    }
    sampleList.innerHTML=matching.map(sample=>{
      const source=sampleSourceBlob(sample),kind=sample.sourceType==='imported-image'?'image':'video';
      const fileDetails=source?Math.max(.01,source.size/1048576).toFixed(1)+' MB '+kind:'no source file';
      const sampleId=String(sample.id),label=String(sample.label||''),signer=String(sample.signerId||'');
      const metadata=sampleLabelMetadata(label,sample);
      return '<article class="sample"><div class="sample-heading-row"><label class="sample-select"><input type="checkbox" data-sample-select="'+safeText(sampleId)+'"'+(selectedSampleIds.has(sampleId)?' checked':'')+' aria-label="Select '+safeText(label)+' sample from '+safeText(signer)+'"></label>'+
        '<div class="sample-heading"><strong>'+safeText(label)+' <span aria-hidden="true">·</span> '+safeText(signer)+'</strong></div></div>'+
        '<small class="sample-meta">'+new Date(sample.createdAt).toLocaleString()+'<br>'+safeText(sample.frameCount)+' frames · '+fileDetails+' · '+(sample.verified?'verified form':'unverified')+'</small>'+
        '<small class="sample-meta">'+safeText(metadata.unitType)+' | '+safeText(metadata.motionType)+' motion'+(metadata.displayText?' | '+safeText(metadata.displayText):'')+'</small>'+
        '<details class="sample-management"><summary>Edit and manage</summary><div class="sample-management-content"><div class="sample-edit-fields">'+
        '<label>FSL label<input data-sample-label maxlength="40" required value="'+safeText(sample.label||'')+'"></label>'+
        '<label>Signer name / alias<input data-sample-signer maxlength="40" required value="'+safeText(sample.signerId||'')+'"></label>'+
        '<label>Sign unit<select data-sample-unit><option value="fingerspelling"'+(metadata.unitType==='fingerspelling'?' selected':'')+'>Fingerspelling</option><option value="word"'+(metadata.unitType==='word'?' selected':'')+'>Word</option><option value="phrase"'+(metadata.unitType==='phrase'?' selected':'')+'>Phrase</option><option value="background"'+(metadata.unitType==='background'?' selected':'')+'>Background</option></select></label>'+
        '<label>Motion<select data-sample-motion><option value="static"'+(metadata.motionType==='static'?' selected':'')+'>Static</option><option value="dynamic"'+(metadata.motionType==='dynamic'?' selected':'')+'>Dynamic</option></select></label>'+
        '<label>Display text<input data-sample-display maxlength="120" value="'+safeText(metadata.displayText)+'"></label>'+
        '</div><div class="sample-actions"><button type="button" data-save="'+safeText(sample.id)+'">Save changes</button><button type="button" data-download="'+safeText(sample.id)+'">Source file</button><button type="button" data-delete="'+safeText(sample.id)+'">Delete</button></div></div></details></article>';
    }).join('');
    sampleList.querySelectorAll('[data-sample-select]').forEach(input=>{
      input.disabled=importing||importBatchRunning||training||deploying||capture||saving||deletingSamples;
      input.addEventListener('change',()=>{
        if(input.checked)selectedSampleIds.add(input.dataset.sampleSelect);
        else selectedSampleIds.delete(input.dataset.sampleSelect);
        updateBulkSelectionControls();
      });
    });
    sampleList.querySelectorAll('[data-download]').forEach(button=>button.addEventListener('click',()=>{
      const sample=samples.find(item=>item.id===button.dataset.download),source=sampleSourceBlob(sample);
      if(source)downloadBlob(source,sample.signerId+'_'+sample.label+'_'+sample.id+'.'+(sample.sourceExtension||sample.videoExtension||'webm'));
    }));
    sampleList.querySelectorAll('[data-save]').forEach(button=>{
      button.disabled=importing||importBatchRunning||training||deploying||capture||saving||deletingSamples;
      button.addEventListener('click',async()=>{
        const sample=samples.find(item=>item.id===button.dataset.save),card=button.closest('.sample');
        if(!sample||!card)return;
        if(importing||importBatchRunning||training||capture||saving){notify('Finish the current capture, import, or training run before editing samples.');return;}
        const label=card.querySelector('[data-sample-label]').value.trim();
        const signerId=card.querySelector('[data-sample-signer]').value.trim();
        const unitType=card.querySelector('[data-sample-unit]').value;
        const motionType=card.querySelector('[data-sample-motion]').value;
        const displayText=card.querySelector('[data-sample-display]').value.trim();
        if(!label||!signerId){notify('Both the FSL label and signer name are required.');return;}
        if(label.length>40||signerId.length>40){notify('The label and signer name must each be 40 characters or fewer.');return;}
        if((label.toUpperCase()==='NO SIGN') !== (unitType==='background')){notify('NO SIGN uses the Background unit; other labels must be letters, words, or phrases.');return;}
        if(unitType==='background'&&motionType!=='static'){notify('Background samples must use Static motion.');return;}
        if(motionType==='dynamic'&&sample.sourceType==='imported-image'){notify('Dynamic signs need a video showing the complete movement.');return;}
        const labelMetadata={unitType,motionType,displayText};
        if(label===String(sample.label||'')&&signerId===String(sample.signerId||'')&&JSON.stringify(labelMetadata)===JSON.stringify(sampleLabelMetadata(sample.label,sample))){notify('No sample changes to save.');return;}
        button.disabled=true;
        try{
          await saveSample({...sample,label,signerId,...labelMetadata});
          await refreshSamples();
          notify('Updated sample label, sign type, motion, or display text.');
        }catch(error){
          button.disabled=false;
          notify('Could not save sample changes: '+(error.message||error.name||'Unknown error'));
        }
      });
    });
    sampleList.querySelectorAll('[data-delete]').forEach(button=>{
      button.disabled=importing||importBatchRunning||training||deploying||capture||saving||deletingSamples;
      button.addEventListener('click',async()=>{
        if(deletingSamples)return;
        if(!confirm('Delete this local sample and its source file?'))return;
        deletingSamples=true;
        updateWorkflow();
        try{
          await deleteSample(button.dataset.delete);
          selectedSampleIds.delete(button.dataset.delete);
          await refreshSamples();
          notify('Sample deleted from this browser.');
        }catch(error){notify('Could not delete sample: '+(error.message||error.name||'Unknown error'));}
        finally{deletingSamples=false;updateWorkflow();}
      });
    });
  }

  async function analyzeSavedSamples() {
    if(analyzingSamples||applyingSampleCleanup)return;
    analyzingSamples=true;
    sampleCleanupPlan=null;
    sampleHealthSummary.textContent='Analyzing saved feature frames locally…';
    sampleCleanupDetails.hidden=true;
    sampleCleanupList.innerHTML='';
    updateWorkflow();
    try{
      const samples=await getSamples(),validSamples=[],invalidSamples=[];
      let droppedFrameCount=0,droppedNoHandFrameCount=0;
      for(let index=0;index<samples.length;index++){
        const sample=samples[index],analysis=analyzeSampleFrames(sample,window.SignARBrowserTrainer.canonicalizeFrame);
        if(analysis.reason)invalidSamples.push({sample,reason:analysis.reason});
        else{
          validSamples.push({...sample,features:analysis.validFrames,frameCount:analysis.validFrames.length});
          droppedFrameCount+=analysis.invalidFrameCount;
          droppedNoHandFrameCount+=analysis.noHandFrameCount;
        }
        if((index+1)%40===0){
          sampleHealthSummary.textContent='Analyzed '+(index+1)+' of '+samples.length+' saved samples…';
          await new Promise(resolve=>setTimeout(resolve,0));
        }
      }
      sampleCleanupPlan={validSamples,invalidSamples,droppedFrameCount,droppedNoHandFrameCount};
      const noHandSamples=invalidSamples.filter(({reason})=>reason==='No usable hand, face, or body frames').length;
      sampleHealthSummary.textContent='Analysis complete: '+validSamples.length+' valid sample'+(validSamples.length===1?'':'s')+' to resave; '+invalidSamples.length+' sample'+(invalidSamples.length===1?'':'s')+' to delete ('+noHandSamples+' with no usable holistic cues); '+droppedFrameCount+' malformed frame'+(droppedFrameCount===1?'':'s')+' to remove from valid samples. Missing-cue frames remain in valid sequences to preserve timing. Source media and metadata are kept.';
      sampleCleanupDetails.hidden=!invalidSamples.length;
      sampleCleanupList.innerHTML=invalidSamples.map(({sample,reason})=>'<li><strong>'+safeText(sample.label||'Unlabeled')+' · '+safeText(sample.signerId||'Unknown signer')+'</strong><br><small>'+safeText(String(sample.id))+' · '+safeText(reason)+'</small></li>').join('');
    }catch(error){
      sampleCleanupPlan=null;
      sampleHealthSummary.textContent='Analysis failed: '+(error.message||error.name||'Unknown error');
      notify('Could not analyze saved samples: '+(error.message||error.name||'Unknown error'));
    }finally{
      analyzingSamples=false;
      updateWorkflow();
    }
  }

  async function persistSampleCleanup(plan) {
    const database=await openDb();
    return new Promise((resolve,reject)=>{
      let transaction;
      try{
        transaction=database.transaction('samples','readwrite');
        const store=transaction.objectStore('samples');
        plan.validSamples.forEach(sample=>store.put(sample));
        plan.invalidSamples.forEach(({sample})=>store.delete(sample.id));
      }catch(error){reject(error);return;}
      transaction.oncomplete=()=>resolve();
      transaction.onerror=()=>reject(transaction.error||new Error('Could not resave valid samples.'));
      transaction.onabort=()=>reject(transaction.error||new Error('Sample cleanup was aborted.'));
    });
  }

  async function applySampleCleanup() {
    const plan=sampleCleanupPlan;
    if(!plan||applyingSampleCleanup)return;
    const validCount=plan.validSamples.length,invalidCount=plan.invalidSamples.length;
    if(!confirm('Resave '+validCount+' valid sample'+(validCount===1?'':'s')+' with only validated frames and permanently delete '+invalidCount+' sample'+(invalidCount===1?'':'s')+' with no usable frames, including their source media?'))return;
    applyingSampleCleanup=true;
    updateWorkflow();
    try{
      await persistSampleCleanup(plan);
      plan.invalidSamples.forEach(({sample})=>selectedSampleIds.delete(String(sample.id)));
      sampleCleanupPlan=null;
      await refreshSamples();
      sampleHealthSummary.textContent='Cleanup complete: resaved '+validCount+' valid sample'+(validCount===1?'':'s')+' and deleted '+invalidCount+' invalid sample'+(invalidCount===1?'':'s')+'. '+plan.droppedFrameCount+' invalid frame'+(plan.droppedFrameCount===1?'':'s')+' removed; original media for valid samples was preserved.';
      sampleCleanupDetails.hidden=true;
      notify('Sample cleanup complete.');
    }catch(error){
      sampleHealthSummary.textContent='Cleanup failed; the database transaction was not completed: '+(error.message||error.name||'Unknown error');
      notify('Could not clean saved samples: '+(error.message||error.name||'Unknown error'));
    }finally{
      applyingSampleCleanup=false;
      updateWorkflow();
    }
  }

  async function refreshSamples() {
    if(!applyingSampleCleanup){
      sampleCleanupPlan=null;
      sampleCleanupDetails.hidden=true;
      sampleHealthSummary.textContent='Saved samples changed; analyze again before cleanup.';
    }
    const samples=await getSamples();
    trainingCoverage.textContent=trainingCoverageSummary(samples);
    const currentIds=new Set(samples.map(sample=>String(sample.id)));
    for(const id of selectedSampleIds)if(!currentIds.has(id))selectedSampleIds.delete(id);
    sampleRecords=samples;
    updateLabelOptions(samples);
    updateSignerOptions(samples);
    updateReviewerOptions(samples);
    updateSampleFilterOptions(samples);
    const sampleSignature=JSON.stringify(samples.map(sample=>({
      id:String(sample.id||''),label:String(sample.label||''),signerId:String(sample.signerId||''),unitType:String(sample.unitType||''),motionType:String(sample.motionType||''),displayText:String(sample.displayText||'')
    })).sort((a,b)=>a.id.localeCompare(b.id)));
    const labelCount=new Set(samples.map(sample=>String(sample.label||'').trim().toUpperCase()).filter(Boolean)).size;
    const samplesChanged=savedSampleSignature!==null&&sampleSignature!==savedSampleSignature;
    if(samplesChanged){
      trainingCompleted=false;
      candidateDeployable=false;
      deploymentMessage.dataset.state='';
      deploymentMessage.textContent='Saved samples changed; train a new candidate before transferring it.';
      trainingProgressWrap.removeAttribute('data-state');
      trainingProgressWrap.setAttribute('aria-busy','false');
      trainingMessage.dataset.state='';
      trainingMessage.textContent=samples.length&&labelCount<1
        ? 'Samples are saved locally. Add a sign label before training.'
        : samples.length
          ? localTrainerAvailable
            ? 'Your saved samples changed. Select Start Training when you are ready to include the latest data.'
            : 'Start the local studio with start_training_studio.bat to enable the PowerShell trainer.'
        : localTrainerAvailable
          ? 'Save at least one labeled sample, then select Start Training.'
          : 'Start the local studio with start_training_studio.bat to enable the PowerShell trainer.';
      trainingProgress.value=0;
      trainingProgressLabel.textContent=samples.length&&labelCount<1?'Waiting for a labeled sample':samples.length?'Ready to start training':'Waiting for you to start';
      trainingEpochCount.textContent='Trainer idle';
      testAccuracy.textContent='—';
      testSampleCount.textContent='—';
      evaluationScopeMetric.textContent='—';
      deploymentEligibleMetric.textContent='—';
      clearTrainingEvaluation();
    }
    savedSampleCount=samples.length;
    savedLabelCount=labelCount;
    savedSampleSignature=sampleSignature;
    sampleSummary.textContent=samples.length
      ? samples.length+' sample'+(samples.length===1?'':'s')+' saved on this device. '+
        (trainingHistoryAvailable
          ? 'Select Train Again after adding reviewed samples to retrain and reevaluate the full eligible dataset.'
          : 'Add reviewed samples before the first full training run.')
      : 'No samples yet. Each recording is saved only in this browser.';
    updateWorkflow();
    renderSampleList(samples);
  }

  async function initialize() {
    try {
      await openDb();
      let migratedSamples=0,migrationError=null;
      try{migratedSamples=await migrateLegacyImportedHandSlots();}
      catch(error){migrationError=error;console.error('Existing imported hand-slot migration failed:',error);}
      status.textContent=migrationError?'Local database ready · import update failed':'Local database ready';
      await refreshSamples();
      if(migratedSamples)notify('Updated left/right landmark features in '+migratedSamples+' existing imported sample'+(migratedSamples===1?'':'s')+'. Source files were unchanged.');
      if(migrationError)notify('Existing imported samples were not updated: '+migrationError.message);
      await checkLocalTrainer();
    }catch(error){status.textContent='Local storage error';notify('Could not open local sample storage: '+error.message);}
  }

  startCameraBtn.addEventListener('click',startCamera);
  switchCameraBtn.addEventListener('click',switchCamera);
  stopCameraBtn.addEventListener('click',stopCamera);
  startRecordBtn.addEventListener('click',()=>startCapture(false));
  stopRecordBtn.addEventListener('click',stopCapture);
  sampleSearch.addEventListener('input',()=>renderSampleList(sampleRecords));
  sampleLabelFilter.addEventListener('change',()=>renderSampleList(sampleRecords));
  sampleSignerFilter.addEventListener('change',()=>renderSampleList(sampleRecords));
  sampleSort.addEventListener('change',()=>renderSampleList(sampleRecords));
  selectFilteredSamples.addEventListener('change',()=>{
    filteredSampleIds.forEach(id=>{
      if(selectFilteredSamples.checked)selectedSampleIds.add(id);
      else selectedSampleIds.delete(id);
    });
    sampleList.querySelectorAll('[data-sample-select]').forEach(input=>{
      input.checked=selectedSampleIds.has(input.dataset.sampleSelect);
    });
    updateBulkSelectionControls();
  });
  deleteSelectedBtn.addEventListener('click',async()=>{
    if(deletingSamples)return;
    const records=sampleRecords.filter(sample=>selectedSampleIds.has(String(sample.id)));
    const ids=records.map(sample=>sample.id);
    if(!ids.length){
      selectedSampleIds.clear();
      renderSampleList(sampleRecords);
      notify('Select at least one saved sample to delete.');
      return;
    }
    const hiddenCount=ids.length-filteredSampleIds.filter(id=>selectedSampleIds.has(id)).length;
    const hiddenWarning=hiddenCount?' This includes '+hiddenCount+' sample'+(hiddenCount===1?' hidden':'s hidden')+' by the current filters.':'';
    if(!confirm('Permanently delete '+ids.length+' selected sample'+(ids.length===1?'':'s')+' and their source media from this browser?'+hiddenWarning))return;
    deletingSamples=true;
    updateWorkflow();
    try{
      await deleteSamples(ids);
      selectedSampleIds.clear();
      await refreshSamples();
      notify(ids.length+' selected sample'+(ids.length===1?'':'s')+' deleted from this browser.');
    }catch(error){
      notify('Could not delete selected samples: '+(error.message||error.name||'Unknown error'));
    }finally{
      deletingSamples=false;
      updateWorkflow();
    }
  });
  downloadSelectedBtn.addEventListener('click',downloadSelectedSamples);
  analyzeSamplesBtn.addEventListener('click',analyzeSavedSamples);
  applySampleCleanupBtn.addEventListener('click',applySampleCleanup);
  [signerInput,notesInput,displayTextInput].forEach(input=>input.addEventListener('input',updateWorkflow));
  labelSelect.addEventListener('input',()=>{syncLabelMetadataFields(true);updateWorkflow();});
  labelSelect.addEventListener('change',()=>{syncLabelMetadataFields();updateWorkflow();});
  unitTypeSelect.addEventListener('change',updateWorkflow);
  motionTypeSelect.addEventListener('change',()=>{motionTypeManuallySet=true;updateWorkflow();});
  [verifyCheck,consentCheck].forEach(input=>input.addEventListener('change',updateWorkflow));
  importMediaInput.addEventListener('change',()=>{
    const files=Array.from(importMediaInput.files || []);
    importBatchStatus.hidden=true;importBatchStatus.textContent='';
    const unsupported=files.filter(file=>!importMediaKind(file));
    if(unsupported.length) {
      importProgress.textContent='Unsupported file type: '+unsupported.map(file=>file.name).join(', ')+'. Choose a video (MP4/WebM) or still image (PNG/JPEG/WebP/BMP).';
      importMediaInput.value='';
      notify('Choose a video (MP4/WebM) or still image (PNG/JPEG/WebP/BMP).');
      updateWorkflow();
      return;
    }
    if(files.length)importProgress.textContent=isImportSetupReady(files)
      ? 'Ready to analyze '+files.length+' selected file'+(files.length===1?'':'s')+' locally; import starts automatically.'
      : 'Media selected. Complete the signer, reference, verification, and consent; each label will be derived from its filename.';
    else importProgress.textContent='Choose one or more videos or images. If sign details or consent are missing, import waits; copy the full message here if it fails.';
    updateWorkflow();
  });
  autoCheck.addEventListener('change',()=>{
    captureHelp.textContent=autoCheck.checked
      ? 'Hands-free mode starts when movement is detected and saves after a short pause. Make sure the signer is ready before enabling it.'
      : 'Manual mode: press Start recording, then Stop and save when the sign is complete.';
    updateWorkflow();
  });
  $('exportJson').addEventListener('click',exportJson);
  trainModelBtn.addEventListener('click',trainModel);
  deployModelBtn.addEventListener('click',deployCandidate);
  deleteAllBtn.addEventListener('click',async()=>{
    if(deletingSamples||!confirm('Permanently delete all SignAR source media and landmarks from this browser?'))return;
    deletingSamples=true;
    updateWorkflow();
    try{
      await stopCamera();
      await clearSamples();
      selectedSampleIds.clear();
      await refreshSamples();
      notify('All local samples deleted.');
    }catch(error){
      notify('Could not delete all local samples: '+(error.message||error.name||'Unknown error'));
    }finally{
      deletingSamples=false;
      updateWorkflow();
    }
  });
  window.addEventListener('beforeunload',()=>{if(stream)stream.getTracks().forEach(track=>track.stop());});
  updateRecordState(false);
  syncLabelMetadataFields(true);
  initialize();
})();
