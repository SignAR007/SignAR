/* Node.js tests for the shared collector/interpreter tracking and feature pipeline. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
global.window = {};
const pipelinePath=path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'core', 'fsl-pipeline.js');
vm.runInThisContext(fs.readFileSync(pipelinePath, 'utf8'));
const landmarkRendererPath=path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'ui', 'landmark-renderer.js');
vm.runInThisContext(fs.readFileSync(landmarkRendererPath, 'utf8'));
const extractor = global.window.SignARPipeline;
const decoderExtractor = extractor;
assert.equal(extractor.version, 'signar-fsl-456-v2');
assert.equal(extractor.dimension, 456);
assert.equal(decoderExtractor.version, extractor.version);
assert.equal(decoderExtractor.dimension, extractor.dimension);
assert.equal(extractor.config.recognition.minimumConfidencePercent, 85, 'interpreter should accept model scores from 85%');
assert.deepEqual(extractor.sequenceTensorShape(null), [null, 30, 456]);
assert.deepEqual(extractor.sequenceTensorShape(1), [1, 30, 456]);
assert.equal(extractor.sequenceGapToleranceMs(0), 200, 'unknown tracker speed should retain the nominal sequence-gap tolerance');
assert.equal(extractor.sequenceGapToleranceMs(30), 200, 'fast tracking should not relax stale-frame detection');
assert.equal(extractor.sequenceGapToleranceMs(10), 250, 'slower tracking should tolerate its measured frame cadence');
assert.equal(extractor.sequenceGapToleranceMs(5), 500, 'slow tracking tolerance should remain capped at half a second');
const interpolationStart=Array(456).fill(0),interpolationEnd=Array(456).fill(2);
interpolationStart[132]=-3;
interpolationEnd[132]=2;
assert.equal(extractor.interpolateFeatureFrame(interpolationStart,interpolationEnd,.25)[0],.5,'gap filling should interpolate numeric features');
assert.equal(extractor.interpolateFeatureFrame(interpolationStart,interpolationEnd,.25)[132],-3,'gap filling should preserve missing-landmark sentinels');
assert.throws(()=>extractor.interpolateFeatureFrame(interpolationStart,interpolationEnd,NaN),/interpolation weight/,'gap filling should reject invalid weights');
assert.equal(extractor.config.sequence.minImportedFrames, 1, 'very short imports should be analyzed at least once');
assert.deepEqual(extractor.config.camera, { width: 960, height: 540, frameRateIdeal: 30, frameRateMax: 60 }, 'both workflows should request the shared lower-bandwidth camera profile');
const initialTrackingState = extractor.createTrackingState();
assert.deepEqual(extractor.trackingFrameSize(1280, 720, initialTrackingState), {
  width: 720, height: 405, scale: 0.5625
}, 'start tracker inference at 720px and adapt resolution with measured performance');
assert.equal(initialTrackingState.dimensionIndex, 1, 'initialize landmark processing below the maximum resolution');
const slowTrackingState = extractor.createTrackingState();
for (let frame = 0; frame < extractor.config.tracking.adaptation.slowFrames; frame++) {
  extractor.adaptTrackingState(slowTrackingState, extractor.config.tracking.adaptation.slowFrameMs + 1);
}
assert.equal(slowTrackingState.dimensionIndex, 0, 'allow sustained slow tracking to fall to the supported 640px profile');
for (let frame = 0; frame < extractor.config.tracking.adaptation.fastFrames; frame++) {
  extractor.adaptTrackingState(initialTrackingState, extractor.config.tracking.adaptation.fastFrameMs - 1);
}
assert.equal(initialTrackingState.dimensionIndex, 2, 'scale back up only after sustained fast tracking');
const collectorSource = fs.readFileSync(path.join(__dirname, '..', 'src', 'collector', 'collect.js'), 'utf8');
const collectorHtml = fs.readFileSync(path.join(__dirname, 'collect.html'), 'utf8');
const collectorTheme = fs.readFileSync(path.join(__dirname, '..', 'src', 'ui', 'collect-theme.css'), 'utf8');
const collectorStyles = fs.readFileSync(path.join(__dirname, '..', 'src', 'ui', 'collect.css'), 'utf8');
const trackerSource = fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'tracker', 'mediapipe-tracker.js'), 'utf8');
const studioServerSource = fs.readFileSync(path.join(__dirname, 'studio_server.py'), 'utf8');
const trainerSource = fs.readFileSync(path.join(__dirname, 'train_model.py'), 'utf8');
const interpreterHtml = fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'index.html'), 'utf8');
const interpreterServiceWorker = fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'sw.js'), 'utf8');
const interpreterStyles = fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'ui', 'style.css'), 'utf8');
const predictionHistorySource = fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'interpreter', 'prediction-history.js'), 'utf8');
const landmarkRendererSource = fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'ui', 'landmark-renderer.js'), 'utf8');
const decoderSource = fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'src', 'interpreter', 'app.js'), 'utf8');
const modelMetadata = JSON.parse(fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'model', 'metadata.json'), 'utf8'));
const modelVocabulary = JSON.parse(fs.readFileSync(path.join(__dirname, '..', '..', 'SignLanguageAR', 'model', 'vocabulary.json'), 'utf8'));
function resolvedBrandAsset(html, pagePath, className) {
  const brandSource = html.match(new RegExp('<img class="' + className + '" src="([^"]+)"'))?.[1];
  const iconSource = html.match(/<link rel="icon" href="([^"]+)"/)?.[1];
  const appleIconSource = html.match(/<link rel="apple-touch-icon" href="([^"]+)"/)?.[1];
  assert.ok(brandSource && iconSource && appleIconSource, 'brand and app icon assets should be declared');
  const pageDirectory = path.dirname(pagePath);
  const brandPath = path.resolve(pageDirectory, brandSource);
  assert.equal(brandPath, path.resolve(pageDirectory, iconSource), 'header brand should use its app icon');
  assert.equal(brandPath, path.resolve(pageDirectory, appleIconSource), 'touch icon should match the header brand');
  return brandPath;
}
const projectRoot = path.join(__dirname, '..', '..');
assert.equal(
  resolvedBrandAsset(collectorHtml, path.join(__dirname, 'collect.html'), 'mark'),
  resolvedBrandAsset(interpreterHtml, path.join(projectRoot, 'SignLanguageAR', 'index.html'), 'brand-mark'),
  'collector and interpreter should share one SignAR brand asset'
);
assert.match(collectorTheme, /\.mark\s*\{[^}]*object-fit:\s*cover;/s, 'collector brand image should fill its logo frame');
assert.match(interpreterStyles, /\.brand-mark\s*\{[^}]*object-fit:\s*cover;/s, 'interpreter brand image should fill its logo frame');
assert.doesNotMatch(interpreterHtml, /intro-banner|intro-copy|intro-note|A TOOL FOR LEARNING, NOT TRANSLATION|See the movement\.|Explore the meaning\./, 'decoder should not include the removed introduction banner');
assert.match(interpreterHtml, /<span>Model score<\/span>/, 'decoder should describe its uncalibrated top-class output as a model score');
assert.match(predictionHistorySource, /best\.confidence < minimumConfidencePercent/, 'decoder should apply the shared model-score threshold');
assert.match(decoderSource, /async function loadModelMetadata/, 'decoder should load model metadata dynamically');
assert.match(decoderSource, /JSON\.stringify\(metadata\.labels\)!==JSON\.stringify\(modelConfig\.labels\)/, 'decoder should reject metadata label-order mismatches');
assert.deepEqual(modelMetadata.labels, modelVocabulary.labels, 'active metadata should preserve the exact interpreter output order');
assert.equal(modelMetadata.sequence_length, extractor.config.sequence.length, 'active metadata should match the shared temporal window');
assert.equal(modelMetadata.feature_count, extractor.config.features.dimension, 'active metadata should match the shared landmark dimension');
assert.match(trainerSource, /ModelCheckpoint\(/, 'training should checkpoint its best weights');
assert.match(trainerSource, /training-history\.json/, 'training should persist loss and accuracy curves');
assert.match(trainerSource, /metadata\.json/, 'training should export dynamic model metadata');
assert.match(decoderSource, /Below the '\+MIN_CONFIDENCE_PERCENT\+'% model-score threshold/, 'decoder should show its configured threshold');
assert.doesNotMatch(decoderSource, /99% model-score threshold|Waiting for 99% score/, 'decoder should not retain the old 99% threshold messaging');
assert.match(decoderSource,/window\.tf\.tidy\(\(\)=>\{[\s\S]*?model\.predict\(input\)/,'inference should scope input tensor allocations and prediction intermediates');
assert.match(decoderSource,/predictionHistory\.update\(probabilities,modelConfig\.labels,MIN_CONFIDENCE_PERCENT\)/,'decoder should use the shared prediction-history policy');
assert.match(decoderSource,/const NO_SIGN_LABEL='NO SIGN'/,'decoder should reserve a trained background class for rejection');
assert.match(decoderSource,/label\.trim\(\)\.toUpperCase\(\)===NO_SIGN_LABEL[\s\S]*?showNoSign\('No sign detected/,'a stable background-class prediction should display no sign instead of appending a label');
assert.match(decoderSource,/noSignNote[\s\S]*?idle or unfamiliar input can be labeled as a sign/,'decoder should warn when its loaded model has no background class');
assert.match(predictionHistorySource,/if \(history\.length > 5\) history\.shift\(\)/,'prediction output should use a bounded smoothing history');
assert.match(decoderSource,/segmentFrames\.length<SEQUENCE_LENGTH/,'live inference should wait for 30 captured frames instead of stretching a short buffer');
assert.match(decoderSource,/NO_PREDICTION_RESTART_MS=PREDICTION_HISTORY\.noPredictionRestartMs\(/,'an uncommitted static sign should have enough time to fill a window and complete verification');
assert.match(predictionHistorySource,/history\[history\.length - 1\]\.label !== best\.label[\s\S]*?history = \[\]/,'a class change should discard stale prediction votes');
assert.match(predictionHistorySource,/const stable = streak >= REQUIRED_STABLE_PREDICTIONS/,'a sign should require three consecutive qualifying model windows');
assert.match(decoderSource,/if\(!stable\)\{\s*detected\.textContent='VERIFYING SIGN'[\s\S]*?return;\s*\}\s*detected\.textContent=label/,'the displayed sign should update only after consecutive evidence stabilizes');
assert.match(decoderSource,/function addWord\(label,confidence\)\{if\(segmentCommitted\|\|/,'repeated stable predictions should not append a sign twice in one segment');
assert.match(decoderSource,/predictionStreak>=REQUIRED_STABLE_PREDICTIONS&&predictionStreak\/length>=\.6/,'sentence output should require a sustained stable sign before committing');
assert.match(decoderSource,/PREDICTION_HISTORY\.shouldCommitPrediction\(/,'a stable segment should use the tested single-commit policy');
assert.match(predictionHistorySource,/function shouldCommitPrediction\(/,'single-commit criteria should be isolated for regression coverage');
assert.match(decoderSource,/const hadPrediction=segmentCommitted;[\s\S]*?else if\(handsVisible\)\{\s*showNoSign\('A new sign sequence is being checked\.'\)/,'discarding an uncommitted sequence should clear its stale displayed label');
assert.match(decoderSource,/PIPELINE\.interpolateFeatureFrame/,'short tracking gaps should be interpolated using the shared missing-value rules');
assert.match(decoderSource,/track\.addEventListener\('ended'/,'an ended camera track should trigger graceful interruption handling');
for(const id of ['trackingFps','inferenceLatency','topKScores'])assert.match(interpreterHtml,new RegExp('id="'+id+'"'),'decoder should expose '+id+' diagnostics');
assert.match(decoderSource,/if\(!handsVisible\)\{resetRecognitionSegment\(\);signerVisible=false/,'decoder should discard a segment when no usable signing hand is tracked');
assert.match(decoderSource,/function maxSequenceGapMs\(\)\{return PIPELINE\.sequenceGapToleranceMs\(trackingFps\)\}/,'sequence gap tolerance should adapt to measured tracker cadence');
assert.match(decoderSource,/function adaptTrackingResolution\(elapsed\)\{PIPELINE\.adaptTrackingState\(trackingState,elapsed\);processEvery=trackingState\.processIntervalMs\}/,'slow tracking should use the supported minimum processing resolution');
assert.match(decoderSource,/lastTrackingResultAt>maxSequenceGapMs\(\)/,'a long tracking stall should invalidate the full recognition segment');
assert.match(decoderSource,/elapsed>maxSequenceGapMs\(\)/,'sequence sampling should discard stale frames but interpolate slower regular tracking');
assert.match(decoderSource,/gap<=maxSequenceGapMs\(\)/,'pre-roll interpolation should use the same adaptive gap tolerance');
assert.match(decoderSource,/showMultiPersonUnavailable\(\);return/,'prediction should remain paused when a fresh single-signer check is unavailable');
assert.match(decoderSource,/resetRecognitionSegment\(\);signerVisible=false;showNoSign\('Tracking failed/,'tracking errors should clear stale predictions and show an explicit error state');
assert.match(decoderSource,/function resetRecognitionSegment\(\)[\s\S]*?predictionHistory\.clear\(\)[\s\S]*?clearTopKScores\(\)/,'segment reset should clear both smoothing history and stale top-class diagnostics');
assert.match(interpreterHtml,/src\/interpreter\/prediction-history\.js/,'decoder should load its tested prediction-history policy before app initialization');
assert.match(interpreterServiceWorker,/src\/interpreter\/prediction-history\.js/,'offline interpreter cache should include the prediction-history policy');
assert.match(decoderSource,/PIPELINE\.hasUsableHand\(r\.leftHandLandmarks\)\|\|PIPELINE\.hasUsableHand\(r\.rightHandLandmarks\)/,'decoder should use the shared minimum hand-landmark quality check');
assert.match(decoderSource,/guide\?\.classList\.toggle\('is-hidden',hands>0\|\|multiplePeopleDetected\)/,'camera guide should remain visible until a usable hand is tracked');
assert.doesNotMatch(interpreterStyles, /\.intro-banner|\.intro-note/, 'decoder styles should not retain introduction banner rules');
assert.match(interpreterStyles,/\.landmark-overlay\s*\{[^}]*object-fit:\s*fill;/s,'interpreter overlay should not re-crop landmarks after the renderer maps camera cropping');
assert.doesNotMatch(collectorHtml, /COLLECTION FLOW|FROM SAMPLES TO INTERPRETER|workflowTitle|interpreterGuideTitle/, 'requested guidance panels should be removed');
assert.doesNotMatch(collectorSource, /workflowStatus|workflowStep[1-4]/, 'collector should not require removed workflow elements');
assert.doesNotMatch(collectorTheme, /workflow-step|workflow-status|interpreter-guide|interpreter-steps/, 'theme should not retain styles for removed panels');
assert.doesNotMatch(collectorStyles, /workflow-card|workflow-intro|workflow-status|workflow-steps|workflow-step|interpreter-guide|interpreter-steps/, 'base styles should not retain styles for removed panels');
assert.match(collectorTheme, /\.sample-list\s*\{[^}]*max-height:\s*min\(65vh,\s*36rem\)[^}]*overflow-y:\s*auto/s, 'saved samples should scroll inside a bounded review panel');
assert.match(collectorHtml, /id="sampleList"[^>]*tabindex="0"[^>]*role="region"[^>]*aria-label="Saved training samples"/, 'saved sample scrolling should be keyboard reachable and labeled');
for (const id of ['sampleSearch','sampleLabelFilter','sampleSignerFilter','sampleSort']) {
  assert.match(collectorHtml,new RegExp('id="'+id+'"'),'review panel should expose '+id);
}
for (const id of ['selectFilteredSamples','selectedSampleCount','downloadSelected','deleteSelected','analyzeSamples','applySampleCleanup','sampleHealthSummary']) {
  assert.match(collectorHtml,new RegExp('id="'+id+'"'),'review panel should expose '+id);
}
assert.match(collectorSource,/function renderSampleList\(samples\)[\s\S]*?samples\.filter\(/,'review panel should search and filter the saved sample list');
assert.match(collectorSource,/function normalizeSavedLabel\(value\)[\s\S]*?toLocaleUpperCase\(\)/,'saved sample label filters should normalize case and whitespace');
assert.match(collectorSource,/normalizeSavedLabel\(label\)===normalizeSavedLabel\(labelFilter\)/,'saved sample label filtering should compare normalized label values');
assert.match(collectorSource,/if\(sortMode==='label'\)[\s\S]*?if\(sortMode==='signer'\)[\s\S]*?if\(sortMode==='oldest'\)/,'review panel should support label, signer, and date sorting');
assert.match(collectorSource,/data-sample-select=/,'each saved sample should support selection');
assert.match(collectorSource,/filteredSampleIds\.forEach\(id=>\{[\s\S]*?selectedSampleIds\.add\(id\)/,'select filtered should select all filtered sample rows');
assert.match(collectorSource,/async function deleteSamples\(ids\)[\s\S]*?ids\.forEach\(id => store\.delete\(id\)\)/,'batch delete should use one IndexedDB transaction');
assert.match(collectorSource,/Permanently delete '\+ids\.length\+' selected sample/,'batch delete should require confirmation with the selected count');
assert.match(collectorSource,/selectedSampleIds\.size \+ ' selected'[\s\S]*?hidden by filters/,'batch selection should disclose selected samples hidden by current filters');
assert.match(collectorSource,/async function downloadSelectedSamples\(\)[\s\S]*?createZipBlob\(files\)[\s\S]*?manifest\.csv/,'batch download should archive selected source media with a manifest');
assert.match(collectorSource,/downloadSelectedBtn\.disabled = !downloadableCount \|\| blocked/,'batch download should require selected samples with source media');
assert.match(collectorSource,/function analyzeSampleFrames\(sample,canonicalizeFrame\)/,'saved sample cleanup should validate and normalize stored frame arrays');
assert.match(collectorSource,/async function analyzeSavedSamples\(\)[\s\S]*?sampleCleanupPlan=\{validSamples,invalidSamples,droppedFrameCount,droppedNoHandFrameCount\}/,'saved sample cleanup should analyze before enabling destructive changes');
assert.match(collectorSource,/plan\.invalidSamples\.forEach\(\(\{sample\}\)=>store\.delete\(sample\.id\)\)/,'saved sample cleanup should remove only analyzed invalid sample records');
assert.match(collectorSource,/if\(!isNoSign&&normalized\.every\(value=>value===0\)\)\{noHandFrameCount\+\+;continue;\}/,'cleanup should remove no-hand frames from signed samples while retaining NO SIGN negatives');
const validFrameFunction=collectorSource.match(/function isValidFeatureFrame\(frame\) \{[\s\S]*?\n  \}/);
const sampleFrameAnalysisFunction=collectorSource.match(/function analyzeSampleFrames\(sample,canonicalizeFrame\) \{[\s\S]*?\n  \}/);
assert.ok(validFrameFunction&&sampleFrameAnalysisFunction,'saved sample cleanup should expose frame-level analysis helpers');
const sampleFrameAnalysis=vm.runInNewContext(validFrameFunction[0]+'\n'+sampleFrameAnalysisFunction[0]+'\nglobalThis.analyze=analyzeSampleFrames;',{FEATURE_DIMENSION:68});
const usableHandFrame=Array(68).fill(0);usableHandFrame[0]=.2;
const mixedFrameAnalysis=sampleFrameAnalysis.analyze({label:'A',features:[Array(68).fill(0),usableHandFrame,Array(68).fill(NaN)]},frame=>frame);
assert.equal(mixedFrameAnalysis.validFrames.length,1,'signed samples should retain valid hand frames only');
assert.equal(mixedFrameAnalysis.noHandFrameCount,1,'signed samples should count and remove no-hand frames');
assert.equal(mixedFrameAnalysis.malformedFrameCount,1,'signed samples should count malformed frames separately');
assert.equal(sampleFrameAnalysis.analyze({label:'A',features:[Array(68).fill(0)]},frame=>frame).reason,'No usable hand frames','all-no-hand sign samples should be marked for deletion');
assert.equal(sampleFrameAnalysis.analyze({label:'NO SIGN',features:[Array(68).fill(0)]},frame=>frame).reason,'','all-zero NO SIGN negatives should be preserved');
assert.match(collectorSource,/saveSample\(\{\.\.\.sample,label,signerId\}\)/,'sample edits should persist updated labels and signer names');
assert.match(collectorHtml,/id="signerId"[^>]*list="signerOptions"/,'signer alias input should offer saved aliases');
assert.match(collectorHtml,/id="verificationNotes"[^>]*list="reviewerOptions"/,'reference/reviewer input should offer saved notes');
assert.match(collectorHtml,/<datalist id="labelOptions"><option value="NO SIGN"><\/datalist>/,'the collector should offer the reserved background label explicitly');
assert.match(collectorHtml,/usable hand in frame and no intended sign/,'NO SIGN guidance should distinguish visible idle hands from missing-hand frames');
assert.match(collectorHtml,/For NO SIGN, they confirmed that no intended sign is being performed/,'verification should cover reviewed background samples');
assert.match(collectorSource,/startRecordBtn\.disabled = !stream \|\| !trackerReady \|\| capture/,'the recording control should remain clickable so capture validation can explain missing setup requirements');
assert.match(collectorSource,/if \(!verifyCheck\.checked\)[\s\S]*?Have a qualified reviewer confirm that no intended sign is being performed\./,'NO SIGN capture should explain the required reviewer check when recording is attempted');
assert.match(collectorSource,/await saveSample\(sample\);\s*saving = false;\s*updateWorkflow\(\);[\s\S]*?await refreshSamples\(\);/,'recording should unlock after persistence without waiting for library refresh');
assert.match(collectorSource,/const noSignSample = captureMeta\.label\.trim\(\)\.toUpperCase\(\) === 'NO SIGN'/,'capture instructions should explain how to record background samples');
assert.match(collectorSource,/updateSignerOptions\(samples\)/,'sample refresh should populate signer alias suggestions');
assert.match(collectorSource,/updateReviewerOptions\(samples\)/,'sample refresh should populate reviewer suggestions');
assert.match(collectorSource,/savedSampleSignature!==null&&sampleSignature!==savedSampleSignature/,'metadata edits should invalidate the prior training result');
assert.match(collectorSource,/class="sample-management"><summary>Edit and manage<\/summary>[\s\S]*?class="sample-actions"/,'sample actions should be tucked behind one expandable control');
assert.match(collectorHtml,/id="deployModel" disabled/,'model transfer should be disabled until candidate validation');
assert.match(collectorHtml,/id="deploymentMessage"/,'collector should show model transfer readiness');
assert.match(collectorSource,/deployModelBtn\.addEventListener\('click',deployCandidate\)/,'collector should connect the transfer control');
assert.match(collectorSource,/fetch\('\/api\/model\/deploy',\{method:'POST'\}\)/,'collector should request local model deployment');
function trackingProfile(source) {
  assert.match(source,/const PIPELINE\s*=\s*window\.SignARPipeline/,'pipeline should consume the shared configuration module');
  assert.match(source,/window\.SignARTracker\.create/,'tracking should use the shared MediaPipe lifecycle module');
  assert.match(source,/PIPELINE\.createTrackingState\(\)/,'tracking resolution/cadence state should come from shared config');
  assert.match(source,/PIPELINE\.(?:extract|createSequenceTensor)/,'feature and sequence preparation should use shared utilities');
  return extractor.config;
}
assert.deepEqual(trackingProfile(decoderSource), trackingProfile(collectorSource), 'collector and decoder must consume the same tracker and preprocessing config');
const renderedX=[];
const renderContext={save(){},restore(){},clearRect(){},beginPath(){},arc(x){renderedX.push(x)},fill(){},moveTo(){},lineTo(){},stroke(){}};
const renderCanvas={width:0,height:0,getContext(){return renderContext}};
const renderResults={rightHandLandmarks:[{x:.25,y:.5}]};
const renderState=extractor.createTrackingState();
window.SignARLandmarkRenderer.draw(renderCanvas,renderResults,100,100,100,100,renderState);
assert.deepEqual(renderedX,[25],'landmarks should stay in the coordinate system of the original source frame');
renderedX.length=0;
const lowConfidencePose=Array.from({length:33},()=>null);
lowConfidencePose[11]={x:.1,y:.5,visibility:.1};
lowConfidencePose[12]={x:.8,y:.5,visibility:.1};
lowConfidencePose[13]={x:.2,y:.4,visibility:.95};
lowConfidencePose[15]={x:.25,y:.35,visibility:.95};
window.SignARLandmarkRenderer.draw(renderCanvas,{...renderResults,poseLandmarks:lowConfidencePose},100,100,100,100,renderState);
assert.deepEqual(renderedX,[25],'the renderer should omit low-confidence and hand-only pose points while keeping hand landmarks');
renderedX.length=0;
lowConfidencePose[11].visibility=.9;
lowConfidencePose[12].visibility=.9;
window.SignARLandmarkRenderer.draw(renderCanvas,{...renderResults,poseLandmarks:lowConfidencePose},100,100,100,100,renderState);
assert.deepEqual(renderedX,[10,80,20,25,25],'body and arm points should render when both in-frame shoulders clear the configured threshold');
renderedX.length=0;
window.SignARLandmarkRenderer.draw(renderCanvas,renderResults,1600,900,1600,1000,renderState);
assert.ok(Math.abs(renderedX[0]-160)<.01,'renderer should map landmarks into a cropped viewport before the canvas is stretched to match it');
assert.match(collectorSource,/function processLandmarkResults\(results, sourceWidth=video\.videoWidth\|\|PIPELINE\.config\.camera\.width, sourceHeight=video\.videoHeight\|\|PIPELINE\.config\.camera\.height\) \{\s*if\(importSession\)results=swapImportedHandSides\(results\);\s*paintLandmarks\(results,sourceWidth,sourceHeight\);/,'collector should swap imported hand sides before rendering every processed frame');
assert.match(collectorSource,/if\(importSession\.processedFrames%5===0\)\{\s*frameCounter\.textContent = importSession\.frames\.length \+ ' valid frames';\s*\}\s*trackStatus\.textContent = feature \? 'Imported signer tracked' : 'Signer not detected in this frame';/,'batch import tracking status should reflect the current processed frame');
assert.match(collectorSource, /stage\.classList\.toggle\('mirror',facingMode==='user'\)/, 'collector should mirror a front-camera preview like the interpreter');
assert.match(decoderSource, /video\.classList\.toggle\('mirror',facingMode==='user'\)/, 'interpreter should mirror a front-camera preview');
assert.match(decoderSource,/const trackingState=PIPELINE\.createTrackingState\(\);/,'interpreter should use the shared adaptive tracking profile');
assert.doesNotMatch(decoderSource,/trackingState\.dimensionIndex=PIPELINE\.config\.tracking\.dimensions\.length-1/,'interpreter should not force maximum tracker resolution on startup');
assert.match(decoderSource,/function adaptTrackingResolution\(elapsed\)\{PIPELINE\.adaptTrackingState\(trackingState,elapsed\);processEvery=trackingState\.processIntervalMs\}/,'interpreter should follow the shared performance-driven resolution and frame-rate profile');
assert.match(collectorTheme, /\.stage\.mirror video, \.stage\.mirror canvas \{ transform: scaleX\(-1\); \}/, 'collector should mirror video and landmark overlay together');
assert.match(collectorSource, /await tracker\.initialize\(\);\s*stage\.classList\.remove\('mirror'\);/, 'batch imports should retain source orientation rather than inherit camera mirroring');
assert.match(collectorSource, /async function importMediaSample\(file, fallbackLabel\)[\s\S]*?stage\.classList\.remove\('mirror'\);/, 'batch import should clear its preview orientation when processing finishes');
assert.match(collectorHtml, /id="switchCamera"/, 'collector should expose the same front/back camera choice');
assert.match(collectorSource, /processContext\.drawImage\(video,/, 'collector should extract landmarks from untransformed camera pixels');
assert.match(collectorSource, /processContext\.drawImage\(source,0,0,width,height\)/, 'batch imports should track original pixels; mirroring must remain preview-only');
assert.match(decoderSource, /processCtx\.drawImage\(video,/, 'interpreter should extract landmarks from untransformed camera pixels');
for (const source of [collectorSource,decoderSource]) {
  assert.match(source,/generation!==cameraStartGeneration/,'camera processing should discard stale frames after stop or switch');
}
assert.match(trackerSource,/TRACKER_RESULT_TIMEOUT_MS\s*=\s*10000/,'the shared tracker should bound callback waits');
assert.match(trackerSource,/Promise\.race\(\[[\s\S]*?timeoutPromise/,'tracker stalls should surface instead of freezing either frame loop');
assert.match(trackerSource,/PIPELINE\.config\.tracking\.(?:holistic|hands)Options/,'the shared tracker should use centralized MediaPipe options');
assert.doesNotMatch(decoderSource,/await checkFaceDetection\(/,'single-signer checks should not block landmark tracking');
assert.match(decoderSource,/lastFaceDetectionResultAt>PERSON_CHECK_INTERVAL_MS\*3/,'decoder should pause when its multi-person check becomes stale');
assert.doesNotMatch(decoderSource,/\bupdateWorkflow\s*\(/,'decoder camera startup should not call a collector-only workflow helper');
assert.match(collectorSource,/if \(generation===null&&processBusy\) await processIdlePromise/,'media imports should wait for any active camera tracker send');
assert.match(collectorHtml, /id="importMediaFile"[^>]*image\/jpeg/, 'collector should accept common still-image formats');
assert.match(collectorHtml, /id="importMediaFile"[^>]*multiple/, 'collector should allow selecting multiple media files');
assert.match(collectorSource, /function importMediaKind\(file\)/, 'collector should classify imported videos and still images');
const emptyImportFileErrorFunction = collectorSource.match(/function emptyImportFileError\(file\) \{[\s\S]*?\n  \}/);
assert.ok(emptyImportFileErrorFunction, 'collector should detect empty selected files before attempting media decode');
const emptyImportFileError = vm.runInNewContext('(' + emptyImportFileErrorFunction[0] + ')');
assert.equal(emptyImportFileError({ name: 'A.webm', size: 0 }), 'The selected file "A.webm" is empty. Check that its download or export completed, then select it again.', 'empty videos should explain that the source has no bytes instead of reporting a codec failure');
assert.equal(emptyImportFileError({ name: 'valid.webm', size: 1 }), null, 'nonempty files should continue to codec and decoder validation');
assert.match(collectorSource, /const emptyFileError=emptyImportFileError\(file\);[\s\S]*?if\(emptyFileError\)\{importProgress\.textContent=emptyFileError;notify\(emptyFileError\);return false;\}/, 'empty files should be rejected before attempting to decode them');
assert.match(collectorSource, /function importLabelForFile\(file, fallbackLabel = labelSelect\.value\.trim\(\)\)/, 'batch imports should read per-file labels from supported filenames');
assert.match(collectorSource, /for \(let index = 0; index < files.length; index\+\+\)/, 'batch imports should process selected media sequentially');
assert.match(collectorHtml, /id="importBatchStatus"[^>]*role="status"[^>]*aria-live="polite"[^>]*hidden/, 'batch save count should have a dedicated accessible live status');
assert.match(collectorSource, /importBatchStatus\.textContent = 'Saved 0 of '\s*\+\s*files\.length\s*\+\s*' files\.'/,'batch import should show the initial saved count');
assert.match(collectorSource, /importBatchStatus\.textContent = 'Saved '\s*\+\s*importedCount\s*\+\s*' of '\s*\+\s*files\.length\s*\+\s*' files · processing file '/, 'saved count should remain visible while each file is processed');
assert.match(collectorSource, /importBatchStatus\.textContent = 'Saved '\s*\+\s*importedCount\s*\+\s*' of '\s*\+\s*files\.length\s*\+\s*' files'\s*\+\s*\(failures\.length \? ' · '\s*\+\s*failures\.length\s*\+\s*' failed' : ''\)/, 'final batch status should show both saved and failed totals');
assert.match(collectorSource, /async function importMediaSample\(file, fallbackLabel\) \{[\s\S]*?const label = importLabelForFile\(file, fallbackLabel\);\s*labelSelect\.value = label;\s*if\(!validImportForm\(file, label\)\)return false;/, 'each import should update the visible label from its filename before processing');
assert.match(collectorSource, /const sampleLabel=labelSelect\.value\.trim\(\);[\s\S]*?label:sampleLabel/, 'saved sample labels should use the current FSL sign-label field');
assert.match(collectorSource, /importProgress\.textContent='Saving '\+file\.name\+' with FSL sign label: '\+sampleLabel/, 'progress should show the label being saved for each video');
assert.match(interpreterHtml, /id="landmarkCanvas" class="landmark-overlay" aria-hidden="true"/, 'interpreter preview should include a dedicated landmark overlay canvas');
assert.match(interpreterHtml, /id="toggleLandmarksBtn"[^>]*aria-pressed="true"[^>]*aria-controls="landmarkCanvas">Hide landmarks<\/button>/, 'interpreter should expose an accessible landmark visibility toggle');
assert.match(decoderSource, /if\(!landmarksVisible\)return;/, 'hiding landmarks should skip overlay rendering without stopping tracker callbacks');
assert.match(decoderSource, /landmarkCanvas\.hidden=!visible/, 'landmark visibility should only hide the overlay canvas');
assert.match(decoderSource, /toggleLandmarksBtn\.addEventListener\('click',\(\)=>setLandmarksVisible\(!landmarksVisible\)\)/, 'the camera control should toggle landmark visibility');
assert.match(interpreterHtml, /src="\.\/src\/ui\/landmark-renderer\.js"/, 'interpreter should load the shared landmark renderer');
assert.match(collectorHtml, /src="\.\.\/\.\.\/SignLanguageAR\/src\/ui\/landmark-renderer\.js"/, 'collector should load the same landmark renderer');
assert.match(decoderSource, /window\.SignARLandmarkRenderer\.draw\(landmarkCanvas,r,width,height,rect\.width,rect\.height,trackingState\)/, 'interpreter should render tracker output through the shared landmark renderer');
assert.match(decoderSource, /tracker\.serializeInitialization\(\(\)=>detector\.initialize\(\)\)/, 'interpreter face detection should share the serialized MediaPipe initialization queue');
assert.match(decoderSource, /modelLoading=false/,'decoder should expose a distinct model-loading state');
assert.match(decoderSource, /switchBtn\.textContent='Use front camera';updateSentence\(\);modelLoading=true;[\s\S]*?void loadModel\(\)/,'decoder should start background model validation and loading on page startup');
assert.match(collectorSource, /window\.SignARLandmarkRenderer\.draw\(canvas,r,sw,sh,rect\.width,rect\.height,activeTrackingState\)/, 'collector should render imported landmarks in source-frame coordinates');
assert.match(landmarkRendererSource, /const HAND_CONNECTIONS=/, 'shared renderer should define hand connections once');
assert.match(landmarkRendererSource, /const POSE_CONNECTIONS=/, 'shared renderer should define pose connections once');
assert.match(landmarkRendererSource, /const FACE_PATHS=/, 'shared renderer should define face paths once');
assert.match(decoderSource, /landmarkCanvas\.classList\.toggle\('mirror',facingMode==='user'\)/, 'landmark overlay should mirror with the front-camera preview');
assert.match(interpreterStyles, /\.landmark-overlay \{[^}]*object-fit: fill[^}]*pointer-events: none/s, 'landmark overlay should reuse the renderer crop without applying a second crop');
const labelFunction = collectorSource.match(/function importLabelForFile\(file, fallbackLabel = labelSelect\.value\.trim\(\)\) \{[\s\S]*?\n  \}/);
assert.ok(labelFunction, 'batch label parser should be available for behavior checks');
const handSwapFunction = collectorSource.match(/function swapImportedHandSides\(results\) \{[\s\S]*?\n  \}/);
assert.ok(handSwapFunction, 'batch import should have a dedicated left/right hand swap');
const swapImportedHandSides = vm.runInNewContext('(' + handSwapFunction[0].replace(/^function swapImportedHandSides/, 'function') + ')');
const importedLeftHand = [{x:0.2,y:0.5}], importedRightHand = [{x:0.8,y:0.5}];
const swappedImportHands = swapImportedHandSides({leftHandLandmarks:importedLeftHand,rightHandLandmarks:importedRightHand});
assert.equal(swappedImportHands.leftHandLandmarks,importedRightHand,'batch import should move right-hand results into the left slot');
assert.equal(swappedImportHands.rightHandLandmarks,importedLeftHand,'batch import should move left-hand results into the right slot');
const featureSwapFunction=collectorSource.match(/function swapImportedFeatureHandSlots\(frame\) \{[\s\S]*?\n  \}/);
const sampleMigrationFunction=collectorSource.match(/function migrateImportedSampleRecordHandSlots\(sample\) \{[\s\S]*?\n  \}/);
assert.ok(featureSwapFunction&&sampleMigrationFunction,'existing imported sample features should have a dedicated one-time migration');
const migrationContext={PIPELINE:extractor,POSE_FEATURE_VALUES:132,HAND_FEATURE_VALUES:126,IMPORT_HAND_SIDE_VERSION:1};
const swapFeatureHandSlots=vm.runInNewContext('('+featureSwapFunction[0].replace(/^function swapImportedFeatureHandSlots/,'function')+')',migrationContext);
migrationContext.swapImportedFeatureHandSlots=swapFeatureHandSlots;
const migrateImportedSample=vm.runInNewContext('('+sampleMigrationFunction[0].replace(/^function migrateImportedSampleRecordHandSlots/,'function')+')',migrationContext);
const storedFrame=Array.from({length:extractor.dimension},(_,index)=>index),originalSourceBlob={size:10};
const migratedSample=migrateImportedSample({id:'old-import',sourceType:'imported-video',features:[storedFrame],sourceBlob:originalSourceBlob});
assert.deepEqual(Array.from(migratedSample.features[0].slice(0,132)),storedFrame.slice(0,132));
assert.deepEqual(Array.from(migratedSample.features[0].slice(132,258)),storedFrame.slice(258,384),'migration should move old right-hand features into the left slot');
assert.deepEqual(Array.from(migratedSample.features[0].slice(258,384)),storedFrame.slice(132,258),'migration should move old left-hand features into the right slot');
assert.deepEqual(Array.from(migratedSample.features[0].slice(384)),storedFrame.slice(384));
assert.equal(migratedSample.sourceBlob,originalSourceBlob,'migration should preserve the original source media');
assert.equal(migratedSample.importHandSideVersion,1);
assert.equal(migrateImportedSample({...migratedSample,sourceType:'camera'}),null,'camera samples should not be migrated');
assert.equal(migrateImportedSample(migratedSample),null,'already corrected imports should not be migrated twice');
assert.match(collectorSource,/migratedSamples=await migrateLegacyImportedHandSlots\(\)/,'the collector should migrate older saved imports automatically on startup');
assert.match(collectorSource,/sourceType:'imported-'\+kind,importHandSideVersion:IMPORT_HAND_SIDE_VERSION/,'new imports should be marked so startup does not swap them twice');
assert.match(collectorSource,/Source files were unchanged/,'the migration should tell users that their original videos were preserved');
const importLabelForFile = vm.runInNewContext('(' + labelFunction[0] + ')', {
  signerInput: { value: 'signer-01' },
  labelSelect: { value: 'fallback label' }
});
assert.equal(importLabelForFile({ name: 'signer-01_A_94b86dbe-e01b-4180-a637-8966e4702069.webm' }), 'A', 'batch import should extract the sign label from the tagged clip filename');
assert.equal(importLabelForFile({ name: 'my-clip.webm' }), 'fallback label', 'batch import should use the entered label for other filenames');
assert.equal(importLabelForFile({ name: 'my-clip.webm' }, 'batch label'), 'batch label', 'batch imports should preserve the initial fallback label across files');
const labelFunctionWithDifferentSigner = vm.runInNewContext('(' + labelFunction[0] + ')', {
  signerInput: { value: 'another-signer' },
  labelSelect: { value: 'fallback label' }
});
assert.equal(labelFunctionWithDifferentSigner({ name: 'signer-01_A_94b86dbe-e01b-4180-a637-8966e4702069.webm' }), 'A', 'filename labels should not depend on the signer alias entered in the form');
for (let letterCode = 65; letterCode <= 90; letterCode++) {
  const letter = String.fromCharCode(letterCode);
  assert.equal(importLabelForFile({ name: 'signer-01_' + letter + '_00000000-0000-4000-8000-000000000000.webm' }), letter, 'batch import should preserve each alphabetic filename label');
  assert.equal(importLabelForFile({ name: letter + '.webm' }), letter, 'batch import should use a single-letter video filename as its label');
}
assert.match(collectorSource, /processImportedFrame\(kind==='image'\?image:video,sourceWidth,sourceHeight,steps===1\)/, 'single-frame imports should wait for still-image hand tracking');
assert.match(collectorSource, /processTrackingFrame\(processCanvas,sourceWidth,sourceHeight,waitForHands\)/, 'imported frames should pass hand-tracker readiness through the frame helper');
assert.match(collectorSource, /Math\.ceil\(duration\s*\*\s*PIPELINE\.config\.sequence\.samplingFps\)/, 'video duration in seconds should be sampled using the configured frames per second');
assert.equal(Math.ceil(2 * extractor.config.sequence.samplingFps), 30, 'a two-second video should analyze 30 frames at 15 samples per second');
const importFrameCountFunction = collectorSource.match(/function importedVideoFrameCount\(duration\) \{[\s\S]*?\n  \}/);
assert.ok(importFrameCountFunction, 'video imports should bound frame analysis');
const importedVideoFrameCount = vm.runInNewContext('(' + importFrameCountFunction[0] + ')', {
  MIN_IMPORTED_FRAMES: extractor.config.sequence.minImportedFrames,
  MAX_IMPORTED_FRAMES: extractor.config.sequence.length,
  PIPELINE: extractor
});
assert.equal(importedVideoFrameCount(2), 30, 'normal two-second video sampling should remain unchanged');
assert.equal(importedVideoFrameCount(60), extractor.config.sequence.length, 'long video analysis should cap at the model sequence length');
assert.match(collectorSource, /if\(sampledFrames>=MAX_IMPORTED_FRAMES\)return/, 'unknown-duration videos should also bound tracker work');
assert.match(collectorHtml, /video analysis is limited to 30 sampled frames per clip/, 'the import UI should disclose the analysis limit');
assert.match(collectorSource, /video\.preload='metadata'/, 'large video imports should avoid eager full-file preloading');
assert.match(collectorSource, /await saveSample\(sample\);\s*if\(!importBatchRunning\)await refreshSamples\(\)/, 'batch imports should avoid refreshing the full sample library after every file');
assert.match(collectorSource, /if \(importedCount\) \{\s*try \{\s*await refreshSamples\(\)/, 'batch imports should refresh the saved sample library once after processing');
assert.match(collectorSource, /MediaRecorder\.isTypeSupported\(t\)&&video\.canPlayType\(t\)/, 'camera recordings should use a codec the same browser can play');
assert.match(collectorSource, /const candidates=\['video\/webm;codecs=vp8','video\/mp4;codecs=avc1\.42E01E','video\/webm;codecs=vp9'\]/, 'prefer broadly playable VP8 and avoid unspecified WebM codecs');
assert.match(collectorSource, /if\s*\(!mimeType\)\s*\{\s*notify\(/, 'do not record with an unverified browser-default codec');
assert.match(collectorSource, /function readableVideoDuration\(\)[\s\S]*?video\.seekable\.end\(video\.seekable\.length-1\)/, 'imported WebM should use a finite seekable range when duration metadata is missing');
assert.match(collectorSource, /function seekImportedVideo\(time\)[\s\S]*?readableVideoDuration\(\)/, 'video seeks should use the readable duration fallback');
assert.match(collectorSource, /if\(video\.error\)throw new Error\(videoDecodeErrorMessage\(\)\)/, 'surface unsupported-codec errors before reporting missing duration');
assert.match(collectorSource, /async function importUnknownDurationVideo[\s\S]*await video\.play\(\)[\s\S]*while\(!video\.ended\)/, 'decode durationless videos by sampling during playback through EOF');
assert.doesNotMatch(collectorSource, /This video has no readable duration metadata/, 'durationless videos should not be rejected before playback sampling');
assert.match(collectorHtml, /class="training-progress-track"/, 'training progress should have a dedicated visual track');
assert.doesNotMatch(collectorHtml, /training-progress-indicator/, 'training progress should use the determinate bar, not an indeterminate overlay');
assert.match(collectorSource, /trainingProgressWrap\.dataset\.state = 'running'/, 'collector should expose the running state to the trainer spinner');
assert.match(collectorHtml, /class="training-spinner"/, 'trainer status should include a dedicated loading spinner');
assert.match(collectorSource, /trainingProgressWrap\.setAttribute\('aria-busy','true'\)/, 'trainer status should announce when work is active');
assert.match(collectorSource, /Elapsed '\+Math\.floor\(elapsed\/60\)/, 'trainer status should show elapsed runtime while the process runs');
assert.match(collectorSource, /fetch\('\/api\/training\/progress'/, 'collector should poll the local trainer for real progress');
assert.match(collectorSource, /trainingProgress\.value\s*=\s*Math\.max\(trainingProgress\.value/, 'progress percentage should not move backward');
assert.match(studioServerSource, /route == "\/api\/training\/progress"/, 'local server should expose the current training progress');
assert.match(studioServerSource, /SIGNAR_PROGRESS/, 'server should parse completed training epochs');
assert.match(trainerSource, /SIGNAR_PROGRESS \{epoch \+ 1\}\/\{args\.epochs\}/, 'Python trainer should report each completed epoch');
assert.match(collectorTheme, /@keyframes training-spinner-spin/, 'trainer loading spinner should visibly animate');
assert.doesNotMatch(collectorTheme, /training-progress-slide/, 'determinate progress should not be obscured by a looping animation');
assert.match(collectorTheme, /prefers-reduced-motion: reduce/, 'progress animation should respect reduced-motion preferences');
assert.match(collectorHtml, /id="evaluationScope"/, 'collector should expose the evaluation scope metric');
assert.match(collectorHtml, /id="deployModel" disabled/, 'model transfer should be disabled until candidate validation');
assert.match(collectorHtml, /id="deploymentMessage"/, 'collector should explain model transfer readiness');
assert.match(collectorSource, /deployModelBtn\.addEventListener\('click',deployCandidate\)/, 'collector should connect the transfer control');
assert.match(collectorSource, /fetch\('\/api\/model\/deploy',\{method:'POST'\}\)/, 'collector should request local model deployment');
assert.match(collectorSource, /evaluationScopeMetric\.textContent = evaluation\.evaluationScope === 'signer-independent'/, 'collector should display the signer-independent scope');
assert.match(collectorSource, /evaluation\.evaluationScope === 'sample-level' \? 'Sample-level'/, 'collector should display the sample-level scope');
assert.match(collectorSource, /evaluation\.evaluationScope === 'not-evaluated' \? 'Not evaluated'/, 'collector should display when no held-out evaluation was possible');
for (const id of ['trainingCoverage','trainingEvaluationDetails','trainingClassMetrics','trainingEvaluationNote']) {
  assert.match(collectorHtml,new RegExp('id="'+id+'"'),'trainer diagnostics should expose '+id);
}
assert.match(collectorSource,/trainingCoverage\.textContent=trainingCoverageSummary\(samples\)/,'sample refresh should update label and signer coverage');
assert.match(collectorSource,/renderTrainingEvaluation\(evaluation\)/,'completed training should display per-sign evaluation');
assert.match(collectorSource,/The trainer did not have held-out test clips, so no per-sign metrics were computed/,'unavailable evaluation should not invent per-sign results');
assert.match(collectorSource,/complete per-sign metrics were missing from the trainer response/,'incomplete evaluation should be disclosed separately from unavailable tests');
const coverageSummaryFunction = collectorSource.match(/function trainingCoverageSummary\(samples\) \{[\s\S]*?\n  \}/);
assert.ok(coverageSummaryFunction, 'training coverage should be available for behavior checks');
const trainingCoverageSummary = vm.runInNewContext('(' + coverageSummaryFunction[0] + ')');
const limitedCoverage = trainingCoverageSummary([
  { label: 'A', signerId: 'signer-01' },
  { label: 'A', signerId: 'signer-02' },
  { label: 'A', signerId: 'signer-03' },
  { label: 'B', signerId: 'signer-01' }
]);
assert.match(limitedCoverage,/4 samples · 2 labels · 3 signers/,'coverage should count samples, labels, and distinct signer aliases');
assert.match(limitedCoverage,/Add samples to reach 3 per label for held-out evaluation: B \(1\)/,'coverage should identify labels below the sample-level split minimum');
assert.match(limitedCoverage,/signer-independent evaluation needs 5 signers overall and at least 3 per label: B \(1\)/i,'coverage should identify signer-independent class gaps');
const signerOptionsFunction = collectorSource.match(/function updateSignerOptions\(samples = \[\]\) \{[\s\S]*?\n  \}/);
const reviewerOptionsFunction = collectorSource.match(/function updateReviewerOptions\(samples = \[\]\) \{[\s\S]*?\n  \}/);
assert.ok(signerOptionsFunction && reviewerOptionsFunction,'saved signer and reviewer options should have dedicated updaters');
const optionSamples = [
  { signerId: 'signer-02', verificationNotes: 'Teacher review <FSL>' },
  { signerId: 'signer-01', verificationNotes: 'Reference checked' },
  { signerId: 'signer-02', verificationNotes: 'Reference checked' }
];
const signerOptionList = { innerHTML: '' };
vm.runInNewContext('(' + signerOptionsFunction[0] + ')', {
  signerOptions: signerOptionList,
  safeText: value => String(value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]))
})(optionSamples);
assert.match(signerOptionList.innerHTML,/signer-01/,'signer suggestions should include saved aliases');
assert.match(signerOptionList.innerHTML,/signer-02/,'signer suggestions should include all unique saved aliases');
assert.equal((signerOptionList.innerHTML.match(/<option/g) || []).length,2,'signer suggestions should be unique');
const reviewerOptionList = { innerHTML: '' };
vm.runInNewContext('(' + reviewerOptionsFunction[0] + ')', {
  reviewerOptions: reviewerOptionList,
  safeText: value => String(value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]))
})(optionSamples);
assert.match(reviewerOptionList.innerHTML,/Reference checked/,'reviewer suggestions should include saved notes');
assert.match(reviewerOptionList.innerHTML,/Teacher review &lt;FSL&gt;/,'reviewer suggestions should safely escape saved text');
assert.equal((reviewerOptionList.innerHTML.match(/<option/g) || []).length,2,'reviewer suggestions should be unique');
const evaluationRowsFunction = collectorSource.match(/function evaluationMetricRows\(evaluation\) \{[\s\S]*?\n  \}/);
assert.ok(evaluationRowsFunction, 'per-sign metrics should be available for behavior checks');
const evaluationMetricRows = vm.runInNewContext('(' + evaluationRowsFunction[0] + ')');
const weakestFirstRows = evaluationMetricRows({
  labels: ['A', 'B'],
  classificationReport: {
    A: { precision: 0.8, recall: 0.9, 'f1-score': 0.85, support: 4 },
    B: { precision: 0.4, recall: 0.2, 'f1-score': 0.27, support: 3 },
    accuracy: 0.6
  }
});
assert.deepEqual(Array.from(weakestFirstRows, row => row.label), ['B', 'A'],'per-sign metrics should prioritize the lowest recall');
assert.deepEqual(Array.from(evaluationMetricRows({ labels: ['A'], classificationReport: {} })), [],'missing held-out metrics should remain unavailable');
assert.deepEqual(Array.from(evaluationMetricRows({
  labels: ['A', 'B'],
  classificationReport: { A: { precision: 1, recall: 1, 'f1-score': 1, support: 1 } }
})), [],'partial class reports should not be presented as complete evaluation');
assert.match(decoderSource, /evaluation\.deploymentEligible!==true/, 'interpreter should require explicit deployment eligibility');
assert.match(decoderSource, /if\(blockers\.length\|\|!modelConfig\.modelPath\)/, 'interpreter should gate model loading on evaluation blockers');
assert.match(decoderSource, /Number\.isFinite\(evaluation\.testAccuracy\)&&Number\.isFinite\(evaluation\.minimumAccuracy\)/, 'interpreter should retain its explicit accuracy threshold check');
assert.doesNotMatch(collectorSource + decoderSource, /evaluationMeaningful|signerIndependent|passedMinimum/, 'collector and interpreter should not depend on removed evaluation flags');
const assertDecoderParity = (results, label) => {
  const collected = extractor.extract(results, 640, 480);
  const decoded = decoderExtractor.extract(results, 640, 480);
  assert.equal(JSON.stringify(decoded), JSON.stringify(collected), label + ' feature parity');
};
const point = (x, y, z = 0, visibility = 1) => ({ x, y, z, visibility });
const makeHand = (offsetX = 0.5, offsetY = 0.5) => Array.from({ length: 21 }, (_, i) => {
  const col = i % 5, row = Math.floor(i / 5);
  return { x: offsetX + (col - 2) * 0.012, y: offsetY + row * 0.012, z: 0 };
});
assert.equal(extractor.hasUsableHand(makeHand().slice(0, 2)), false, 'two tracked points should not qualify as a usable signing hand');
assert.equal(extractor.hasUsableHand(makeHand().slice(0, 3)), true, 'three tracked points should meet the shared minimum hand quality');
const makeFace = () => {
  const face = Array.from({ length: 468 }, (_, i) => ({ x: 0.5, y: 0.35, z: 0 }));
  for (const [n, id] of extractor.faceIds.entries()) face[id] = { x: 0.42 + (n % 6) * 0.025, y: 0.3 + Math.floor(n / 6) * 0.025, z: 0 };
  return face;
};
const assertFrame = (frame, label) => {
  assert.ok(Array.isArray(frame), label + ' should produce a frame');
  assert.equal(frame.length, 456, label + ' has fixed feature width');
  assert.ok(frame.every(Number.isFinite), label + ' produces finite values');
};

// One hand only: no full body, other hand, or face required.
const oneHand = extractor.extract({ leftHandLandmarks: makeHand(), rightHandLandmarks: [], poseLandmarks: [], faceLandmarks: [] }, 640, 480);
assertFrame(oneHand, 'one-hand-only clip');
assert.ok(oneHand.slice(132, 258).some(v => v !== extractor.missingValue));
assert.ok(oneHand.slice(258, 384).every(v => v === extractor.missingValue));
assert.ok(oneHand.slice(384).every(v => v === extractor.missingValue));
assertDecoderParity({ leftHandLandmarks: makeHand(), rightHandLandmarks: [], poseLandmarks: [], faceLandmarks: [] }, 'one-hand-only clip');

const oneHandMoved = extractor.extract({ leftHandLandmarks: makeHand(0.7, 0.5), rightHandLandmarks: [], poseLandmarks: [], faceLandmarks: [] }, 640, 480);
assertFrame(oneHandMoved, 'moved one-hand-only clip');
assert.notDeepEqual(oneHand.slice(132, 258), oneHandMoved.slice(132, 258), 'hand-only directional movement should remain represented');
assertDecoderParity({ leftHandLandmarks: makeHand(0.7, 0.5), rightHandLandmarks: [], poseLandmarks: [], faceLandmarks: [] }, 'moved one-hand-only clip');

// Face plus a hand and partial pose, with shoulders and hips absent.
const partialPose = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
partialPose[13] = point(0.4, 0.45, 0, 0.95);
partialPose[15] = point(0.35, 0.5, 0, 0.95);
const faceAndPartialPose = extractor.extract({ poseLandmarks: partialPose, faceLandmarks: makeFace(), leftHandLandmarks: [], rightHandLandmarks: makeHand(0.35, 0.5) }, 640, 480);
assertFrame(faceAndPartialPose, 'partial upper-body + face + one hand');
assertDecoderParity({ poseLandmarks: partialPose, faceLandmarks: makeFace(), leftHandLandmarks: [], rightHandLandmarks: makeHand(0.35, 0.5) }, 'partial upper-body + face + one hand');

// Face-only footage is accepted when several selected facial landmarks are visible.
const faceOnly = extractor.extract({ faceLandmarks: makeFace(), poseLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 640, 480);
assertFrame(faceOnly, 'face-only clip');
assertDecoderParity({ faceLandmarks: makeFace(), poseLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 'face-only clip');

// Partial upper-body pose is accepted without shoulders, hips, hands, or face.
const armOnlyPose = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
armOnlyPose[15] = point(0.42, 0.40, 0, 0.9);
armOnlyPose[16] = point(0.58, 0.46, 0, 0.9);
const poseOnly = extractor.extract({ poseLandmarks: armOnlyPose, faceLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 640, 480);
assertFrame(poseOnly, 'partial-pose-only clip');
assertDecoderParity({ poseLandmarks: armOnlyPose, faceLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 'partial-pose-only clip');

// Missing all meaningful regions is still rejected so blank/background video isn't saved.
assert.equal(extractor.extract({ poseLandmarks: [], faceLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }), null);
assertDecoderParity({ poseLandmarks: [], faceLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [] }, 'empty frame');

// Missing-value sentinels must remain discrete during temporal resampling.
const a = new Array(456).fill(0), b = new Array(456).fill(0);
a[132] = extractor.missingValue;
const resampled = extractor.resampleSequence([a, b], 3);
assert.equal(JSON.stringify(decoderExtractor.resampleSequence([a, b], 3)), JSON.stringify(resampled), 'decoder and collector sequence resampling should match');
assert.equal(resampled[0][132], extractor.missingValue);
assert.equal(resampled[1][132], 0);
assert.equal(resampled[2][132], 0);
const oneFrameResampled = extractor.resampleSequence([a], 30);
assert.equal(oneFrameResampled.length, 30, 'one-frame video should be expanded to the model sequence length');
assert.ok(oneFrameResampled.every(frame => JSON.stringify(frame) === JSON.stringify(a)), 'one-frame video should preserve its static snapshot');
const longSequence=Array.from({length:35},(_,index)=>new Array(456).fill(index));
const recentWindow=extractor.selectRecentSequenceWindow(longSequence);
assert.equal(recentWindow.length,30,'live model windows should keep the model input length');
assert.equal(recentWindow[0][0],5,'long live sequences should retain the newest two-second window instead of compressing the whole gesture');
assert.equal(recentWindow[29][0],34,'recent windows should end at the latest tracked frame');
assert.equal(extractor.selectRecentSequenceWindow([a]).length,30,'short signs should still be expanded to the fixed model window');
assert.match(decoderSource,/PIPELINE\.createRecentSequenceTensor\(window\.tf,segmentFrames\)/,'live inference should classify a rolling recent window');
assert.match(decoderSource,/preRollFrames\.slice\(\)/,'live segmentation should retain a short pre-roll before motion starts');
console.log('Partial-landmark feature tests passed.');
