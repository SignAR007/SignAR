# SignAR

SignAR is a browser-based Filipino Sign Language recognition system. Its vocabulary can include fingerspelling, word signs, whole phrase signs, static signs, and dynamic signs. Camera frames and landmarks are processed locally. The decoder displays labels emitted by a compatible model.

## Current model status

The installed model uses 10 frames of 68 hand-v3 values and recognizes the existing fingerspelling vocabulary. The live tracker and collector use holistic-v4: 30 frames that represent both hands, face, and body. The interpreter selects features from the version declared by the model, so the installed weights keep working while a holistic-v4 candidate is collected, trained, evaluated, and transferred.

The installed weights were evaluated on 992 sample windows: 969 imported-video samples and 23 camera samples. Every label has five-signer coverage; all camera samples are NO SIGN, so live-camera recognition of the letters is unverified. The five signer-held-out folds average 97.85% accuracy. NO SIGN recall is 99.06%, the NO SIGN false-accept rate is 0.47%, and NO SIGN is 21.37% of the dataset. See `model/evaluation.json` for the report. The SignAR standard validation profile requires 90% mean accuracy, 80% recall for every sign label, at most 10% NO SIGN false accepts, at least 90% on stress checks, five signers per label, and at least 10 NO SIGN samples (1% to 25% share). Live-camera samples are optional. These figures describe the hand-v3 model and do not validate holistic-v4 recognition or performance on new camera sessions.

## Decoder pipeline

- `src/core/fsl-pipeline.js` emits 422 values per holistic-v4 frame: 71 values for each hand, 33 body landmarks, 48 face points, and four cue-presence flags. Missing regions use explicit sentinel values.
- `src/tracker/mediapipe-tracker.js` uses pinned MediaPipe Tasks Vision HolisticLandmarker in VIDEO mode. It tries the GPU delegate and falls back to CPU. A separate face-count check marks multi-person views while decoding continues from the landmark track HolisticLandmarker currently follows.
- `src/interpreter/app.js` keeps a rolling three-second holistic-v4 window (two seconds for the legacy hand-v3 model), resamples it to the model's expected frames, and benchmarks WebGPU, WebGL, WebAssembly, and CPU on the loaded decoder to select the fastest working backend. Inference is eligible on each incoming 15 fps feature update. The streaming segmenter closes a sign at a stable class change or after a 450 ms pause, including repeated labels after a pause. Label metadata supplies display text and static/dynamic names.
- `src/interpreter/stream-segmenter.js` commits class boundaries into the live transcript. It segments recognized units; it does not translate untrained signs or infer FSL grammar.
- The interpreter reads both holistic-v4 and legacy hand-v3 model contracts. New holistic cues and vocabulary become recognized after a matching candidate is trained and transferred.
- `?debug=1` enables hand-slot, face/body, source, and stage timing telemetry. The export control keeps at most 100 recent live windows for offline replay; this cap is unrelated to training sample requirements.

## Collect and train

The private collector and trainer are in `../SignAR_FSL_Training_Tools_Video_Import_v5/`. Labels can be fingerspelling letters, words, whole phrase signs, or background, with static/dynamic movement and optional display text. Dynamic examples preserve their full movement in ordered 30-frame windows. Samples may come from the live holistic tracker or reviewed imported media; live-camera sample counts are optional and do not affect eligibility. For NO SIGN, collect varied neutral posture and non-sign movement; frames without detected cues are stored with missing-region flags.

The standard validation profile requires five signers covering each label and at least 10 NO SIGN examples (1% to 25% of the dataset); live-camera samples are optional. `training/train_hand_model.py` learns from ordered holistic sequences, calibrates temperature on a held-out signer, runs leave-one-signer-out evaluation, and reports per-class recall, confusion matrix, calibration error, NO SIGN false-accept rate, stress results, and optional camera coverage.

Start the local studio with `start_training_studio.bat`, collect and export the dataset, then use **Train candidate**. The trainer is invoked through `train.ps1`; it writes artifacts only to the private toolkit `trained-model/` folder. Review the evaluation and confirm sign forms before transferring a candidate into `model/`.

Raw videos and signer data should stay in private local storage. The public decoder contains no collector, trainer, simulation, or sample-prediction mode.

## Run locally

Serve this folder over HTTPS or localhost, for example:

```sh
npx serve .
```

Open the served page, allow camera access, and select **Start camera**. Use `npm run format:check` to check formatting.
