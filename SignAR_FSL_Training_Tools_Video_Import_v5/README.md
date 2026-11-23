# SignAR private FSL data studio

This local toolkit collects, reviews, exports, and evaluates landmark sequences for SignAR. Camera frames and features are processed locally. Raw recordings and exported datasets remain private unless you deliberately move them.

## Holistic-v4 feature schema

The collector shares `SignLanguageAR/src/core/fsl-pipeline.js` and `SignLanguageAR/src/tracker/mediapipe-tracker.js`. New samples use MediaPipe Tasks Vision HolisticLandmarker and export 30 ordered frames, sampled at 15 FPS. Each frame has 422 values:

- 71 per hand: 21 wrist-relative, palm-scaled 3D landmarks, five finger angles, and three body-relative wrist-placement values.
- 132 body values: 33 landmarks with normalized position, depth, and visibility.
- 144 face values: 48 landmarks around the eyes, eyebrows, nose, and mouth.
- Four presence flags for the left hand, right hand, body, and face. Missing features use an explicit `-3` sentinel.

Labels are open-ended. Each records its unit type (`fingerspelling`, `word`, `phrase`, or `background`), motion type (`static` or `dynamic`), and optional display text. Legacy hand-v3 and 456-value samples are converted into holistic-v4 during export, with unavailable cues marked missing.

## Collect

Start `start_training_studio.bat`, enter a signer alias, verify the sign form, record consent, and add a short reviewer/reference note. A sample can be a fingerspelling letter, word sign, whole phrase sign, or background; select static or dynamic movement and provide display text when needed. Record the complete sign through the live holistic tracker or import reviewed media. Dynamic signs need video showing the full movement; still images cannot establish their movement. Keep both hands, face, and upper body in frame for signs that use those cues.

Choose **NO SIGN** for varied neutral posture and non-sign movement. The collector keeps frames with no detected cues so the sequence preserves timing. Live-camera samples are optional; imported media can provide all signer and label coverage.

The SignAR standard validation profile requires five signers for every declared label and at least 10 NO SIGN samples (1% to 25% of the data). It also requires 90% mean signer-held-out accuracy, 80% recall for every non-background sign label, no more than 10% NO SIGN false accepts, and at least 90% on every stress check. Live-camera samples and their counts are optional and are not a gate. Metrics describe the recorded dataset and may differ across signers, cameras, and environments.

## Train and evaluate

Use **Train candidate** in the studio. `train.ps1` calls `training/train_hand_model.py`, which trains on holistic-v4 ordered sequences and supports migration from hand-v3 and legacy 456 datasets. Training augmentation applies temporal warping, left/right slot variation, spatial jitter, scale/rotation, and occasional missing frames while preserving ordered movement. Temperature scaling is fitted on a held-out signer. Evaluation includes leave-one-signer-out folds, class recall, confusion matrix, calibration error, NO SIGN false-accept rate, optional camera counts, and stress checks for missing regions, slot swaps, jitter, scale/rotation, static holds, and dropped frames.

The interpreter implements continuous stream segmentation separately from model training. It confirms prediction runs and closes a sign segment at a stable class change or a 450 ms pause. Phrase labels still represent one whole phrase sign; the boundary layer does not infer untrained vocabulary or FSL grammar.

For a direct training run:

```powershell
.\.venv\Scripts\python.exe training\train_hand_model.py --dataset fsl-dataset.json --output-dir trained-model
```

Review `trained-model/evaluation.json`. Eligible TensorFlow.js artifacts are exported only when the SignAR validation checks pass. Transfer the candidate from the local collection page after reviewing the vocabulary and evaluation.

Keep raw video, signer aliases, exported features, and private references out of the public application folder. The public decoder contains no collection, training, or simulation controls.
