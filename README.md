# SignAR: Filipino Sign Language Recognition System

SignAR is a browser-based system for recognizing Filipino Sign Language (FSL) fingerspelling, isolated word and phrase signs, and static or dynamic sign sequences. Its feature, data, training, and output contracts support these sign types. This workspace contains a public browser decoder and a separate local-first training toolkit.

## System map

| Area | Purpose | Start here |
|---|---|---|
| [SignLanguageAR/](./SignLanguageAR/) | Public browser decoder for supported FSL signs | [Decoder guide](./SignLanguageAR/README.md) |
| [SignAR_FSL_Training_Tools_Video_Import_v5/](./SignAR_FSL_Training_Tools_Video_Import_v5/) | Private, local-first sample collection, validation, and model training; not part of the public deployment | [Toolkit guide](./SignAR_FSL_Training_Tools_Video_Import_v5/README.md) |
| Training workflow | Train and evaluate SignAR models | [Training guide](./SignAR_FSL_Training_Tools_Video_Import_v5/TRAINING_GUIDE.md) |

### Decoder layout

```text
SignLanguageAR/
  index.html                         Static browser entry point
  src/
    core/fsl-pipeline.js             Versioned holistic features and sequence/tensor helpers
    tracker/mediapipe-tracker.js     Shared MediaPipe holistic tracking lifecycle
    interpreter/app.js               Live camera, sequence inference, and phrase segmentation
    interpreter/stream-segmenter.js   Online sign-unit boundaries from class changes and pauses
    ui/                              Shared landmark renderer, theme controller, and decoder stylesheet
  model/                             SignAR model manifests and artifacts
  icons/, manifest.webmanifest       Installable web app assets
  sw.js                              Static asset cache
```

### Training toolkit layout

```text
SignAR_FSL_Training_Tools_Video_Import_v5/
  training/collect.html              Local data-collection page
  src/collector/                     Browser sample and training modules
  src/ui/                            Collector styles
  training/train_hand_model.py       Sequence-sign training and TensorFlow.js export
  training/validate_dataset.py       Dataset checks and readiness report
  training/augmentation.py           Training-only data augmentation
  train.bat, train.ps1               Windows setup, validation, and training launchers
  TRAINING_GUIDE.md                  Detailed training and deployment workflow
```

Both browser workflows share a holistic tracker and a versioned feature pipeline. New training exports use 30-frame windows containing both hands, 33 body landmarks, 48 facial landmarks around the eyes, eyebrows, nose, and mouth, plus cue-presence flags. Labels can be fingerspelling, words, whole phrase signs, static signs, or dynamic signs. The streaming decoder segments consecutive predictions when a different sign becomes stable or a pause is detected, then composes the configured sign text. It benchmarks WebGPU, WebGL, WebAssembly, and CPU on each loaded model and uses the fastest working backend available on that device. This boundary layer segments recognized labels; it does not translate untrained signs or infer grammar. The installed weights remain a legacy 10×68 hand-v3 fingerspelling model until a holistic-v4 candidate is trained and transferred. The collector is local-only; keep its toolkit files outside the public Pages artifact.

This repository includes the decoder, local training-tool source, tests, and technical documentation. Deploy only `SignLanguageAR/` as the public website; do not expose the training collector on the deployed site. The top-level `.gitignore` excludes local environments, generated model artifacts, sample datasets, reports, and videos. Source tests and synthetic-data generators remain available in GitHub for review.

### Deploy to GitHub Pages

The [Pages workflow](./.github/workflows/pages.yml) publishes only `SignLanguageAR/`; the toolkit source and tests remain in the repository but are not served as the website. In the repository settings, set Pages build and deployment source to GitHub Actions. Push to `main` or run the workflow manually to publish the decoder.

## Quick start

### Run the public decoder

Serve [SignLanguageAR/](./SignLanguageAR/) from localhost or HTTPS and open the site in a modern browser. The decoder loads the installed SignAR model and its supported vocabulary. Collection, training, and candidate review are handled by the separate local toolkit and are not part of the deployed site. See the [decoder guide](./SignLanguageAR/README.md).

### Collect data and train

Run the local collection and training studio from the workspace root by double-clicking `SignAR_FSL_Training_Tools_Video_Import_v5\start_training_studio.bat`. Keep its console open, then visit:

`http://localhost:8001/SignAR_FSL_Training_Tools_Video_Import_v5/training/collect.html`

The collector stores samples in browser IndexedDB, exports landmark JSON, and can invoke the local Python trainer through the studio API. To train manually from an exported dataset:

```powershell
Set-Location .\SignAR_FSL_Training_Tools_Video_Import_v5
.\train.bat -Dataset .\fsl-dataset.json
```

The first run creates the private Python environment and installs packages as needed; later runs skip installation when the required versions are already present. The SignAR standard validation profile uses five signers, at least 10 NO SIGN examples, and signer-held-out evaluation. Live-camera samples are optional and reported as coverage context; they do not block training or model transfer. The local dataset and generated candidates are excluded from GitHub. Use `-TestMode -Epochs 1` only to mark output as a pipeline-test artifact.

Review the candidate evaluation before transferring it. Use **Transfer candidate to interpreter** in the local collection page to validate and copy eligible TensorFlow.js model files into `SignLanguageAR/model/`; then reload the decoder at `http://localhost:8000/` (or your deployed HTTPS URL). For manual serving, run `python -m http.server 8000 --directory SignLanguageAR` from the workspace root. See the [training guide](./SignAR_FSL_Training_Tools_Video_Import_v5/TRAINING_GUIDE.md) for the validation thresholds and limitations.

## Shared model contract

The browser decoder and training toolkit share the `signar-fsl-holistic-v4` feature schema. The interpreter also retains a reader for the installed `signar-fsl-hand-v3` model while new holistic data are collected:

- 422 values per frame: 71 per hand (21 wrist-relative 3D points, five finger angles, and wrist placement), 33 body landmarks with visibility, 48 normalized face points, and four presence flags.
- Thirty ordered frames sampled at 15 FPS preserve dynamic movement for the temporal model.
- Missing cues have an explicit sentinel and presence flags, so one-hand, face, body, and coordinated two-hand samples share one tensor contract.
- Existing 68-value and legacy 456-value records are migrated during export. Migrated hand-only samples keep their available hand cues and mark unavailable regions as missing.
- Vocabulary order and per-label unit, motion, and display metadata come from the model manifest.
- Online segmentation confirms a label across prediction windows, closes it at a stable class change or 450 ms pause, and permits repeated labels after a pause. A configured phrase label still represents a whole phrase sign.

### SignAR standard validation profile

| Check | Requirement |
|---|---:|
| Mean signer-held-out accuracy | At least 90% |
| Recall for every non-background sign label | At least 80% |
| NO SIGN false-accept rate | At most 10% |
| Stress checks | At least 90% correct or rejected |
| Signer coverage | Five signers per label |
| NO SIGN data | At least 10 samples and 1% to 25% of the dataset |

The gates check signer-held-out accuracy, label recall, NO SIGN false accepts, signer coverage, and stress cases. NO SIGN recall is reported but is not a separate acceptance gate. Metrics describe the evaluated dataset and may differ across signers, cameras, and environments.

Imported samples may provide the five-signer coverage for each label. Live-camera samples are optional; the reports show which labels have camera examples, and their absence does not affect eligibility. Imported-video evaluation alone does not establish live-camera performance.

### Current model and data

The installed hand-v3 weights recognize the current fingerspelling vocabulary. They were evaluated on 992 sample windows: 969 imported-video samples and 23 camera samples. Every label has five-signer coverage; the camera samples are all NO SIGN, so live-camera recognition of the sign labels is unverified. The five held-out folds average 97.85% accuracy. NO SIGN recall is 99.06%, the NO SIGN false-accept rate is 0.47%, and NO SIGN is 21.37% of the dataset. See `SignLanguageAR/model/evaluation.json` for the report and source counts. Imported samples dominate this evaluation, so these figures do not guarantee performance on new camera sessions or signers. Training and deployment tooling now accepts holistic-v4; new labels and holistic cues become recognized after a matching candidate has been trained and transferred.

## Data and privacy

Raw recordings and exported participant datasets are private and are not included in this repository. Keep them in a private local location, use signer aliases rather than identifying information, and do not copy source videos or raw datasets into GitHub. The collector stores recordings locally in the browser until a user exports them; it does not upload data.

Dataset reports are snapshots from a particular validation run, not the live source of truth. Re-run validation against the exact dataset you intend to train.

## UI design consistency

Every UI update should prioritize clarity and consistency. Keep the decoder and training collector recognizably part of the same product by reusing their blue palette, light/dark theme behavior, Segoe UI body text, Georgia display headings, and shared corner-radius scale. The applications may keep layouts tailored to their distinct workflows; consistency means matching their visual language and interaction patterns, not forcing identical screens. Prefer clear hierarchy, scannable content, and restrained surfaces over decorative effects. Keep icons only when they clarify an action or status.

The design system uses a restrained blue palette (including `#286aa5`), coordinated light and dark surfaces, 8px control / 12px surface / 16px camera-stage radii, Segoe UI body text, and Georgia display headings. Keep both apps visually consistent when changing shared design elements. Styles live in [SignLanguageAR/src/ui/style.css](./SignLanguageAR/src/ui/style.css) and [SignAR_FSL_Training_Tools_Video_Import_v5/src/ui/collect-theme.css](./SignAR_FSL_Training_Tools_Video_Import_v5/src/ui/collect-theme.css).
