# SignAR holistic collection and training guide

## Collection

Use the local studio at `start_training_studio.bat`. For each sample, enter a stable signer alias, verify the exact FSL form with a qualified reviewer or reference, record consent, and add a short review note. Camera examples are optional; imported videos and images can provide signer and label coverage.

For every label, declare whether it is a fingerspelling letter, word sign, whole phrase sign, or background, whether it is static or dynamic, and any display text to append. Record one complete sign per sample. Dynamic signs need video showing the full movement; a still image is useful only for static examples. Frame both hands, face, and upper body when the sign depends on them.

Include at least 10 NO SIGN samples (1% to 25% of the dataset), with varied neutral posture and non-sign movement. Frames without detected cues remain in the sample with missing-region flags, preserving sequence timing. Do not infer a sign from its filename alone. Keep uncertain clips out until their form is reviewed, and keep raw recordings and exported data private.

## Feature schema

Holistic-v4 uses 30 frames at 15 FPS and 422 values per frame. The layout contains both hands (71 values each), body pose (33 landmarks × 4 values), face cues (48 landmarks × 3 values), and four region-presence flags. The face points cover eyes, eyebrows, nose, and mouth. Hand placement is normalized to the body; hand shape is wrist-relative and palm-scaled. Missing regions use `-3` plus the presence flags.

The exporter migrates older 68-value hand-v3 and legacy 456-value samples into the new layout. Older hand-only samples keep their hand data while unavailable face and body cues remain marked missing. Newly collected records and exported training datasets use holistic-v4.

## Training and deployment

Select **Train candidate** in the local studio or run:

```powershell
.\.venv\Scripts\python.exe training\train_hand_model.py --dataset fsl-dataset.json --output-dir trained-model
```

Every training run fits the full eligible vocabulary. It uses signer-held-out evaluation and reports recall for each sign label, confusion matrix, expected calibration error, NO SIGN false-accept rate, optional live-camera counts, and stress checks. Augmentation varies time, spatial scale and rotation, landmark noise, hand slots, and occasional missing frames while retaining temporal order for dynamic signs.

The SignAR standard validation profile requires at least 90% mean leave-one-signer-out accuracy, 80% recall for each non-background sign label, at most 10% NO SIGN false accepts, at least 90% correct-or-reject on every stress check, five signers per label, and at least 10 NO SIGN samples (1% to 25% of the dataset). Live-camera samples are optional and do not affect readiness or transfer eligibility. Evaluation metrics describe the collected dataset and may differ across signers, cameras, and environments.

The live interpreter segments recognized labels when a new sign becomes stable or after a 450 ms pause. It can append the configured display text and separate word/phrase units. This stream boundary logic does not translate signs that the model has not learned or infer FSL grammar. A phrase label itself represents one complete phrase sign.

Review `trained-model/evaluation.json` and verify the vocabulary before transferring a candidate. The installed decoder weights still use hand-v3 until a holistic-v4 candidate passes evaluation and is transferred.
