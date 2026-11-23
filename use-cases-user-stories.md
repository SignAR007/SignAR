# Use Cases and User Stories


## Primary users

- **FSL learner or guided learner:** wants to practice and inspect supported fingerspelling, word, and phrase signs using a camera-based system.
- **Service operator:** needs to explain supported recognition, confidence, and translation limits while helping users troubleshoot the system.
- **Data collector/model reviewer:** works in the private toolkit to gather consented examples, verify labels, evaluate a candidate, and decide whether it is appropriate to promote.

## Use cases

**UC-1: Start an FSL recognition session.** The learner opens the public decoder over localhost or HTTPS and selects Start Camera. The browser requests permission. If permission is denied or no compatible camera is available, the application explains the state and does not claim recognition is active.

**UC-2: Recognize supported signs in a live stream.** The interpreter tracks both hands, face, and body cues and builds an ordered holistic-v4 sequence. It smooths model predictions and segments recognized labels when a different sign becomes stable or a pause marks a boundary. Accepted labels and configured meanings appear in a two-dimensional card over the camera preview. It continues decoding from the current landmark track when multiple people are visible. The installed hand-v3 model recognizes the current fingerspelling set; additional word, phrase, and cue-dependent signs become recognizable after a compatible holistic-v4 candidate is trained, evaluated, and transferred. Unsupported or uncertain patterns remain in a waiting/reject state.

**UC-3: Build text from recognized units.** The learner performs letters one at a time, or performs supported word and phrase signs. Fingerspelled letters join together and pauses separate spelled words; each word or phrase sign is appended as a complete display-text unit. A committed sign is not repeated continuously while held; the learner releases or changes it to permit another commit. The user can review or clear the current output.

**UC-4: Collect a consented training example.** An authorized team member starts the local studio, uses a signer alias, records consent and a review note, selects a verified label, declares its unit type and static/dynamic motion, optionally adds display text, and captures or imports one complete sample. Dynamic samples preserve the complete movement. The collector stores data locally; private raw media and exports are not pushed to public repositories.

**UC-5: Evaluate and promote a model candidate.** A reviewer checks dataset readiness, metadata consistency, label/signer coverage, held-out signer results, per-class recall, NO SIGN false accepts, and stress checks. Optional live-camera coverage is shown as evaluation context; missing camera samples do not block transfer. The reviewer confirms sign forms, then transfers an eligible and compatible candidate to the public model directory. Otherwise, the candidate remains private and the blockers are documented.

## User stories and acceptance criteria

| ID | User story | Acceptance criteria |
|---|---|---|
| US-1 | As a learner, I want to start and stop camera recognition so that I control when camera input is active. | Permission is requested by an explicit action; status reflects off/loading/active/error; stopping ends active capture. |
| US-2 | As a learner, I want predictions only after a sign sequence is stable so transient tracking noise is less likely to become output. | No label is committed from one frame; low-confidence or unstable input remains waiting/rejected; each committed sign has release/repeat behavior. |
| US-3 | As a learner, I want fingerspelling, word signs, and phrase signs to produce readable text. | Manifest metadata identifies the unit; letters join, word/phrase display text is appended as a token, and users can review, delete, copy, or clear output. |
| US-4 | As a learner, I want dynamic signs to be recognized from their movement. | Complete ordered sequences reach the model; dynamic examples are not flattened by static-hold augmentation; tracking gaps use the documented interpolation/reset behavior. |
| US-5 | As a user, I want recognized signs segmented into readable units. | The stream closes units at stable label changes or pauses; phrase labels remain trained phrase signs and the decoder does not infer FSL grammar. |
| US-6 | As a data collector, I want samples tied to consent and reviewed sign metadata so that training data are traceable without exposing identity. | A stable alias, consent, reviewer note, label, unit type, motion type, and optional display text are saved; conflicting metadata blocks training. |
| US-7 | As a model reviewer, I want signer-held-out and stress-test results before promotion so that evaluation is not based only on training examples. | Candidate evaluation reports coverage and key metrics; promotion is blocked or deferred when required gates fail. |
| US-8 | As a participant, I want my recordings kept private so that sample collection does not silently publish my data. | Raw recordings and exported participant data remain local/private and are excluded from public GitHub artifacts. |

## Main alternate flows

- Camera permission denied: provide a recoverable explanation; do not start prediction.
- No hand or tracking gap: show waiting status, interpolate only within the configured short gap, and reset after a prolonged gap.
- Multiple people detected: continue decoding from the current holistic landmark track and indicate that separate per-person transcripts are not produced.
- Model missing or schema-incompatible: show an error and prevent inference; do not load a mismatched vocabulary/model pair.
- Candidate below a required gate: retain it in the private toolkit, show blockers, and collect or review more data before reconsideration.

## Current scope boundary

The installed hand-v3 weights cover the current fingerspelling classes. The holistic-v4 collector and trainer support reviewed word, phrase, static, and dynamic classes. The streaming decoder segments recognized model labels continuously; it does not translate arbitrary sentences or infer FSL grammar. The current browser app is installable but is not a native mobile app. Its camera result card is a two-dimensional overlay, not spatially anchored 3D AR. A standalone sign chart and structured lesson curriculum are not included in this release.

The current holistic-v4 schema captures both hands, face, and body cues. The installed hand-v3 model remains hand-only, so signs that depend on additional cues require a trained and transferred holistic-v4 model.
