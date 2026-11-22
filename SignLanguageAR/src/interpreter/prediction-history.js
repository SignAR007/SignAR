/* Probability smoothing and release-aware streaming commits. */
(() => {
  "use strict";
  const DEFAULTS = window.SignARPipeline?.config?.recognition || {};

  function createStreaming(options = {}) {
    const alpha = options.alpha ?? DEFAULTS.emaAlpha ?? 0.45;
    const requiredFrames = options.stableFrames ?? DEFAULTS.stableFrames ?? 6;
    const cooldownMs = options.cooldownMs ?? DEFAULTS.commitCooldownMs ?? 300;
    let ema = null;
    let candidateLabel = "";
    let candidateFrames = 0;
    let committedLabel = "";
    let lastCommitAt = -Infinity;
    let recent = [];

    function clear({ release = true } = {}) {
      ema = null;
      candidateLabel = "";
      candidateFrames = 0;
      recent = [];
      if (release) committedLabel = "";
    }

    function release() {
      committedLabel = "";
      candidateLabel = "";
      candidateFrames = 0;
      recent.push({
        label: "NO SIGN",
        accepted: false,
        confidence: 0,
        margin: 0,
      });
      if (recent.length > 10) recent.shift();
    }

    function update(
      scores,
      labels,
      {
        rejectLabel = "NO SIGN",
        minimumConfidencePercent = 70,
        minimumMarginPercent = 18,
        now = performance.now(),
      } = {},
    ) {
      const values =
        scores && typeof scores.length === "number" ? Array.from(scores) : [];
      const sum = values.reduce((total, value) => total + value, 0);
      if (
        !Array.isArray(labels) ||
        labels.length < 2 ||
        labels.length !== values.length ||
        labels.some((label) => typeof label !== "string" || !label.trim()) ||
        new Set(labels).size !== labels.length ||
        values.some(
          (value) => !Number.isFinite(value) || value < 0 || value > 1,
        ) ||
        Math.abs(sum - 1) > 0.02
      )
        throw new Error(
          "Model returned invalid class probabilities or label metadata.",
        );

      if (!ema) ema = values;
      else {
        for (let index = 0; index < values.length; index++)
          ema[index] = alpha * values[index] + (1 - alpha) * ema[index];
        const emaSum = ema.reduce((total, value) => total + value, 0);
        if (emaSum > 0)
          for (let index = 0; index < ema.length; index++) ema[index] /= emaSum;
      }
      const ranked = ema
        .map((score, index) => ({
          label: labels[index],
          score,
          confidence: score * 100,
        }))
        .sort((a, b) => b.score - a.score);
      const best = ranked[0];
      const margin = (best.score - (ranked[1]?.score || 0)) * 100;
      const isReject = best.label.toUpperCase() === rejectLabel.toUpperCase();
      const scorePassed = best.confidence >= minimumConfidencePercent;
      const marginPassed = margin >= minimumMarginPercent;
      const accepted = !isReject && scorePassed && marginPassed;

      if (isReject) {
        release();
      } else if (accepted) {
        if (candidateLabel === best.label) candidateFrames++;
        else {
          candidateLabel = best.label;
          candidateFrames = 1;
        }
      } else {
        candidateLabel = "";
        candidateFrames = 0;
      }

      const stable = accepted && candidateFrames >= requiredFrames;
      const commit =
        stable &&
        committedLabel !== best.label &&
        now - lastCommitAt >= cooldownMs;
      if (commit) {
        committedLabel = best.label;
        lastCommitAt = now;
      }
      recent.push({
        label: best.label,
        accepted,
        confidence: best.confidence,
        margin,
        streak: candidateFrames,
      });
      if (recent.length > 10) recent.shift();
      return {
        ranked,
        best,
        margin,
        scorePassed,
        marginPassed,
        accepted,
        stable,
        commit,
        release: isReject,
        label: best.label,
        confidence: best.confidence,
        streak: candidateFrames,
        requiredFrames,
        committedLabel,
        probabilities: ema.slice(),
        recent: recent.map((item) => ({ ...item })),
      };
    }

    return {
      clear,
      release,
      update,
      get recent() {
        return recent.map((item) => ({ ...item }));
      },
      get length() {
        return recent.length;
      },
      get committedLabel() {
        return committedLabel;
      },
    };
  }

  window.SignARPredictionHistory = Object.freeze({ createStreaming });
})();
