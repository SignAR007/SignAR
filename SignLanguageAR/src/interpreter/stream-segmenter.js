/* Online unit boundaries for continuous runs of model predictions. */
(() => {
  "use strict";

  function create(options = {}) {
    const pauseMs = options.pauseMs ?? 450;
    const transitionWindows = options.transitionWindows ?? 3;
    const minimumWindows = options.minimumWindows ?? 3;
    let active = null;
    let pending = null;
    let recent = [];

    function finish(reason, endTime) {
      if (!active || active.acceptedWindows < minimumWindows) {
        active = null;
        return null;
      }
      const event = {
        label: active.label,
        confidence: active.confidenceCount ? active.confidenceTotal / active.confidenceCount : 0,
        startedAt: active.startedAt,
        endedAt: Math.max(active.startedAt, endTime),
        durationMs: Math.max(0, endTime - active.startedAt),
        acceptedWindows: active.acceptedWindows,
        boundary: reason,
      };
      recent.push(event);
      if (recent.length > 12) recent.shift();
      active = null;
      pending = null;
      return event;
    }

    function startFromPending() {
      if (!pending || pending.windows < minimumWindows) return false;
      active = {
        label: pending.label,
        startedAt: pending.startedAt,
        lastAcceptedAt: pending.lastAcceptedAt,
        acceptedWindows: pending.windows,
        confidenceTotal: pending.confidenceTotal,
        confidenceCount: pending.windows,
      };
      pending = null;
      return true;
    }

    function update(label, confidence, now = performance.now()) {
      const cleanLabel = typeof label === "string" ? label.trim() : "";
      if (!cleanLabel || !Number.isFinite(confidence) || confidence < 0 || confidence > 100) {
        pending = null;
        const event = active && now - active.lastAcceptedAt >= pauseMs
          ? finish("pause", active.lastAcceptedAt)
          : null;
        return { event, events: event ? [event] : [], activeLabel: active?.label || "", pendingLabel: "" };
      }

      const events = [];
      if (active && now - active.lastAcceptedAt >= pauseMs) {
        const ended = finish("pause", active.lastAcceptedAt);
        if (ended) events.push(ended);
      }

      if (!active) {
        if (pending?.label === cleanLabel) {
          pending.windows++;
          pending.lastAcceptedAt = now;
          pending.confidenceTotal += confidence;
        } else {
          pending = { label: cleanLabel, startedAt: now, lastAcceptedAt: now, windows: 1, confidenceTotal: confidence };
        }
        startFromPending();
      } else if (active.label === cleanLabel) {
        active.lastAcceptedAt = now;
        active.acceptedWindows++;
        active.confidenceTotal += confidence;
        active.confidenceCount++;
        pending = null;
      } else {
        if (pending?.label === cleanLabel) {
          pending.windows++;
          pending.lastAcceptedAt = now;
          pending.confidenceTotal += confidence;
        } else {
          pending = { label: cleanLabel, startedAt: now, lastAcceptedAt: now, windows: 1, confidenceTotal: confidence };
        }
        if (pending.windows >= transitionWindows) {
          const next = pending;
          const previous = finish("class-change", pending.startedAt);
          if (previous) events.push(previous);
          pending = next;
          startFromPending();
        }
      }
      return {
        event: events.at(-1) || null,
        events,
        activeLabel: active?.label || "",
        pendingLabel: pending?.label || "",
        activeWindows: active?.acceptedWindows || 0,
      };
    }

    function tick(now = performance.now()) {
      const event = active && now - active.lastAcceptedAt >= pauseMs
        ? finish("pause", active.lastAcceptedAt)
        : null;
      if (!active && pending && now - pending.lastAcceptedAt >= pauseMs) pending = null;
      return { event, events: event ? [event] : [], activeLabel: active?.label || "", pendingLabel: pending?.label || "" };
    }

    function clear() {
      active = null;
      pending = null;
      recent = [];
    }

    function flush(now = performance.now(), reason = "flush") {
      const event = finish(reason, now);
      return { event, events: event ? [event] : [], activeLabel: "", pendingLabel: "" };
    }

    return Object.freeze({
      update,
      tick,
      clear,
      flush,
      get activeLabel() { return active?.label || ""; },
      get pendingLabel() { return pending?.label || ""; },
      get recent() { return recent.map((event) => ({ ...event })); },
    });
  }

  window.SignARStreamSegmenter = Object.freeze({ create });
})();
