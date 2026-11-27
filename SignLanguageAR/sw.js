const CACHE = "signar-v47";
const CORE = [
  "./",
  "./index.html",
  "./src/ui/style.css",
  "./src/ui/theme.js",
  "./src/ui/feedback-config.js",
  "./src/ui/landmark-renderer.js",
  "./src/core/fsl-pipeline.js",
  "./src/tracker/mediapipe-tracker.js",
  "./src/interpreter/prediction-history.js",
  "./src/interpreter/stream-segmenter.js",
  "./src/interpreter/app.js",
  "./manifest.webmanifest",
  "./icons/icon.svg",
  "./icons/fsl-alphabet-chart.jpg",
  "./model/vocabulary.json",
  "./model/metadata.json",
  "./model/evaluation.json",
  "./model/model-fb5b258b0458fa2e3646.json",
  "./model/weights-7c01bea8c4378b719fe7c64a958209389406346b99cd34a67a63c6c60371b903.bin",
];
self.addEventListener("install", (e) =>
  e.waitUntil(
    caches
      .open(CACHE)
      .then((c) => c.addAll(CORE))
      .then(() => self.skipWaiting()),
  ),
);
self.addEventListener("activate", (e) =>
  e.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k.startsWith("signar-v") && k !== CACHE)
            .map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  ),
);
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;
  e.respondWith(
    fetch(e.request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone();
          caches
            .open(CACHE)
            .then((cache) => cache.put(e.request, copy))
            .catch(() => {});
        }
        return response;
      })
      .catch(async () => {
        const cached = await caches.match(e.request);
        if (cached) return cached;
        if (e.request.mode === "navigate")
          return (await caches.match("./index.html")) || Response.error();
        return new Response("", { status: 503, statusText: "Offline" });
      }),
  );
});
