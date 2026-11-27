# Shared Feedback API

The feedback board uses a public Cloudflare Worker API backed by D1. Anyone using the app can read and submit entries. Messages and optional aliases are not private; do not submit sensitive or identifying information. The API has no anonymous edit or delete endpoint.

## Deploy

1. From `SignLanguageAR/`, authenticate Wrangler with `npm exec -- wrangler login`.
2. Create the D1 database with `npm exec -- wrangler d1 create signar-feedback`.
3. Copy the returned database ID into `feedback-api/wrangler.toml`, replacing the all-zero placeholder.
4. In that file, replace the `YOUR_GITHUB_PAGES_ACCOUNT` origin with the exact GitHub Pages origin, such as `https://account.github.io`. Keep the two localhost origins for development.
5. Apply the schema with `npm run feedback:migrate:remote`.
6. Deploy with `npm run feedback:deploy` and copy the resulting `workers.dev` URL.
7. Set `productionApiUrl` in `src/ui/feedback-config.js` to that URL and publish the static app through the existing GitHub Pages workflow.

The Cloudflare Workers Free plan has daily request and D1 usage limits. The Worker URL and D1 database ID are identifiers, not secrets. Never put a Cloudflare account token in browser code or commit it to the repository.

## Local development

Run `npm run feedback:migrate:local`, then start the Worker with `npm run feedback:dev`. Serve the app at `http://localhost:8123`; the local feedback config points to `http://localhost:8787`. Local D1 state is stored under `.wrangler/` and is ignored by Git.
