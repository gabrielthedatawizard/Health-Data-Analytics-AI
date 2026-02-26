# Vercel Deployment

This repository is configured for Vercel with the frontend in `app/`.

## Configuration Included

- `vercel.json` at repo root:
  - `installCommand`: `cd app && npm ci --include=dev`
  - `buildCommand`: `cd app && npm run build`
  - `outputDirectory`: `app/dist`
  - SPA rewrite to `index.html`
- `app/vercel.json` (fallback when Vercel Project Root Directory is set to `app/`):
  - `installCommand`: `npm ci --include=dev`
  - `buildCommand`: `npm run build`
  - `outputDirectory`: `dist`
  - SPA rewrite to `index.html`
- `.nvmrc` set to `22.12.0` (compatible with Vite 7)
- `.vercelignore` excludes non-runtime assets and large local reference files
- `app/package.json` includes Node engine `>=20.19.0`

## Deploy From Vercel Dashboard

1. Import this GitHub repository into Vercel.
2. Use either project root:
   - Repository root (uses root `vercel.json`)
   - `app/` root (uses `app/vercel.json`)
3. In Vercel Project Settings, clear any manual Build/Install command overrides so config files are honored.
4. Deploy.

## Deploy With Vercel CLI

```bash
npm i -g vercel
vercel
vercel --prod
```
