# ── MAGE Frontend Dockerfile ──────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (deps):    Install only production dependencies.
#   Stage 2 (builder): Build the Next.js application.
#   Stage 3 (runner):  Serve the pre-built app with minimal footprint.

# ── Stage 1: Dependencies ─────────────────────────────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /app

RUN apk add --no-cache libc6-compat

COPY package.json package-lock.json* ./
RUN npm ci

# ── Stage 2: Builder ──────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

ENV NEXT_TELEMETRY_DISABLED=1

# NEXT_PUBLIC_* vars are inlined into the client bundle at build time, not
# read at container startup — a runtime-only env var here would leave every
# browser request pointed at the api.ts fallback (localhost:8000), unreachable
# from outside the container. Must be declared as a build ARG and exported to
# the build's env so `npm run build` actually sees it. Railway (and `docker
# build --build-arg`) forwards a matching service variable automatically once
# the ARG exists — nothing else to configure on the platform side.
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# ── Stage 3: Runner ───────────────────────────────────────────────────────────
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Note: curl is not installed in the alpine runner, so use node itself.
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD node -e "require('http').get('http://localhost:3000',r=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))" || exit 1

CMD ["node", "server.js"]
