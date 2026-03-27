# RL Forge

A browser-based 2D robotics sandbox for designing creatures, training them with reinforcement learning, and watching them learn to walk, balance, and navigate. Everything runs client-side — no server, no GPU rental, no Python.

## What it does

**Train Mode** — Pick an environment, tune PPO hyperparameters, and watch an agent learn in real time. Training runs in a Web Worker with GPU-accelerated neural networks via WebGPU. Live charts show reward curves, loss, and episode metrics as the agent improves.

**Test Mode** — A physics lab for verifying bodies before training. Load any character, test joints and motors, drag bodies with the mouse, and inspect physical properties. Useful for debugging custom morphologies.

**Build Mode** — Visual creature designer (coming soon).

**Inference Mode** — Load trained policies and compare runs (coming soon).

## Environments

| Environment | Description | Obs | Actions |
|---|---|---|---|
| CartPole | Balance an inverted pendulum on a sliding cart | 9 | 1 |
| Hopper | Single-legged hopper — learn to jump forward | 10 | 2 |
| Walker2D | Bipedal walker — coordinate two legs | 15 | 4 |
| Biped | Larger humanoid on procedural terrain | 55 | 4 |
| Acrobot | Double pendulum swing-up | 10 | 1 |
| Acrobot (Damped) | Acrobot with joint friction | 10 | 1 |
| Spinner (Constant) | Match a target angular velocity | 9 | 1 |
| Spinner (Max) | Spin as fast as possible | 9 | 1 |
| Red Light Green Light | Stop/go locomotion game | 10 | 1 |

All environments use the same unified physics pipeline backed by [Rapier2D](https://rapier.rs/) WASM.

## Tech stack

- **React** — UI
- **TensorFlow.js** — PPO training (WebGPU backend with WebGL/CPU fallback)
- **Rapier2D** — 2D rigid-body physics via WASM
- **Vite** — Build tooling
- **Recharts** — Training visualizations

Training runs entirely in a Web Worker so the UI stays responsive. The WebGPU backend requests full adapter limits (including storage buffer counts) to avoid pipeline failures on capable hardware.

## Getting started

```bash
npm install
npm run dev
```

Open `http://localhost:5173` in Chrome or Edge (WebGPU support required for GPU acceleration; falls back to WebGL/CPU otherwise).

### Build for deployment

```bash
npm run build
```

Output goes to `dist/`. Uses relative paths and HashRouter, so it works on any static host (GitHub Pages, S3, etc.) without server-side routing.

## Project structure

```
src/
  app/routes/       — Page-level components (Train, Test, Build, Infer)
  components/       — Renderers, charts, creature builder, UI panels
  env/
    characters/     — Character definitions (cartpole, hopper, walker2d, ...)
    unifiedEnv.js   — Unified Rapier-based environment
    vecEnv.js       — Vectorized env wrapper for batched training
  rl/
    ppo.js          — PPO agent, GAE, actor-critic networks
    trainWorker.js  — Web Worker entry point for training loop
```

## License

MIT
