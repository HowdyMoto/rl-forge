# Energy Challenge Environment Spec

## Concept

A locomotion agent with a finite energy battery must complete an obstacle course
before running out. The agent sees its remaining energy and the terrain ahead,
and must learn *when* to spend energy (sprinting, jumping, climbing) vs. conserve
it (efficient walking, coasting downhill). The battery IS the constraint — there
is no per-step energy penalty in the reward.

This is novel: no published locomotion RL work uses a depleting energy reservoir
as a terminal constraint. The closest work (EIPO, CoT-based rewards) penalizes
energy *rate*, not cumulative budget. The behaviors that emerge — pacing,
gait switching, strategic bursting — should be visually striking.

---

## Architecture

### Where It Fits

This is a new environment mode for `UnifiedRapierEnv`, not a new env class.
It reuses the existing biped character, PD control, and terrain system.

```
registry.js        → register 'EnergyChallenge-v1'
unifiedEnv.js      → add energy tracking + new reward path + extended obs
terrain.js         → add course generator (fixed-length, section-typed)
characters/biped.js → reuse as-is (no changes)
```

### New Files

```
src/env/course.js  → obstacle course generator (replaces open-ended terrain)
```

---

## Environment Definition

### Constructor Options

```js
// In registry.js
registerEnv('EnergyChallenge-v1', {
  factory: (spec, opts) => new UnifiedRapierEnv(BIPED, {
    controlMode: 'pd',
    terrain: true,
    physicsHz: 240,
    policyHz: 30,
    maxSteps: 3000,           // generous — battery is the real limit
    energyBudget: 800,        // joules, tunable per difficulty
    courseLength: 30,          // meters, shorter than open-ended terrain
    courseDifficulty: 0.5,    // 0-1, controls obstacle density/severity
    courseSeed: null,          // null = random per episode
    ...opts,
  }),
  maxEpisodeSteps: 3000,
  rewardThreshold: 100,       // TBD after tuning
})
```

### Energy System

#### Energy Computation

Energy is computed as mechanical work at each physics substep:

```js
// Per substep, per joint:
power_j = |torque_j * angularVelocity_j|    // watts (instantaneous)
energy_j = power_j * physicsDt               // joules (per substep)

// Total per policy step (summed across all substeps and joints):
stepEnergy = sum over substeps and joints of energy_j
```

This means:
- Standing still with PD holding position = low energy (small corrections)
- Walking efficiently = moderate energy (~2-5 J/step)
- Sprinting = high energy (~8-15 J/step)
- Jumping/climbing = very high energy (20+ J/step burst)

#### Energy State

```js
// Added to UnifiedRapierEnv instance state:
this._energyBudget = opts.energyBudget ?? 800     // max joules
this._energyRemaining = this._energyBudget         // current joules
this._energyUsedThisStep = 0                        // for info/rendering
this._courseComplete = false                         // reached finish line
```

#### Energy Tracking in Step

Energy is accumulated during the physics substep loop. In `_stepPD()`:

```js
_stepPD(actions) {
  const pd = this._pdCache
  if (!pd) return

  // Map actions to target angles (existing code, unchanged)
  for (let i = 0; i < pd.joints.length; i++) { /* ... existing ... */ }

  let stepEnergy = 0

  // PD control each substep
  for (let sub = 0; sub < this.substeps; sub++) {
    for (let i = 0; i < pd.joints.length; i++) {
      const bA = pd.bodyAs[i]
      const bB = pd.bodyBs[i]
      const currentAngle = bB.rotation() - bA.rotation()
      const currentAngVel = bB.angvel() - bA.angvel()

      let torque = pd.kps[i] * (this._targetAngles[i] - currentAngle)
                 - pd.kds[i] * currentAngVel
      torque = Math.max(-pd.maxTorques[i], Math.min(pd.maxTorques[i], torque))

      bB.addTorque(torque, true)
      bA.addTorque(-torque, true)

      // >>> NEW: accumulate energy
      if (this._energyBudget !== undefined) {
        stepEnergy += Math.abs(torque * currentAngVel) * this.physicsDt
      }
    }
    this.world.step()
  }

  // >>> NEW: deduct from battery
  if (this._energyBudget !== undefined) {
    this._energyUsedThisStep = stepEnergy
    this._energyRemaining = Math.max(0, this._energyRemaining - stepEnergy)
  }
}
```

---

## Observation Space

Extend the existing terrain observation with 2 new dimensions:

```
Standard terrain obs (55 dims):
  [0-4]     Torso: height, angle, vx, vy, angvel
  [5-12]    Joints: 4 joints x (angle, angvel)
  [13-14]   Foot contacts: left, right
  [15-54]   Terrain heightfield: 40 samples

Energy challenge additions (2 dims):
  [55]      Energy remaining (normalized 0-1: energyRemaining / energyBudget)
  [56]      Course progress (normalized 0-1: torsoX / courseLength)

Total: 57 dimensions
```

The agent MUST see its battery level to plan ahead. Without it, the problem
is a POMDP and the agent can't learn pacing.

Course progress tells the agent how much course remains — critical for
deciding whether to conserve or spend.

### Implementation

In `_getObs()`, after the existing terrain heightfield push:

```js
if (this._energyBudget !== undefined) {
  obs.push(this._energyRemaining / this._energyBudget)       // [0, 1]
  obs.push(Math.min(1.0, torsoPos.x / this.courseLength))     // [0, 1]
}
```

In `computeDerivedFields()`, when energyBudget is present:

```js
obsSize += 2  // energy + progress
```

---

## Reward Function

The reward is fundamentally different from standard locomotion. There is
**no energy penalty** — the battery is the penalty. Spending energy freely
is fine as long as you finish.

```js
_computeReward(forwardVel, actions, healthy, done, timedOut, torsoPos) {
  if (this._energyBudget === undefined) {
    return /* existing reward logic */
  }

  let reward = 0

  if (healthy || timedOut) {
    // 1. Progress reward: reward forward movement
    //    Use delta-x, not velocity, to avoid rewarding oscillation
    const deltaX = torsoPos.x - this._prevTorsoX
    reward += 2.0 * Math.max(0, deltaX)

    // 2. Alive bonus: small constant for not falling
    reward += 0.5

    // 3. NO energy penalty — the battery IS the constraint
    // 4. NO control cost — let the agent move however it needs to
  }

  // 5. Course completion bonus
  if (this._courseComplete) {
    // Big reward for finishing + bonus for remaining energy
    reward += 50.0
    reward += 20.0 * (this._energyRemaining / this._energyBudget)
  }

  // 6. Termination: no extra penalty (losing remaining course progress
  //    is punishment enough)

  return reward
}
```

### Reward Design Rationale

| Term | Purpose |
|------|---------|
| `2.0 * deltaX` | Incentivizes forward progress. Using delta-x (not velocity) prevents reward from oscillating back and forth. Only positive delta counts — no reward for going backward. |
| `0.5 alive` | Small incentive to stay upright, but not so large that standing still is optimal. Must be small relative to progress reward. |
| `50.0 completion` | Large one-time bonus for reaching the finish line. Must dominate episodic return to make finishing the primary objective. |
| `20.0 * remaining_energy` | Bonus scaled by leftover battery. An agent that finishes with 80% battery gets 16.0 extra. Incentivizes efficiency *given* completion, not instead of it. |
| No energy penalty | The battery enforces efficiency. Wasteful agents die before finishing. No reward shaping needed. |

### Expected Return Ranges

- **Fails immediately (falls):** ~0-5
- **Walks 10m then runs out of energy:** ~20-25
- **Finishes course inefficiently (barely):** ~100-120
- **Finishes course efficiently (energy to spare):** ~120-140

---

## Termination Conditions

```js
_checkTermination(torsoPos, torsoRot) {
  // ... existing health checks (height, angle) ...

  // >>> NEW: energy depletion
  const energyDepleted = (this._energyBudget !== undefined)
                      && (this._energyRemaining <= 0)

  // >>> NEW: course completion
  if (this._energyBudget !== undefined && torsoPos.x >= this.courseLength) {
    this._courseComplete = true
  }

  const done = !healthy || timedOut || energyDepleted || this._courseComplete
  return { healthy, done, timedOut, energyDepleted, courseComplete: this._courseComplete }
}
```

The agent's episode ends when ANY of:
1. It falls (existing health check)
2. It runs out of energy (battery = 0)
3. It reaches the finish line (success!)
4. Max steps exceeded (safety timeout — should be generous)

---

## Obstacle Course Generator

### New File: `src/env/course.js`

Unlike the existing open-ended terrain, the course is a **fixed-length sequence
of designed sections** that test different energy strategies.

```js
/**
 * Course section types and their energy implications:
 *
 *   flat        — efficient walking, lowest energy cost
 *   gentle_hill — moderate energy, tests uphill efficiency
 *   steep_hill  — high energy burst required
 *   downhill    — opportunity to coast (near-zero energy)
 *   gap         — requires jump (one-time high energy burst)
 *   stairs      — sustained moderate-high energy
 *   narrow      — requires slow, precise movement (low energy but slow progress)
 *   sprint_zone — long flat section, agent chooses speed/energy tradeoff
 */
```

### Course Structure

Every course follows a pattern that forces varied strategies:

```
[spawn_flat] [mixed_sections...] [finish_flat]
```

Course parameters:
- `courseLength`: total length in meters (default 30)
- `courseDifficulty`: 0-1, controls section severity
- `courseSeed`: for reproducibility (null = random)

### Section Selection

Difficulty controls which sections appear and their severity:

```
Easy   (0.0-0.3): flat, gentle_hill, downhill, sprint_zone
Medium (0.3-0.7): + steep_hill, stairs, gap (small)
Hard   (0.7-1.0): + gap (wide), steep_hill (longer), narrow
```

### Course Length Tuning

The `energyBudget` and `courseLength` must be tuned together:

```
Budget guideline:
  efficient_walk_cost ≈ 3 J/m  (on flat)
  minimum_budget = courseLength * efficient_walk_cost * 1.5  (50% headroom)

  For a 30m course: minimum ~135 J
  Default 800 J gives substantial headroom for hills/jumps,
  but wasteful gaits will still run out.

  Tuning target: an efficient agent finishes with 20-40% battery remaining.
  A "sprint everything" agent runs out at ~60-70% course completion.
```

---

## Info Dict Extensions

The `step()` return's `info` object gets new fields:

```js
info: {
  // ... existing fields ...
  forwardVel,
  healthy,
  stepCount: this.stepCount,
  distance: torsoPos.x,
  maxDistance: this._maxTorsoX,

  // >>> NEW
  energyRemaining: this._energyRemaining,
  energyUsedThisStep: this._energyUsedThisStep,
  energyFraction: this._energyRemaining / this._energyBudget,
  courseProgress: Math.min(1.0, torsoPos.x / this.courseLength),
  courseComplete: this._courseComplete,
  energyDepleted: this._energyRemaining <= 0,
}
```

---

## Render Snapshot Extensions

Add to `getRenderSnapshot()` for the UI to display:

```js
if (this._energyBudget !== undefined) {
  snapshot._energy = {
    remaining: this._energyRemaining,
    budget: this._energyBudget,
    fraction: this._energyRemaining / this._energyBudget,
    usedThisStep: this._energyUsedThisStep,
  }
  snapshot._course = {
    length: this.courseLength,
    progress: Math.min(1.0, torsoPos.x / this.courseLength),
    complete: this._courseComplete,
  }
}
```

---

## UI Visualization

### Energy Bar

Render a horizontal bar in the simulation canvas:

```
Position: top-left of viewport, fixed to camera
Size: 200px wide, 16px tall
Color: gradient from green (>50%) → yellow (20-50%) → red (<20%)
Label: "ENERGY: 73%" or "423 / 800 J"
```

When energy is being consumed rapidly (e.g., during a jump), flash the bar
or show a drain animation.

### Course Progress

```
Position: top-center of viewport
Style: thin horizontal progress bar spanning viewport width
Markers: vertical ticks at 25%, 50%, 75%
Finish line: visible flag/marker in the world at courseLength
```

### Metrics Panel (Training Mode)

Add to the existing metrics charts:
- **Energy at completion** (scatter plot per episode): shows if the agent is
  learning to be more efficient over training
- **Course completion rate** (rolling %): fraction of episodes where agent
  reaches finish
- **Average energy remaining at finish** (for completed episodes only)

---

## Training Considerations

### Hyperparameter Adjustments

The energy challenge has different dynamics than open-ended locomotion:

```js
// Suggested PPO overrides for EnergyChallenge-v1
{
  stepsPerUpdate: 4096,      // longer rollouts capture full episodes
  gamma: 0.995,              // long horizon — agent must plan ahead
  maxSteps: 3000,            // generous time limit
  hiddenSizes: [128, 128],   // slightly larger net for planning
}
```

Higher gamma is critical — with gamma=0.99, rewards 300 steps in the future
are discounted to ~5% of face value. The completion bonus at the end of a
3000-step episode would be nearly invisible. Gamma=0.995 gives ~22% at step
300, which is enough signal.

### Curriculum Strategy

Start easy, increase difficulty:

1. **Phase 1** (episodes 0-200): Short course (15m), large budget (1200J),
   easy terrain. Agent learns basic locomotion + energy awareness.

2. **Phase 2** (episodes 200-500): Medium course (25m), moderate budget (800J),
   medium terrain. Agent learns efficiency.

3. **Phase 3** (episodes 500+): Full course (30m), tight budget (600J),
   hard terrain. Agent learns strategic pacing.

Implement via the existing curriculum pattern in `reset()`:

```js
if (this._energyBudget !== undefined) {
  const phase = Math.min(1.0, this._episodeCount / 500)
  this.courseLength = 15 + phase * 15                    // 15 → 30m
  this._energyBudget = 1200 - phase * 600                // 1200 → 600J
  this._effectiveDifficulty = Math.min(1.0, phase)       // 0 → 1
}
```

### What Success Looks Like

A well-trained agent should visibly exhibit:

1. **Efficient flat-ground gait** — smooth, minimal wasted motion
2. **Power bursts on hills** — noticeably more vigorous stride uphill
3. **Coasting downhill** — relaxed, minimal actuator engagement
4. **Jump commitment** — explosive single-step burst to clear gaps
5. **Pacing at low battery** — visibly more conservative gait when battery
   drops below ~30%
6. **Speed variation** — faster on easy sections, slower on hard ones

### Failure Modes to Watch For

- **Standing still:** If alive bonus is too high relative to progress reward,
  the agent may learn to stand still and collect alive bonus. Fix: reduce
  alive bonus or remove it.
- **Suicide at low battery:** If the agent learns that running out of energy
  gives a worse return than falling early, it may intentionally fall when
  battery is low. Fix: ensure the reward for partial progress is always
  better than early termination.
- **Ignoring battery:** If the budget is too generous, the agent never runs
  out and doesn't learn efficiency. Fix: tighten the budget in curriculum.
- **Oscillating for reward:** If using velocity reward instead of delta-x,
  agent may rock back and forth. Fix: use `max(0, deltaX)` as specified.

---

## Implementation Order

1. **`src/env/course.js`** — obstacle course generator with section types
2. **`unifiedEnv.js` energy tracking** — add energy state, tracking in
   `_stepPD()`, deduction per substep
3. **`unifiedEnv.js` observation extension** — add 2 dims (energy, progress)
4. **`unifiedEnv.js` reward function** — new reward path when energyBudget
   is set
5. **`unifiedEnv.js` termination** — energy depletion + course completion
6. **`unifiedEnv.js` info + snapshot** — expose energy/course data
7. **`registry.js`** — register `EnergyChallenge-v1`
8. **UI: energy bar + progress bar** — render overlay in simulation canvas
9. **UI: training metrics** — add energy/completion charts
10. **Curriculum tuning** — adjust budget/length/difficulty ramp
11. **Hyperparameter defaults** — gamma, rollout length, network size

Steps 1-7 are the environment. Steps 8-9 are visualization. Steps 10-11
are tuning. The environment is testable with headless training after step 7.

---

## Open Questions

- **Should energy regenerate on downhill sections?** Biological analogy:
  muscles don't regenerate, but eccentric loading is cheaper. Could add a
  small energy recoup when joints do negative work (torque opposes motion).
  Start without this, add later if pacing behavior is insufficient.

- **Multiple battery sizes as a training distribution?** Randomizing the
  budget per episode (e.g., uniform 500-1000J) could produce a more robust
  policy that adapts to any budget. Trade-off: harder to learn initially.

- **Energy-per-joint visualization?** Showing which joints are consuming
  energy (hip vs. knee, left vs. right) could be informative for debugging
  and visually interesting. Low priority but cool.

- **Leaderboard / sharing?** "Complete this course with the least energy"
  is a natural competitive metric. The existing model export system could
  support this.
