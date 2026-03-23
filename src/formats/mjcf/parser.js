/**
 * MJCF Parser: MJCF XML string -> BodyDef
 *
 * Parses MJCF (MuJoCo) files into RL-Forge BodyDef format.
 * Supports both RL-Forge's own MJCF variant and standard Gym MJCF files.
 *
 * Handles:
 *   - Nested <body> elements (tree structure -> flat bodies + joints)
 *   - <geom> with type box/capsule/sphere
 *   - <joint type="hinge"> (revolute) and <joint type="slide"> (prismatic)
 *   - <actuator><motor> for torque/force limits
 *   - <default> attribute inheritance (cascading defaults)
 *   - <compiler angle="degree|radian"> for angle unit conversion
 *   - <custom> for rl-forge extensions (restitution, PD gains, etc.)
 *   - Standard Gym MJCF coordinate conventions (XZ plane -> XY plane)
 */

// ─── DOM helpers ───────────────────────────────────────────────────────────

function childrenByTag(el, tag) {
  const result = []
  for (let i = 0; i < el.childNodes.length; i++) {
    const child = el.childNodes[i]
    if (child.nodeType === 1 && child.tagName === tag) {
      result.push(child)
    }
  }
  return result
}

function firstChildByTag(el, tag) {
  for (let i = 0; i < el.childNodes.length; i++) {
    const child = el.childNodes[i]
    if (child.nodeType === 1 && child.tagName === tag) return child
  }
  return null
}

function findByTag(el, tag) {
  const results = el.getElementsByTagName(tag)
  return results.length > 0 ? results[0] : null
}

function findAllByTag(el, tag) {
  return Array.from(el.getElementsByTagName(tag))
}

// ─── Default attribute inheritance ─────────────────────────────────────────

/**
 * Parse <default> section and build a map of defaults by element type.
 * Returns { joint: {...}, geom: {...}, motor: {...} }
 */
function parseDefaults(mujocoEl) {
  const defaults = { joint: {}, geom: {}, motor: {} }
  const defaultEl = findByTag(mujocoEl, 'default')
  if (!defaultEl) return defaults

  const jointEl = firstChildByTag(defaultEl, 'joint')
  if (jointEl) {
    for (const attr of jointEl.attributes) {
      defaults.joint[attr.name] = attr.value
    }
  }

  const geomEl = firstChildByTag(defaultEl, 'geom')
  if (geomEl) {
    for (const attr of geomEl.attributes) {
      defaults.geom[attr.name] = attr.value
    }
  }

  const motorEl = firstChildByTag(defaultEl, 'motor')
  if (motorEl) {
    for (const attr of motorEl.attributes) {
      defaults.motor[attr.name] = attr.value
    }
  }

  return defaults
}

/**
 * Get an attribute value, falling back to defaults if not explicitly set.
 */
function getAttr(el, name, defaults) {
  const val = el.getAttribute(name)
  if (val !== null) return val
  return defaults[name] ?? null
}

// ─── Main parser ───────────────────────────────────────────────────────────

/**
 * Parse an MJCF XML string into a BodyDef.
 * @param {string} xmlString - MJCF XML content
 * @param {object} [options]
 * @param {function} [options.DOMParser] - custom DOMParser constructor (for Node.js)
 * @returns {object} BodyDef
 * @throws {Error} if parsing fails
 */
export function parseMJCF(xmlString, options = {}) {
  const DOMParserClass = options.DOMParser
    || (typeof DOMParser !== 'undefined' ? DOMParser : null)

  if (!DOMParserClass) {
    throw new Error('parseMJCF: DOMParser not available. Pass options.DOMParser for Node.js.')
  }

  const parser = new DOMParserClass()
  // Inject rl-forge namespace declaration if not present, so DOMParser doesn't choke on rl-forge:* attributes
  let xml = xmlString
  if (xml.includes('rl-forge:') && !xml.includes('xmlns:rl-forge')) {
    xml = xml.replace('<mujoco', '<mujoco xmlns:rl-forge="http://rl-forge.dev/mjcf"')
  }
  const doc = parser.parseFromString(xml, 'text/xml')

  const parseError = doc.querySelector('parsererror')
  if (parseError) {
    throw new Error(`MJCF XML parse error: ${parseError.textContent}`)
  }

  const mujoco = findByTag(doc, 'mujoco')
  if (!mujoco) {
    throw new Error('MJCF: no <mujoco> root element found')
  }

  const name = mujoco.getAttribute('model') || 'unnamed'

  // Compiler settings
  const compilerEl = findByTag(mujoco, 'compiler')
  const angleUnit = compilerEl?.getAttribute('angle') || 'degree'
  const toRadians = angleUnit === 'degree' ? (deg) => deg * Math.PI / 180 : (rad) => rad

  // Detect coordinate convention from gravity direction
  // Standard MuJoCo: gravity "0 0 -9.81" (Z-up, XZ locomotion plane)
  // RL-Forge: gravity "0 -9.81 0" (Y-up, XY locomotion plane)
  const optionEl = findByTag(mujoco, 'option')
  const gravityStr = optionEl?.getAttribute('gravity') || '0 0 -9.81'
  const gravityVec = parseVec(gravityStr)
  const isZUp = Math.abs(gravityVec[2] || 0) > Math.abs(gravityVec[1] || 0)

  // Parse defaults
  const defaults = parseDefaults(mujoco)

  // Parse custom rl-forge metadata
  const custom = {}
  const customEl = findByTag(mujoco, 'custom')
  if (customEl) {
    for (const el of [...findAllByTag(customEl, 'numeric'), ...findAllByTag(customEl, 'text')]) {
      const key = el.getAttribute('name')
      const data = el.getAttribute('data')
      if (key) custom[key] = data
    }
  }

  const gravityScale = custom.gravityScale !== undefined ? parseFloat(custom.gravityScale) : 1.0
  const forwardBody = custom.forwardBody || undefined

  // Ground
  const ground = {
    y: custom.groundY !== undefined ? parseFloat(custom.groundY) : 0.0,
    friction: custom.groundFriction !== undefined ? parseFloat(custom.groundFriction) : 0.8,
    restitution: custom.groundRestitution !== undefined ? parseFloat(custom.groundRestitution) : 0.1,
  }

  // Default reward
  let defaultReward = undefined
  if (custom.defaultReward) {
    try {
      defaultReward = JSON.parse(custom.defaultReward)
    } catch { /* ignore */ }
  }

  // Parse actuators (joint name -> gear/maxTorque)
  const actuatorMap = new Map()
  const actuatorEl = findByTag(mujoco, 'actuator')
  if (actuatorEl) {
    for (const motor of findAllByTag(actuatorEl, 'motor')) {
      const jointName = motor.getAttribute('joint')
      const gear = parseFloat(getAttr(motor, 'gear', defaults.motor) || '0')
      if (jointName) actuatorMap.set(jointName, gear)
    }
  }

  // Context passed down during tree walk
  const ctx = { defaults, actuatorMap, toRadians, isZUp }

  // Walk the body tree
  const bodies = []
  const joints = []
  const worldbody = findByTag(mujoco, 'worldbody')

  if (worldbody) {
    for (const bodyEl of childrenByTag(worldbody, 'body')) {
      walkBody(bodyEl, null, bodies, joints, ctx)
    }
  }

  const def = {
    name,
    gravityScale,
    ground,
    bodies,
    joints,
  }

  if (forwardBody) def.forwardBody = forwardBody
  if (defaultReward) def.defaultReward = defaultReward

  return def
}

// ─── Body tree walker ──────────────────────────────────────────────────────

/**
 * Recursively walk <body> elements, extracting bodies and joints.
 */
function walkBody(bodyEl, parentId, bodies, joints, ctx) {
  const { defaults, actuatorMap, toRadians, isZUp } = ctx
  const id = bodyEl.getAttribute('name') || `body_${bodies.length}`
  const rawPos = parseVec(bodyEl.getAttribute('pos'))
  const fixed = bodyEl.getAttribute('rl-forge:fixed') === 'true'

  // Convert position based on coordinate convention
  const pos = isZUp
    ? [rawPos[0] || 0, rawPos[2] || 0]  // XZ plane -> XY (swap Y and Z)
    : [rawPos[0] || 0, rawPos[1] || 0]  // XY plane (RL-Forge native)

  // Parse geom (first direct child <geom>)
  const geom = firstChildByTag(bodyEl, 'geom')
  const body = {
    id,
    shape: 'box',
    mass: 1.0,
    friction: parseFloat(defaults.geom.friction || '0.3'),
    restitution: 0.0,
    spawnX: pos[0],
    spawnY: pos[1],
    spawnAngle: 0,
  }

  // Spawn angle (rl-forge extension)
  const spawnAngleAttr = bodyEl.getAttribute('rl-forge:spawnAngle')
  if (spawnAngleAttr) body.spawnAngle = parseFloat(spawnAngleAttr)

  if (fixed) body.fixed = true

  if (geom) {
    const geomType = getAttr(geom, 'type', defaults.geom) || 'box'

    if (geomType === 'box') {
      body.shape = 'box'
      const size = parseVec(getAttr(geom, 'size', defaults.geom))
      // MJCF box size is half-extents
      body.w = (size[0] || 0.1) * 2
      body.h = (isZUp ? (size[2] || size[1] || 0.1) : (size[1] || 0.1)) * 2
    } else if (geomType === 'capsule') {
      body.shape = 'capsule'
      const sizeStr = getAttr(geom, 'size', defaults.geom)
      body.radius = parseFloat(sizeStr) || 0.05
      const fromto = parseVec(geom.getAttribute('fromto'))
      if (fromto.length >= 6) {
        if (isZUp) {
          // Standard MuJoCo: capsule along Z axis, fromto uses Z components
          const dz = fromto[5] - fromto[2]
          const dy = fromto[4] - fromto[1]
          const dist = Math.sqrt(dz * dz + dy * dy)
          body.length = dist + 2 * body.radius
        } else {
          // RL-Forge: capsule along Y axis
          const dy = fromto[4] - fromto[1]
          body.length = Math.abs(dy) + 2 * body.radius
        }
      } else {
        // size="radius half-length" format
        const sizeParts = parseVec(sizeStr)
        if (sizeParts.length >= 2) {
          body.radius = sizeParts[0]
          body.length = sizeParts[1] * 2 + 2 * sizeParts[0]
        }
      }
    } else if (geomType === 'sphere') {
      body.shape = 'ball'
      body.radius = parseFloat(getAttr(geom, 'size', defaults.geom)) || 0.05
    }

    const mass = geom.getAttribute('mass')
    if (mass) body.mass = parseFloat(mass)

    const friction = geom.getAttribute('friction')
    if (friction) body.friction = parseFloat(friction.split(' ')[0])

    const restitution = geom.getAttribute('rl-forge:restitution')
    if (restitution) body.restitution = parseFloat(restitution)

    if (geom.getAttribute('rl-forge:isFootBody') === 'true') body.isFootBody = true

    const minY = geom.getAttribute('rl-forge:minY')
    if (minY) body.minY = parseFloat(minY)

    const maxAngle = geom.getAttribute('rl-forge:maxAngle')
    if (maxAngle) body.maxAngle = parseFloat(maxAngle)

    if (geom.getAttribute('rl-forge:terminateOnContact') === 'true') body.terminateOnContact = true
  }

  bodies.push(body)

  // Parse ALL direct child <joint> elements (a body can have multiple joints)
  // Standard Gym files use rootx/rootz/rooty on the torso for free-base
  const jointEls = childrenByTag(bodyEl, 'joint')
  for (const jointEl of jointEls) {
    if (!parentId) {
      // Root body joints: check if these are free-base joints (rootx/rootz/rooty)
      // Free-base joints have limited="false" — skip them (body is already free in Rapier)
      const limited = jointEl.getAttribute('limited')
      if (limited === 'false' || limited === null) continue
    }

    const jointId = jointEl.getAttribute('name') || `joint_${joints.length}`
    const jointType = getAttr(jointEl, 'type', defaults.joint) || 'hinge'

    // Parse range
    const range = parseVec(jointEl.getAttribute('range'))
    let lowerLimit, upperLimit
    if (range.length >= 2) {
      if (jointType === 'hinge') {
        lowerLimit = toRadians(range[0])
        upperLimit = toRadians(range[1])
      } else {
        // Prismatic: range is in meters, no conversion needed
        lowerLimit = range[0]
        upperLimit = range[1]
      }
    } else {
      lowerLimit = jointType === 'prismatic' || jointType === 'slide' ? -2.4 : -Math.PI
      upperLimit = jointType === 'prismatic' || jointType === 'slide' ? 2.4 : Math.PI
    }

    // Damping (inherit from defaults)
    const damping = parseFloat(getAttr(jointEl, 'damping', defaults.joint) || '5.0')

    // Max torque/force from actuator
    const maxTorque = actuatorMap.get(jointId) || 0

    // PD gains (rl-forge extension)
    const kpAttr = jointEl.getAttribute('rl-forge:kp')
    const kdAttr = jointEl.getAttribute('rl-forge:kd')
    const maxVelocityAttr = jointEl.getAttribute('rl-forge:maxVelocity')
    const stiffnessAttr = getAttr(jointEl, 'stiffness', defaults.joint)
    const restAngleAttr = jointEl.getAttribute('rl-forge:restAngle')

    // Anchor positions
    const anchorA = parseVec2(jointEl.getAttribute('rl-forge:anchorA'))
    const anchorBRaw = parseVec(jointEl.getAttribute('pos'))
    let anchorB
    if (isZUp && anchorBRaw.length >= 3) {
      anchorB = [anchorBRaw[0], anchorBRaw[2]]  // XZ -> XY
    } else if (anchorBRaw.length >= 2) {
      anchorB = [anchorBRaw[0], anchorBRaw[1]]
    } else {
      anchorB = [0, 0]
    }

    // Joint axis (for prismatic joints)
    const axisRaw = parseVec(jointEl.getAttribute('axis'))
    let axis
    if (isZUp && axisRaw.length >= 3) {
      axis = [axisRaw[0], axisRaw[2]]  // XZ -> XY
    } else if (axisRaw.length >= 2) {
      axis = [axisRaw[0], axisRaw[1]]
    }

    // Map MJCF joint type to BodyDef type
    const type = (jointType === 'slide' || jointType === 'prismatic') ? 'prismatic' : 'revolute'

    const joint = {
      id: jointId,
      type,
      bodyA: parentId,
      bodyB: id,
      anchorA: anchorA || [0, 0],
      anchorB,
      lowerLimit,
      upperLimit,
      maxTorque,
      maxVelocity: maxVelocityAttr ? parseFloat(maxVelocityAttr) : 8.0,
      damping,
    }

    // Prismatic axis
    if (type === 'prismatic' && axis) {
      joint.axis = axis
    }

    if (stiffnessAttr && parseFloat(stiffnessAttr) !== 0) joint.stiffness = parseFloat(stiffnessAttr)
    if (kpAttr) joint.kp = parseFloat(kpAttr)
    if (kdAttr) joint.kd = parseFloat(kdAttr)
    if (restAngleAttr) joint.restAngle = parseFloat(restAngleAttr)

    joints.push(joint)
  }

  // Recurse into child bodies
  for (const child of childrenByTag(bodyEl, 'body')) {
    walkBody(child, id, bodies, joints, ctx)
  }
}

// ─── Utility ───────────────────────────────────────────────────────────────

function parseVec(str) {
  if (!str) return []
  return str.trim().split(/\s+/).map(Number)
}

function parseVec2(str) {
  if (!str) return null
  const parts = str.trim().split(/\s+/).map(Number)
  if (parts.length >= 2) return [parts[0], parts[1]]
  return null
}
