/**
 * MJCF Serializer: BodyDef -> MJCF XML string
 *
 * Produces a subset of MuJoCo's MJCF format suitable for 2D rigid-body
 * characters. Uses the tree structure implied by joint parent-child
 * relationships to produce nested <body> elements.
 *
 * Supported subset:
 *   - <mujoco model="...">
 *   - <option gravity="0 0 -9.81" />
 *   - <default> with friction, damping
 *   - <worldbody> with nested <body>, <geom>, <joint>
 *   - <actuator> with <motor> elements
 *
 * Convention for 2D: we use the XY plane. MJCF is natively 3D so we
 * map our 2D coordinates: x->x, y->z (vertical), and rotation about
 * the z-axis in our 2D maps to hinge joints about the y-axis in MJCF.
 * However, for simplicity and clarity, we use custom attributes
 * (rl-forge:*) to store 2D-specific properties that don't map cleanly.
 */

/**
 * Serialize a BodyDef to an MJCF XML string.
 * @param {object} def - a BodyDef (from bodyDef.js)
 * @returns {string} MJCF XML
 */
export function serializeToMJCF(def) {
  const lines = []
  const indent = (n) => '  '.repeat(n)

  lines.push(`<mujoco model="${escapeXml(def.name || 'unnamed')}">`)

  // Compiler / option
  lines.push(`${indent(1)}<compiler angle="radian" />`)
  lines.push(`${indent(1)}<option gravity="0 -9.81 0" />`)

  // Defaults
  lines.push(`${indent(1)}<default>`)
  lines.push(`${indent(2)}<geom friction="${def.ground?.friction ?? 0.8}" />`)
  lines.push(`${indent(2)}<joint damping="5.0" />`)
  lines.push(`${indent(1)}</default>`)

  // Custom rl-forge metadata
  lines.push(`${indent(1)}<custom>`)
  lines.push(`${indent(2)}<numeric name="gravityScale" data="${def.gravityScale ?? 1.0}" />`)
  if (def.forwardBody) {
    lines.push(`${indent(2)}<text name="forwardBody" data="${escapeXml(def.forwardBody)}" />`)
  }
  if (def.ground) {
    lines.push(`${indent(2)}<numeric name="groundY" data="${def.ground.y ?? 0}" />`)
    lines.push(`${indent(2)}<numeric name="groundFriction" data="${def.ground.friction ?? 0.8}" />`)
    lines.push(`${indent(2)}<numeric name="groundRestitution" data="${def.ground.restitution ?? 0.1}" />`)
  }
  if (def.defaultReward) {
    lines.push(`${indent(2)}<text name="defaultReward" data="${escapeXml(JSON.stringify(def.defaultReward))}" />`)
  }
  lines.push(`${indent(1)}</custom>`)

  // Build the body tree
  const bodyMap = new Map()
  for (const b of def.bodies) bodyMap.set(b.id, b)

  // Determine parent relationships from joints
  // The root body (no parent via joint) gets placed directly in worldbody
  const childOf = new Map() // childBodyId -> { parentBodyId, joint }
  for (const j of def.joints) {
    childOf.set(j.bodyB, { parentId: j.bodyA, joint: j })
  }

  // Find root bodies (not referenced as bodyB in any joint)
  const rootIds = def.bodies.filter(b => !childOf.has(b.id)).map(b => b.id)

  // Worldbody
  lines.push(`${indent(1)}<worldbody>`)

  // Ground plane
  lines.push(`${indent(2)}<geom name="ground" type="plane" pos="0 ${def.ground?.y ?? 0} 0" size="50 1 1" friction="${def.ground?.friction ?? 0.8} 0.005 0.0001" rgba="0.3 0.3 0.3 1" />`)

  // Recursively emit body tree
  function emitBody(bodyId, depth) {
    const b = bodyMap.get(bodyId)
    if (!b) return

    const jointInfo = childOf.get(bodyId)
    const d = depth

    // Body element
    const posAttr = `pos="${b.spawnX} ${b.spawnY} 0"`
    const fixedAttr = b.fixed ? ' rl-forge:fixed="true"' : ''
    lines.push(`${indent(d)}<body name="${escapeXml(b.id)}" ${posAttr}${fixedAttr}>`)

    // Joint (if this body has a parent joint)
    if (jointInfo) {
      const j = jointInfo.joint
      const limited = (j.lowerLimit !== undefined && j.upperLimit !== undefined) ? 'true' : 'false'
      const rangeAttr = limited === 'true' ? ` range="${j.lowerLimit} ${j.upperLimit}"` : ''
      const dampAttr = j.damping !== undefined ? ` damping="${j.damping}"` : ''

      // Store anchor positions as joint pos (relative to parent body)
      const anchorStr = j.anchorB ? `pos="${j.anchorB[0]} ${j.anchorB[1]} 0"` : ''

      let extras = ''
      if (j.kp !== undefined) extras += ` rl-forge:kp="${j.kp}"`
      if (j.kd !== undefined) extras += ` rl-forge:kd="${j.kd}"`
      if (j.stiffness !== undefined && j.stiffness !== 0) extras += ` stiffness="${j.stiffness}"`
      if (j.restAngle !== undefined && j.restAngle !== 0) extras += ` rl-forge:restAngle="${j.restAngle}"`
      if (j.maxVelocity !== undefined) extras += ` rl-forge:maxVelocity="${j.maxVelocity}"`

      // Store parent anchor
      if (j.anchorA) extras += ` rl-forge:anchorA="${j.anchorA[0]} ${j.anchorA[1]}"`

      // Joint type and axis
      const jType = j.type === 'prismatic' ? 'slide' : 'hinge'
      const axisAttr = j.type === 'prismatic'
        ? `axis="${(j.axis || [1, 0]).join(' ')} 0"`
        : 'axis="0 0 1"'

      lines.push(`${indent(d + 1)}<joint name="${escapeXml(j.id)}" type="${jType}" ${axisAttr} limited="${limited}"${rangeAttr} ${anchorStr}${dampAttr}${extras} />`)
    }

    // Geom
    const geomAttrs = buildGeomAttrs(b)
    lines.push(`${indent(d + 1)}<geom ${geomAttrs} />`)

    // Emit children
    const children = def.joints
      .filter(j => j.bodyA === bodyId)
      .map(j => j.bodyB)

    for (const childId of children) {
      emitBody(childId, d + 1)
    }

    lines.push(`${indent(d)}</body>`)
  }

  for (const rootId of rootIds) {
    emitBody(rootId, 2)
  }

  lines.push(`${indent(1)}</worldbody>`)

  // Actuators
  const actuatedJoints = def.joints.filter(j => j.maxTorque > 0)
  if (actuatedJoints.length > 0) {
    lines.push(`${indent(1)}<actuator>`)
    for (const j of actuatedJoints) {
      lines.push(`${indent(2)}<motor name="${escapeXml(j.id)}_motor" joint="${escapeXml(j.id)}" gear="${j.maxTorque}" ctrllimited="true" ctrlrange="-1 1" />`)
    }
    lines.push(`${indent(1)}</actuator>`)
  }

  lines.push(`</mujoco>`)
  return lines.join('\n')
}

function buildGeomAttrs(b) {
  const parts = []
  parts.push(`name="${escapeXml(b.id)}"`)

  if (b.shape === 'box') {
    parts.push(`type="box"`)
    parts.push(`size="${b.w / 2} ${b.h / 2} 0.05"`)
  } else if (b.shape === 'capsule') {
    parts.push(`type="capsule"`)
    const halfLen = (b.length - 2 * b.radius) / 2
    parts.push(`size="${b.radius}"`)
    parts.push(`fromto="0 ${-halfLen} 0 0 ${halfLen} 0"`)
  } else if (b.shape === 'ball') {
    parts.push(`type="sphere"`)
    parts.push(`size="${b.radius}"`)
  }

  parts.push(`mass="${b.mass}"`)
  parts.push(`friction="${b.friction ?? 0.3}"`)

  if (b.restitution !== undefined && b.restitution !== 0) {
    parts.push(`rl-forge:restitution="${b.restitution}"`)
  }

  if (b.isFootBody) parts.push(`rl-forge:isFootBody="true"`)
  if (b.minY !== undefined) parts.push(`rl-forge:minY="${b.minY}"`)
  if (b.maxAngle !== undefined) parts.push(`rl-forge:maxAngle="${b.maxAngle}"`)
  if (b.terminateOnContact) parts.push(`rl-forge:terminateOnContact="true"`)

  return parts.join(' ')
}

function escapeXml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}
