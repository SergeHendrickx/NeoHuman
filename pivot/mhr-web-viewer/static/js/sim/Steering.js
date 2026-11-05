import * as THREE from "three";

/**
 * Craig Reynolds-style steering behaviors for autonomous characters.
 * All functions avoid heap allocations during simulation ticks by reusing internal vector scratchpads.
 */

const _v1 = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _desired = new THREE.Vector3();
const _steer = new THREE.Vector3();

export class Steering {
  /**
   * Smooth wandering using a circle projected ahead of the character.
   */
  static wander(agent, deltaTime, circleDistance = 2.0, circleRadius = 1.2, changeRate = 3.0) {
    // Perturb wander angle
    agent.wanderAngle += (Math.random() - 0.5) * changeRate * deltaTime;

    // Projected circle center ahead of agent heading
    const heading = agent.group.rotation.y;
    const circleCenter = _v1.set(
      agent.position.x + Math.sin(heading) * circleDistance,
      0,
      agent.position.z + Math.cos(heading) * circleDistance,
    );

    // Target point on the circle circumference
    const target = _v2.set(
      circleCenter.x + Math.sin(agent.wanderAngle) * circleRadius,
      0,
      circleCenter.z + Math.cos(agent.wanderAngle) * circleRadius,
    );

    return this.seek(agent, target);
  }

  /**
   * Steer towards a target position.
   */
  static seek(agent, target) {
    _desired.subVectors(target, agent.position);
    _desired.y = 0;
    const distance = _desired.length();
    if (distance < 0.001) return _steer.set(0, 0, 0);

    _desired.normalize().multiplyScalar(agent.maxSpeed);
    _steer.subVectors(_desired, agent.velocity).clampLength(0, agent.maxForce);
    return _steer;
  }

  /**
   * Steer away from a threat / target.
   */
  static flee(agent, threat, panicRadius = 4.0) {
    _desired.subVectors(agent.position, threat);
    _desired.y = 0;
    const dist = _desired.length();
    if (dist > panicRadius || dist < 0.001) return _steer.set(0, 0, 0);

    _desired.normalize().multiplyScalar(agent.maxSpeed);
    _steer.subVectors(_desired, agent.velocity).clampLength(0, agent.maxForce * 1.5);
    return _steer;
  }

  /**
   * Separation: avoid crowding and stepping on nearby agents.
   */
  static separation(agent, allAgents, desiredDistance = 1.2) {
    _steer.set(0, 0, 0);
    let count = 0;

    for (let i = 0; i < allAgents.length; i++) {
      const other = allAgents[i];
      if (other === agent) continue;

      const d = agent.position.distanceTo(other.position);
      if (d > 0 && d < desiredDistance) {
        _v1.subVectors(agent.position, other.position);
        _v1.y = 0;
        _v1.normalize();
        _v1.divideScalar(d); // Weight by inverse distance (stronger when closer)
        _steer.add(_v1);
        count++;
      }
    }

    if (count > 0) {
      _steer.divideScalar(count);
      if (_steer.lengthSq() > 0.001) {
        _steer.normalize().multiplyScalar(agent.maxSpeed);
        _steer.sub(agent.velocity).clampLength(0, agent.maxForce * 2.0);
      }
    }
    return _steer;
  }

  /**
   * Alignment: steer in the average direction of neighbors.
   */
  static alignment(agent, allAgents, neighborDistance = 3.5) {
    _steer.set(0, 0, 0);
    let count = 0;

    for (let i = 0; i < allAgents.length; i++) {
      const other = allAgents[i];
      if (other === agent) continue;

      const d = agent.position.distanceTo(other.position);
      if (d > 0 && d < neighborDistance) {
        _steer.add(other.velocity);
        count++;
      }
    }

    if (count > 0) {
      _steer.divideScalar(count);
      _steer.y = 0;
      if (_steer.lengthSq() > 0.001) {
        _steer.normalize().multiplyScalar(agent.maxSpeed);
        _steer.sub(agent.velocity).clampLength(0, agent.maxForce);
      }
    }
    return _steer;
  }

  /**
   * Cohesion: steer toward the center of mass of neighbors.
   */
  static cohesion(agent, allAgents, neighborDistance = 3.5) {
    _v1.set(0, 0, 0);
    let count = 0;

    for (let i = 0; i < allAgents.length; i++) {
      const other = allAgents[i];
      if (other === agent) continue;

      const d = agent.position.distanceTo(other.position);
      if (d > 0 && d < neighborDistance) {
        _v1.add(other.position);
        count++;
      }
    }

    if (count > 0) {
      _v1.divideScalar(count);
      return this.seek(agent, _v1);
    }
    return _steer.set(0, 0, 0);
  }

  /**
   * Repel from the center obstacle (main character) to prevent clipping.
   */
  static avoidCenter(agent, centerPos, avoidRadius = 1.6) {
    _desired.subVectors(agent.position, centerPos);
    _desired.y = 0;
    const dist = _desired.length();

    if (dist < avoidRadius) {
      _desired.normalize().multiplyScalar(agent.maxSpeed);
      _steer.subVectors(_desired, agent.velocity);
      // Inverse falloff: strong push when inside the radius
      const factor = (avoidRadius - dist) / avoidRadius + 1.0;
      _steer.multiplyScalar(factor);
      _steer.clampLength(0, agent.maxForce * 2.5);
      return _steer;
    }
    return _steer.set(0, 0, 0);
  }

  /**
   * Boundary containment: soft steering force away from outer walls back toward center.
   */
  static contain(agent, bounds, margin = 1.2) {
    _steer.set(0, 0, 0);

    if (agent.position.x < bounds.minX + margin) {
      _steer.x = agent.maxSpeed;
    } else if (agent.position.x > bounds.maxX - margin) {
      _steer.x = -agent.maxSpeed;
    }

    if (agent.position.z < bounds.minZ + margin) {
      _steer.z = agent.maxSpeed;
    } else if (agent.position.z > bounds.maxZ - margin) {
      _steer.z = -agent.maxSpeed;
    }

    if (_steer.lengthSq() > 0) {
      _steer.normalize().multiplyScalar(agent.maxSpeed);
      _steer.sub(agent.velocity).clampLength(0, agent.maxForce * 2.0);
    }
    return _steer;
  }

  /**
   * Orbit/audience behavior: gather in a ring around the center model, facing inward.
   */
  static circleCenter(agent, centerPos, desiredRadius = 3.2) {
    _desired.subVectors(agent.position, centerPos);
    _desired.y = 0;
    const currentDist = _desired.length();

    if (currentDist < 0.001) {
      return _steer.set(agent.maxSpeed, 0, 0);
    }

    // Radial force to adjust distance towards desiredRadius
    const radialError = currentDist - desiredRadius;
    const radialDir = _v1.copy(_desired).normalize();

    // Tangential direction (perpendicular in XZ plane for orbiting)
    const tangentDir = _v2.set(-radialDir.z, 0, radialDir.x);

    // Combine radial seeking with tangential motion
    _steer.copy(tangentDir).multiplyScalar(agent.maxSpeed * 0.6);
    _steer.addScaledVector(radialDir, -radialError * 1.5);
    _steer.sub(agent.velocity).clampLength(0, agent.maxForce);
    return _steer;
  }
}
