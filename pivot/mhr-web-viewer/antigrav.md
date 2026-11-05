# MHR Web Viewer — Architecture Analysis & Multi-Actor Refactor Specification

> **Document Status:** Architectural Blueprint & Analysis  
> **Author:** Antigravity (Advanced Agentic Pair Programming)  
> **Date:** November 2026  
> **Target:** Momentum Human Rig (MHR) Interactive Web Viewer & Multi-Agent Simulation Engine  

---

## 1. Executive Summary

The **MHR Web Viewer** is an interactive, browser-based runtime for the **Momentum Human Rig (MHR)** model. It combines a local Python Flask backend (providing TorchScript ground truth, binary asset packing, and barycentric LOD mappings) with a client-side Three.js frontend capable of sub-millisecond skeletal deformation, CPU CSR blendshapes, and real-time GPU skinning.

Recently, the system was expanded from a single-character parameter inspector into a dual stress-test and autonomous multi-agent simulation platform. However, the application currently exhibits an **architectural duality**:
1. The **core UI and rigging tools** (left sidebar parameters, joint gizmos, 3D skeleton points, reset actions, and backend sync) operate strictly on a **single global character singleton**.
2. The **autonomous simulation engine** manages **multiple independent agents** with dedicated skeletons, kinematics, and locomotion, but their properties are sequestered in an isolated panel.

This document provides a comprehensive technical audit of the current architecture, identifies key bottlenecks and structural couplings, and presents an architectural design for refactoring the viewer into a **unified, multi-actor, agent-focused system**. Under this refactored model, **any actor in the scene can be dynamically selected** (via 3D raycasting, joint sphere picking, or UI dropdowns), and the **entire UI (pose sliders, joint gizmo, blendshapes, appearance, and kinematics) will seamlessly focus on and drive that specific actor**.

---

## 2. Current System Architecture

```
+---------------------------------------------------------------------------------------+
|                                    BACKEND (Flask)                                    |
|  - TorchScript MHR Model (mhr_model.pt)          - Binary Assets (rig_mapping, bs)    |
|  - Barycentric LOD Mappings (LOD 0-6)            - Mesh Exporters (OBJ, GLB, USD)     |
+-------------------------------------------+-------------------------------------------+
                                            | Packed Binary Arrays / REST JSON
                                            v
+---------------------------------------------------------------------------------------+
|                                  FRONTEND (Three.js)                                  |
|                                                                                       |
|   +-----------------------+   +----------------------+   +------------------------+   |
|   |       state.js        |   |       scene.js       |   |         rig.js         |   |
|   | Global singletons,    |   | WebGPU / WebGL,      |   | CPU CSR Blendshapes,   |   |
|   | metadata, DOM queries |   | Lights, OrbitControls|   | Linear Rig Mapping     |   |
|   +-----------+-----------+   +----------+-----------+   +-----------+------------+   |
|               |                          |                           |                |
|               +--------------------------+---------------------------+                |
|                                          |                                            |
|   +-----------------------+   +----------v-----------+   +------------------------+   |
|   |       gizmo.js        |   |        app.js        |   |        params.js       |   |
|   | 3D Joint picker,      |   | Orchestration,       |   | Sliders, search,       |   |
|   | TransformControls     |   | Event wiring, sync   |   | categories, tabs       |   |
|   +-----------------------+   +----------+-----------+   +------------------------+   |
|                                          |                                            |
|                               +----------v-----------+                                |
|                               |       crowd.js       |                                |
|                               | Clones vs Instancing |                                |
|                               +----------+-----------+                                |
|                                          |                                            |
|                               +----------v-----------+                                |
|                               |     sim/ (Engine)    |                                |
|                               | Agent, Steering,     |                                |
|                               | Gait, AgentSim       |                                |
|                               +----------------------+                                |
+---------------------------------------------------------------------------------------+
```

### 2.1 Component Breakdown

#### A. State & Metadata (`state.js`)
- **Central Singleton Pattern:** The `state` object acts as a global blackboard. It directly stores `state.metadata`, `state.skinnedMesh`, `state.rootBone`, `state.skeleton`, `state.rigMapping`, `state.blendshapes`, `state.lodMappings`, and `state.boneIndexMap`.
- **Parameter Mutation:** Parameter values live inside `state.metadata.groups.pose`, `identity`, and `expression`. Modifying a parameter directly mutates `parameter.value` inside these global arrays.
- **Limitation:** There is no concept of multiple character states; every subsystem assumes there is only one character armature in existence.

#### B. Rigging & Evaluation Pipeline (`rig.js`)
- **Client-Side Linear Rig Mapping:** `evaluatePoseClientSide()` maps a 204-dimensional parameter array $\theta_P$ through `TpMatrix` ($889 \times 204$) to produce 127 joint transforms $\theta_J$. It then directly iterates over `state.skeleton.bones[state.boneIndexMap[j]]`, applying positions and quaternions with `ZYX` Euler ordering.
- **Client-Side Blendshape Deformations:** `evaluateBlendshapesClientSide()` performs CPU CSR multiplication of active weights against neutral deltas, directly overwriting `state.skinnedMesh.geometry.attributes.position.array`.
- **Limitation:** Functions do not accept an arbitrary actor or skeleton; they read and write directly to `state.skeleton` and `state.skinnedMesh`.

#### C. Viewport, Materials & Gizmos (`scene.js`, `gizmo.js`)
- **Rendering Loop:** Operates on-demand via `requestRender()`, switching to continuous ticking only when simulation or auto-randomization is active.
- **Single Material Instance:** `bodyMaterial` is a single `MeshStandardMaterial` instance applied to the main mesh.
- **Joint Selection & 3D Gizmo:**
  - Instanced spheres (`THREE.InstancedMesh`) are placed at world positions sampled from `state.jointPositions`.
  - A `THREE.Raycaster` on canvas `pointerdown` detects clicks on these joint spheres.
  - Upon selecting a joint, `transformControls` attaches to an invisible proxy object at that joint's location, with axes locked to the primary driving parameter.
- **Limitation:** The joint picking system and gizmo are coupled to the main character at $(0, 0, 0)$.

#### D. Multi-Agent Simulation Subsystem (`static/js/sim/`)
- **`Agent.js`:** Encapsulates an autonomous character. Deeply clones the bone tree (`sourceRootBone.clone(true)`) and creates an independent `THREE.Skeleton`. It shares the `BufferGeometry` for zero additional VRAM and binds via an enclosing `THREE.Group`.
- **`GaitController.js`:** Solves procedural walking, running, and idle breathing. Takes advantage of the sparsity of MHR's `TpMatrix`: only 11 locomotion parameters are active, touching only **13 non-zero matrix entries**. Evaluates joint transforms in $<0.001\text{ ms}$.
- **`Steering.js`:** Zero-allocation Craig Reynolds steering behaviors (Wander, Separation, Alignment, Cohesion, Obstacle Avoidance, Containment, Circle).
- **`AgentSimulation.js`:** Coordinates the simulation loop (`requestAnimationFrame`), time scaling ($0.25\times$ to $2.0\times$), discrete frame stepping, actor selection, and ground indicator tracking.

---

## 3. Critical Analysis: Structural Bottlenecks

While the simulation subsystem successfully proved that multiple independent MHR skeletons can run concurrently at 60+ FPS, several architectural limitations prevent the viewer from operating as a full multi-actor editor:

| Subsystem | Current State | Bottleneck / Conflict |
| :--- | :--- | :--- |
| **Parameter Model** | Hardcoded to `state.metadata.groups` | Only one character's pose, identity, and expression can be viewed or edited in the main UI sidebar. |
| **Posing Engine** | `evaluatePoseClientSide()` accesses `state.skeleton` directly | Cannot evaluate or apply poses to an arbitrary agent without clobbering the main skeleton. |
| **3D Joint Gizmo** | `gizmo.js` builds joint spheres from `state.jointPositions` | Clicking joints only works on the center character; agents walking across the room cannot have their joints clicked or manipulated. |
| **Appearance & Materials** | Single `bodyMaterial` shared across scene UI | Color and wireframe toggles affect either all agents or only the main model, with ad-hoc overrides. |
| **Simulation vs Studio** | Split UI panels | Global simulation controls (play, step, spawn) are nestled inside a "crowd card", disconnected from the primary inspector controls. |

---

## 4. Target Vision: Unified Multi-Actor Architecture

The goal of the refactoring is to establish a **clean separation between Global Environment Systems and Actor-Scoped Systems**:

```
+---------------------------------------------------------------------------------------+
|                                    GLOBAL SYSTEMS                                     |
|  - Active Camera & OrbitControls                 - Scene Lights, Floor Grid & Bounds  |
|  - Global Simulation State (Play, Step, Speed)   - Spawner / Crowd Manager            |
|  - Selected / Focused Actor Target (activeActor) - Render Pipeline & Telemetry        |
+-------------------------------------------+-------------------------------------------+
                                            | binds
                                            v
+---------------------------------------------------------------------------------------+
|                                 ACTOR-SCOPED SYSTEMS                                  |
|                             (Active Selected Character)                               |
|                                                                                       |
|  - Transform & Kinematics     - Armature & Skeleton         - Posed Geometry          |
|    (Position, Heading, AI)      (Bones, Matrices, Gizmo)      (LOD, Blendshapes)      |
|                                                                                       |
|  - Parameter State Vector     - Material & Appearance       - Control Bindings        |
|    (Pose, Identity, Expr)       (Color, Wireframe, Rough)     (Sliders, Gizmos, Reset)|
+---------------------------------------------------------------------------------------+
```

### Key Principles of the Refactor

1. **Every Character is an `Actor`:**  
   The primary character standing at the origin is simply `Actor #0` (the "Anchor" or "Director Actor"). Spawned characters are `Actor #1`, `Actor #2`, etc. All actors share the same class interface, data layout, and capabilities.

2. **UI is a Stateless View / Projection:**  
   The parameter sidebar, joint list, 3D gizmos, and appearance controls should not own character data. Instead, they **bind** to whichever `Actor` is currently selected (`activeActor`). When switching from Actor #0 to Actor #3, the UI updates its sliders and gizmos to reflect Actor #3's pose and attributes.

3. **3D Interaction Reusability:**  
   The existing joint sphere picking and transform gizmo in `gizmo.js` should not be rewritten; rather, they should transform their coordinate frame into the **active actor's local/world space**. Clicking on any actor's elbow or shoulder should attach the gizmo to that actor's joint.

---

## 5. Proposed Refactor Plan

### Phase 1: The Unified `Actor` Class Abstraction

Create a first-class `Actor` model in `static/js/core/Actor.js` (or `static/js/sim/Actor.js`) encapsulating all character-specific state:

```javascript
export class Actor {
  constructor(id, name, options = {}) {
    this.id = id;
    this.name = name;
    
    // 1. 3D Scene Objects
    this.group = new THREE.Group();
    this.rootBone = null;
    this.skeleton = null;
    this.mesh = null;
    this.material = null;
    
    // 2. Character Parameter State (Full MHR parameter vector)
    this.parameters = {
      pose: new Float32Array(204),
      identity: new Float32Array(idCount),
      expression: new Float32Array(feCount),
    };
    
    // 3. Dynamic Joint Positions in World Space (for Gizmo raycasting)
    this.jointPositions = new Float32Array(jointCount * 3);
    
    // 4. Kinematics & Simulation State
    this.position = new THREE.Vector3();
    this.velocity = new THREE.Vector3();
    this.heading = 0;
    this.maxSpeed = 1.3;
    this.scale = 1.0;
    this.behavior = "wander"; // "wander", "flock", "circle", "manual", "idle"
    
    // 5. Procedural Locomotion State
    this.gaitPhase = 0;
    this.cadence = 3.3;
  }
  
  // Scoped evaluation methods:
  evaluatePose(rigMapping) { ... }
  evaluateBlendshapes(blendshapesData) { ... }
  updateWorldJointPositions() { ... }
  setMaterialColor(color) { ... }
}
```

### Phase 2: Decouple Rigging & Gizmo from Global Singletons

1. **Parameterize `evaluatePose` & `evaluateBlendshapes`:**  
   Refactor `rig.js` functions so they accept the target actor (or target skeleton and parameter array) rather than reading from `state.skeleton`.
   ```javascript
   // Before:
   export function evaluatePoseClientSide() { ... uses state.skeleton ... }
   
   // Refactored:
   export function evaluatePoseOnActor(actor, rigMapping) {
     const thetaJ = computeThetaJFromParams(actor.parameters.pose, rigMapping);
     applyThetaJToSkeleton(actor.skeleton, actor.boneIndexMap, rigMapping, thetaJ);
     actor.updateWorldJointPositions();
   }
   ```

2. **World-Space Joint Gizmo Alignment (`gizmo.js`):**  
   Currently, `gizmo.js` updates `spheres` (instanced spheres) from `state.jointPositions`.  
   When an actor is selected:
   - The instanced joint spheres update their matrices from `activeActor.jointPositions` (which are in world space).
   - The gizmo attaches to `activeActor.skeleton.bones[boneIndex]`.
   - Modifying a joint parameter via the gizmo updates `activeActor.parameters.pose[paramIndex]` and triggers `evaluatePoseOnActor(activeActor)`.

### Phase 3: Global System vs Actor System Partitioning

Re-organize UI controls into two clear functional tiers:

#### Tier 1: Global Controls (Top Bar & Environment Deck)
- **Active Camera & Projection:** Perspective / Ortho toggle, cardinal view buttons (`Front`, `Back`, `Left`, `Right`), Reset Camera.
- **Global Simulation Master:**
  - Play / Pause simulation.
  - Step forward ($+1$ frame), Skip forward ($+10$ frames).
  - Time scale slider / presets ($0.25\times$, $0.5\times$, $1.0\times$, $2.0\times$).
- **Actor Spawner & Directory:**
  - Spawner input (count, group behavior).
  - Actor directory dropdown / list: `[🌟 Main Character | 👤 Actor #1 | 👤 Actor #2 | ...]`.
  - Global Clear / Reset buttons.

#### Tier 2: Actor-Scoped Controls (Inspector Sidebar)
Whenever `activeActor` changes:
- **Header:** Displays active actor badge (e.g. `Actor #2 · Wandering`), with a `[Focus Cam]` button and a toggle between **Manual Posing Mode** and **Autonomous Simulation Mode**.
- **Pose / Identity / Expression Tabs:**
  - Sliders populate with values from `activeActor.parameters`.
  - Moving a slider deforms `activeActor` in real-time.
- **Appearance:**
  - Body color picker reads/writes `activeActor.material.color`.
  - Wireframe toggle reads/writes `activeActor.material.wireframe`.
  - Height / Stature scale reads/writes `activeActor.scale`.
- **Kinematics & Brain (if autonomous):**
  - Behavior mode select (`Wander`, `Flock`, `Circle`, `Scatter`, `Idle`).
  - Max speed & Cadence sliders.
  - Live velocity and coordinate telemetry.

---

## 6. Implementation Strategy & Safety Guarantees

To ensure that the refactor proceeds without breaking regressions or causing service disruption:

1. **Zero-Regression Contract Testing:**  
   The test suite contains strict contract checks in [test_frontend_wiring.py](file:///a:/dev/GNM-V/pivot/MHR/web-viewer/tests/test_frontend_wiring.py) and numerical parity checks in [test_client_parity.py](file:///a:/dev/GNM-V/pivot/MHR/web-viewer/tests/test_client_parity.py).  
   All DOM element IDs required by `test_frontend_wiring.py` (`id="crowd-card"`, `id="crowd-clone-*"`, `id="crowd-inst-*"`, parameter sliders) will be preserved or aliased.

2. **Step-by-Step Migration Phases:**
   - **Step 1:** Wrap the main model into `Actor #0` (`directorActor`) so that the existing single-character experience runs on the new `Actor` interface without behavioral differences.
   - **Step 2:** Implement `ActorManager` (replacing the standalone `AgentSimulation`) to hold `Map<number, Actor>` and track `activeActorId`.
   - **Step 3:** Redirect `params.js` and `app.js` parameter listeners to read/write `activeActor.parameters`.
   - **Step 4:** Update `gizmo.js` to sample joint world positions from `activeActor` and attach `transformControls` to `activeActor`'s joints.
   - **Step 5:** Polish UI bindings, selection indicators, and follow-camera mechanics.

---

## 7. Conclusion

By shifting from a **singleton character architecture** to a **unified actor-centric architecture**, the MHR Web Viewer transforms from a static inspection tool into a comprehensive **character director and multi-agent simulation sandbox**. The existing dynamic joint gizmo, CSR blendshape pipeline, and procedural gait kinematics can all be reused across every actor in the scene without redundant code or performance compromises.
