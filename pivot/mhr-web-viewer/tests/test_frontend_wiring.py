"""Static contract checks between the viewer markup and its JS modules.

No browser or model assets are required: these guard the element-ID wiring
that cards and controls must keep in sync (every state.js query must resolve
in index.html) and pin deliberately removed UI (export, global LOD, the
clone/instance crowd tiers) as absent.
"""

import re
from pathlib import Path

WEB_VIEWER = Path(__file__).resolve().parent.parent
INDEX_HTML = WEB_VIEWER / "static" / "index.html"
STATE_JS = WEB_VIEWER / "static" / "js" / "state.js"
PERF_JS = WEB_VIEWER / "static" / "js" / "perf.js"
GPU_JS = WEB_VIEWER / "static" / "js" / "gpu"


def test_state_queries_resolve_in_markup():
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = STATE_JS.read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    queried = set(re.findall(r'querySelector\("#([^"]+)"\)', js))
    assert queried, "expected state.js to query element IDs"
    missing = sorted(queried - html_ids)
    assert not missing, f"state.js queries missing from index.html: {missing}"


def test_crowd_card_controls_present():
    # Clone/instance expectations were removed with those tiers (see
    # test_clone_and_instance_tiers_removed); the sim controls pin the
    # remaining card in test_agent_simulation_controls_present.
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="crowd-card"' in html, "missing crowd card section"


def test_pipeline_breakdown_is_live_wired():
    """Pipeline bar segments expose IDs and perf.js drives them from measurements."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    js = PERF_JS.read_text(encoding="utf-8")
    stages = (
        ("fk", "Fk"),
        ("blend", "Blend"),
        ("corr", "Corr"),
        ("mesh", "Mesh"),
        ("render", "Render"),
    )
    for slug, key in stages:
        assert f'id="perf-seg-{slug}"' in html, (
            f"missing pipeline segment: {slug}"
        )
        assert f'id="perf-val-{slug}"' in html, (
            f"missing pipeline legend value: {slug}"
        )
        assert f"perfSeg{key}" in js, f"perf.js never updates segment: {slug}"
        assert f"perfVal{key}" in js, f"perf.js never updates legend: {slug}"
    assert "recordStageTimes" in js, (
        "perf.js does not publish live stage timings"
    )
    # WebGPU `info.render.calls` counts render() invocations (never reset);
    # per-object draws live in `drawCalls` (WebGL fallback: classic `calls`).
    assert "infoRender?.drawCalls ?? infoRender?.calls" in js, (
        "drawcalls meter regressed to the invocation counter"
    )
    bench = (WEB_VIEWER / "static" / "js" / "bench.js").read_text(encoding="utf-8")
    assert "info.drawCalls ?? info.calls" in bench, (
        "bench sampler regressed to the invocation counter"
    )
    scene = (WEB_VIEWER / "static" / "js" / "scene.js").read_text(encoding="utf-8")
    assert scene.count("renderer.info?.reset?.();") == 2, (
        "render paths must own info.reset (we drive our own rAF loop)"
    )
    assert "renderDbg" not in scene, "render diagnostic survived the hunt"
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "stepDbg" not in sim, "step diagnostic survived the hunt"


def test_agent_simulation_controls_present():
    html = INDEX_HTML.read_text(encoding="utf-8")
    for suffix in ("count", "spawn", "clear", "behavior", "play", "total", "summary"):
        assert f'id="crowd-sim-{suffix}"' in html, f"missing crowd-sim-{suffix}"
    # Transport is global: play/step/skip/speed live in the navbar, not Crowd.
    top = html[html.index('<header class="topbar">'):html.index("</header>")]
    for suffix in ("play", "step", "skip", "speed"):
        assert f'id="crowd-sim-{suffix}"' in top, f"transport escaped the navbar: crowd-sim-{suffix}"
    crowd = html[html.index('id="wtab-crowd"'):html.index('id="wtab-actors"')]
    for suffix in ("play", "step", "skip", "speed"):
        assert f'id="crowd-sim-{suffix}"' not in crowd, f"transport still in Crowd: crowd-sim-{suffix}"
    for suffix in ("autolod", "lodcolors"):
        assert f'id="crowd-sim-{suffix}"' in crowd, f"crowd setting left Crowd: crowd-sim-{suffix}"


def test_sim_time_is_always_on():
    """Time flows from boot: autostart, no auto-stop on empty, quiet idle log."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "startSimulation();" in app, "boot never starts the sim loop"
    crowd = (WEB_VIEWER / "static" / "js" / "crowd.js").read_text(encoding="utf-8")
    assert "export function startSimulation" in crowd, "crowd.js lost the start export"
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    cleared = sim[sim.index("  clear() {"):sim.index("  clearCpuAgents() {")]
    assert "this.stop()" not in cleared, "clear() still stops time"
    cpu = sim[sim.index("  clearCpuAgents() {"):sim.index("  resetPools() {")]
    assert "this.stop()" not in cpu, "clearCpuAgents() still stops time"
    log = sim[sim.index("  _maybeLogPerf() {"):sim.index("  _tick(currentTime) {")]
    assert "agents.length > 0" in log, "idle [perf] line no longer gated on agents"
    # Pause freezes the satellite clocks too: Main blink scheduler + anchor clip pump.
    eyes = (WEB_VIEWER / "static" / "js" / "eyesOral.js").read_text(encoding="utf-8")
    assert "setPaused(paused)" in eyes, "eyesOral lost its pause seam"
    sched = eyes[eyes.index("  _scheduleBlink() {"):eyes.index("  setTeeth(")]
    assert "_paused" in sched, "blink scheduler ignores pause"
    assert "eyesOral.setPaused" in app, "sim pause never reaches eyesOral"
    pump = app[app.index("function pumpAnchorClip"):app.index("function syncSimPlayButton")]
    assert "isAgentSimulationActive" in pump, "anchor clip pump ignores pause"
    assert "applyClipPose(dt * getSimulationTimeScale())" in pump, (
        "anchor clip pump ignores global speed"
    )
    assert "function stepAnchorClip(dt)" in app, "manual step leaves the anchor clip frozen"
    assert "stepAnchorClip(getSimulationTimeScale() / 60)" in app, "Step +1 skips the anchor clip"
    assert "stepAnchorClip((10 * getSimulationTimeScale()) / 60)" in app, "Skip +10 skips the anchor clip"


def test_world_agent_toggles_present_and_wired():
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="frustum-toggle"' in html, "missing element #frustum-toggle"
    # The Agent-shadows toggle was deleted as a placebo (no shadow pipeline).
    assert 'id="agent-shadows-toggle"' not in html, "removed shadows toggle still present"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "setFrustumCullingEnabled" in app, "frustum toggle not wired"
    assert "setAgentShadowsEnabled" not in app, "removed shadows wiring still in app.js"
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "setAgentShadows" not in sim, "removed shadows method still in sim"


def test_crowd_stats_lod_frustum_spawn_rows_present_and_wired():
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in ("crowd-stats-lodmix", "crowd-stats-lodmoves", "crowd-stats-frustum", "crowd-stats-spawn"):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("crowdStatsLodmix", "crowdStatsLodmoves", "crowdStatsFrustum", "crowdStatsSpawn"):
        assert ref in app, f"stats row not wired: {ref}"
    for behavior in ("wander", "flock", "circle", "scatter"):
        assert f'<option value="{behavior}"' in html, f"missing behavior option {behavior}"


def test_actor_selection_and_toolbar_controls_present():
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in (
        "actor-card",
        "actors-list",
        "crowd-sim-focus",
        "crowd-actor-title",
        "crowd-actor-remove",
        "crowd-actor-behavior",
        "crowd-actor-speed",
        "crowd-actor-scale",
        "crowd-actor-cadence",
        "crowd-actor-lod",
        "crowd-actor-pos",
        "crowd-actor-curspeed",
        "actor-ai-note",
        "actor-gaze-wander",
        "crowd-sim-autolod",
        "crowd-sim-lodcolors",
        "crowd-sim-step",
        "crowd-sim-skip",
        "crowd-sim-speed",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    for gone in ("crowd-sim-actor-inspector", "crowd-actor-color",
                 "active-actor-badge", "eyes-mouth-card"):
        assert f'id="{gone}"' not in html, (
            f"folded into the Actor card, #{gone} should be gone"
        )
    actor_at = html.index('id="actor-card"')
    card_end = html.index("</section>", html.index('id="atab-ai"'))
    for folded in ("parameter-heading", "parameter-slider", "mixamo-play",
                   "eyes-pupil", "eyes-iris",
                   "teeth-fangs", "tongue-roll"):
        at = html.index(f'id="{folded}"')
        assert actor_at < at < card_end, f"#{folded} is not in the Actor card"
    # Old collapsible group summaries are replaced by the actor tab strip.
    for gone_summ in ("Parameters", "Identity", "Face", "Body", "Motion", "Animation"):
        assert f"<summary>{gone_summ}</summary>" not in html, (
            f"{gone_summ} group summary survived the tab refactor"
        )
    for tab in ("pose", "identity", "expression", "eyes", "mouth", "materials",
                "clothing", "animation", "ai"):
        assert f'data-atab="{tab}"' in html, f"missing actor tab: {tab}"
        at = html.index(f'id="atab-{tab}"')
        assert actor_at < at < card_end, f"atab-{tab} is not in the Actor card"
    for mount in ("pose", "expression"):
        at = html.index(f'id="param-mount-{mount}"')
        assert actor_at < at < card_end, f"param-mount-{mount} is not in the Actor card"; assert actor_at < html.index('id="identity-grid"') < card_end and actor_at < html.index('id="identity-reset-all"') < card_end, "identity grid is not in the Actor card"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "refreshActorCard" in app, "Actor card has no refresh function"
    assert "refreshActorInspector" not in app, "stale inspector refresh survived"
    assert "activeActorBadge" not in app, "stale badge ref survived in app.js"
    state = (WEB_VIEWER / "static" / "js" / "state.js").read_text(encoding="utf-8")
    assert "active-actor-badge" not in state, "stale badge ref survived in state.js"


def test_gpu_crowd_modules_present_and_wired():
    for name in ("layouts.js", "kernels.js", "provider.js", "GpuAgent.js"):
        assert (GPU_JS / name).is_file(), f"missing gpu module: {name}"
    sim_dir = WEB_VIEWER / "static" / "js" / "sim"
    assert (sim_dir / "LodPolicy.js").is_file(), "missing LOD policy module"
    assert (sim_dir / "SpatialHash.js").is_file(), "missing spatial hash module"
    sim = (sim_dir / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "GpuCrowd" in sim and "GpuAgent" in sim, (
        "AgentSimulation does not reference the GPU crowd path"
    )
    assert "ensurePool" in sim and "_migrateLodAgents" in sim, (
        "AgentSimulation lost the multi-LOD pool manager"
    )
    provider = (GPU_JS / "provider.js").read_text(encoding="utf-8")
    assert "buildBatchedKernels" in provider, (
        "GPU provider does not build compute kernels"
    )


def test_crowd_stats_panel_present_and_wired():
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in (
        "crowd-stats-details",
        "crowd-stats-headline",
        "crowd-stats-autorefresh",
        "crowd-stats-refresh",
        "crowd-stats-engine",
        "crowd-stats-provider",
        "crowd-stats-pools",
        "crowd-stats-tick",
        "crowd-stats-dispatch",
        "crowd-stats-gpuexec",
        "crowd-stats-body",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "refreshCrowdStats" in app, "app.js never refreshes the crowd stats panel"
    assert "pools × FK+skin" in app, (
        "dispatch row no longer describes the multi-pool batched submit"
    )
    assert "agents × FK+skin" not in app, (
        "dispatch row regressed to the per-agent 2N wording"
    )
    crowd = (WEB_VIEWER / "static" / "js" / "crowd.js").read_text(encoding="utf-8")
    assert "gpuStatus" in crowd and "timings" in crowd, (
        "crowd.js stats do not publish GPU status/timings"
    )


def test_fk_kernel_uses_full_premultiplied_quaternion():
    """Pin the FK w-term fix: TSL bodies only execute on a GPU (see the
    browser probe), so this pins the exact expression instead of executing it."""
    src = (GPU_JS / "kernels.js").read_text(encoding="utf-8")
    assert "const qw = pw.mul(ew).sub(px.mul(ex)).sub(py.mul(ey)).sub(pz.mul(ez))" in src, (
        "FK kernel lost the premultiplied quaternion w term"
    )
    for coord in ("x2", "y2", "z2"):
        assert f"qw.mul({coord})" in src, (
            f"rotation matrix no longer uses qw for {coord}"
        )
        assert f"ew.mul({coord})" not in src, (
            f"rotation matrix regressed to euler-only w for {coord}"
        )


def test_mixamo_card_present_and_wired():
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in (
        "mixamo-count",
        "mixamo-filter",
        "mixamo-refresh",
        "mixamo-clip",
        "mixamo-play",
        "mixamo-stop",
        "mixamo-stop-all",
        "mixamo-loop",
        "mixamo-spine",
        "mixamo-spine-val",
        "mixamo-global",
        "mixamo-status",
        "mixamo-meta-take",
        "mixamo-meta-duration",
        "mixamo-meta-tracks",
        "mixamo-meta-bones",
        "mixamo-meta-unmapped",
        "mixamo-meta-hips",
        "mixamo-meta-loop",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    anim_at = html.index('id="atab-animation"')
    ai_at = html.index('id="atab-ai"')
    play_at = html.index('id="mixamo-play"')
    assert anim_at < play_at < ai_at, "Mixamo should live in the Actor Animation tab"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("mixamoPlay", "playMixamoOnSelected", "loadMixamoClip",
                "mixamoSpine", "setSpineBoost", "mixamoGlobal", "setGlobalMatch",
                "setMixamoTargetRig"):
        assert ref in app, f"mixamo not wired: {ref}"


def test_benchmark_card_present_and_wired():
    """Benchmark details in the Performance tab + bench.js runner wiring."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in (
        "bench-details",
        "bench-headline",
        "bench-device",
        "bench-agents",
        "bench-seconds",
        "bench-lod",
        "bench-run",
        "bench-status",
        "bench-body",
        "bench-export",
        "bench-import",
        "bench-clear",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    perf_at = html.index('id="wtab-performance"')
    crowd_at = html.index('id="wtab-crowd"')
    bench_at = html.index('id="bench-details"')
    assert perf_at < bench_at < crowd_at, "Benchmark should live in the Performance tab"
    assert (WEB_VIEWER / "static" / "js" / "bench.js").is_file(), "missing bench.js"
    bench = (WEB_VIEWER / "static" / "js" / "bench.js").read_text(encoding="utf-8")
    for ref in ("runBenchmark", "summarizeSamples", "detectEngine", "saveRun", "importRuns"):
        assert ref in bench, f"bench.js missing: {ref}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("wireBenchmark", "renderBenchTable", "runBenchmark"):
        assert ref in app, f"benchmark not wired: {ref}"


def test_sim_clear_resets_crowd_eyes():
    """Removal paths must sync the eye instances when no tick will (clear
    stops the sim; single/CPU removal while paused) — else floating eyes."""
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "crowdEyes.update(this.agents, this.simTime, performance.now());" in sim, (
        "eye sync helper lost its update call"
    )
    assert sim.count("this._syncEyesAfterRemoval();") == 3, (
        "clear/removeAgent/clearCpuAgents must all sync crowd eyes after removal"
    )


def test_canvas_click_select_wired():
    """Canvas click builds an NDC vector and picks an actor (missing THREE
    import once broke every click with a ReferenceError)."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert 'import * as THREE from "three"' in app, "app.js lost its THREE import"
    for ref in ("pickAgentAtNdc", "new THREE.Vector2", "renderActorsList"):
        assert ref in app, f"click-select not wired: {ref}"


def test_camera_buttons_in_viewport_overlay():
    """View/fit/projection buttons live in the viewport corner toolbar."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "<summary>Camera</summary>" not in html, "Camera card should be an overlay now"
    vp_at = html.index('id="viewport"')
    bar_at = html.index('<div class="statusbar">')
    for snippet in ('data-view="front"', 'data-view="face"', 'id="camera-reset"',
                    'id="projection-toggle"'):
        at = html.index(snippet)
        assert vp_at < at < bar_at, f"{snippet} is not in the viewport overlay"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("viewButtons", "cameraReset", "projection", "activeActorFocus",
                "cameraNear", "cameraFar", "cameraFov", "setCameraParams"):
        assert ref in app, f"camera control not wired: {ref}"
    scene = (WEB_VIEWER / "static" / "js" / "scene.js").read_text(encoding="utf-8")
    assert "lastFocus" in scene, "fitCamera lost its selection framing target"
    for ref in ("setCameraParams", "syncCameraInputs"):
        assert ref in scene, f"camera param helper missing: {ref}"
    assert "perspectiveCamera.fov / 2" in scene, "fitCamera must frame with the live FOV"


def test_camera_tab_and_effects_wired():
    """Camera tab owns lens/projection/follow/effects; menubar restores panes."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    cam_at = html.index('id="wtab-camera"')
    perf_at = html.index('id="wtab-performance"')
    for elem_id in (
        "camera-near", "camera-far", "camera-fov",
        "camera-proj-persp", "camera-proj-ortho", "camera-follow",
        "fx-dof", "fx-dof-focus", "fx-dof-range", "fx-dof-bokeh",
        "fx-bloom", "fx-bloom-strength", "fx-bloom-threshold",
    ):
        at = html.index(f'id="{elem_id}"')
        assert cam_at < at < perf_at, f"#{elem_id} is not in the Camera tab"
    for elem_id in ("menubar", "menu-panels", "dlg-shortcuts", "dlg-about"):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    dock = (WEB_VIEWER / "static" / "js" / "dock.js").read_text(encoding="utf-8")
    assert '"camera", "Camera"' in dock, "camera pane missing from the dock"
    scene = (WEB_VIEWER / "static" / "js" / "scene.js").read_text(encoding="utf-8")
    for ref in ("PostProcessing", "DepthOfFieldNode", "BloomNode",
                "setProjection", "syncProjectionUI", "setFollow",
                "setEffect", "effectsActive"):
        assert ref in scene, f"camera/effects helper missing: {ref}"
    menu = (WEB_VIEWER / "static" / "js" / "menubar.js").read_text(encoding="utf-8")
    for ref in ("homeRefFor", "initMenubar"):
        assert ref in menu, f"menubar lost: {ref}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "initMenubar({" in app, "menubar not wired in app.js"


def test_selection_ring_honest_at_boot():
    """The ring derives its initial state from the selection (Main), so the
    boot frame already shows what the dropdown/cards claim is selected."""
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    init_at = sim.index("_initSelectionRing() {")
    update_at = sim.index("this._updateSelectionRing();", init_at)
    end_at = sim.index("_updateSelectionRing() {", init_at)
    assert update_at < end_at, "ring init must sync from the current selection"


def test_selection_drives_camera_and_spawn_keeps_selection():
    """Genuine selection changes jump the orbit camera; spawn never steals
    the selection (and so never yanks the camera)."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "lastCamFocusId" in app, "camera no longer follows selection changes"
    assert "fitCamera(state.lastView, activeActorFocus(state.lastView))" in app, (
        "selection change lost its camera jump"
    )
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "selectAgent(this.agents[0].id)" not in sim, (
        "spawn must not auto-select an agent"
    )


def test_skeleton_overlays_follow_selection():
    """Overlay lines, tooltip buffer, and joint spheres track the active
    actor (agents included), not just Main."""
    actor = (WEB_VIEWER / "static" / "js" / "actor.js").read_text(encoding="utf-8")
    assert "export function syncDisplayedSkeleton" in actor, "missing skeleton retarget"
    assert "registerSphereSync" in actor, "sphere refresh lost its registration"
    rig = (WEB_VIEWER / "static" / "js" / "rig.js").read_text(encoding="utf-8")
    assert "syncDisplayedSkeleton()" in rig, "pose eval lost its display sync"
    assert "updateSkeleton(state.jointPositions)" not in rig, (
        "pose eval regressed to the Main-only display write"
    )
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "syncDisplayedSkeleton()" in app, "selection change lost its skeleton sync"
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "_trackSelectionOverlays" in sim, "tick lost its overlay tracking"
    assert sim.count("poseProbe(") == 2, (
        "probe must pose for the active agent on tick and on paused select"
    )
    gizmo = (WEB_VIEWER / "static" / "js" / "gizmo.js").read_text(encoding="utf-8")
    assert "registerSphereSync(syncJointSpheres)" in gizmo, "spheres not registered"


def test_joint_spheres_constant_screen_size():
    """Joint spheres hold ~10px on screen (world size from true pixels),
    rescaled on every camera change."""
    gizmo = (WEB_VIEWER / "static" / "js" / "gizmo.js").read_text(encoding="utf-8")
    assert "SPHERE_SCREEN_PX" in gizmo, "spheres lost their screen-size target"
    assert "setScalar((SPHERE_SCREEN_PX * wpp)" in gizmo, "spheres lost pixel scaling"
    assert 'addEventListener("change", onCameraChanged)' in gizmo, (
        "spheres never rescale on camera moves"
    )


def test_lod_colors_include_main():
    """LOD paint covers Main (server LOD, 0 shares 1) and the color picker
    keeps the stored color it restores from."""
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "bodyMaterial.color.copy(anchor.color)" in sim, "Main lost its paint restore"
    assert "bodyMaterial.color.copy(cached)" in sim, "Main lost its LOD paint"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "actor.setColor(elements.bodyColor.value)" in app, "picker lost its unified path"
    assert "bodyMaterial.color.set(elements.bodyColor.value)" in app, "picker lost boot fallback"
    actor = (WEB_VIEWER / "static" / "js" / "actor.js").read_text(encoding="utf-8")
    assert "this.material.color.copy(this.color)" in actor, "setColor lost its material write"


def test_instanced_pupils_and_lod1_mouths():
    """Pupils are per-instance for every agent; inner mouths exist only on
    LOD1 GPU agents; gaze barely moves the lids."""
    eyes = (WEB_VIEWER / "static" / "js" / "crowdEyes.js").read_text(encoding="utf-8")
    assert 'TSL.attribute("aPupil")' in eyes, "crowd eyes lost the pupil attribute"
    assert "this._pupilArray[n] = pupil" in eyes, "crowd eyes never fill pupils"
    assert "headMatrixFor" in eyes and "mouthFrame" in eyes, "mouth frame access lost"
    actor = (WEB_VIEWER / "static" / "js" / "actor.js").read_text(encoding="utf-8")
    assert "this.pupil = options.pupil ?? 0.45" in actor, "actors lost their pupil field"
    mouth_path = WEB_VIEWER / "static" / "js" / "crowdMouth.js"
    assert mouth_path.is_file(), "missing crowdMouth.js"
    mouth = mouth_path.read_text(encoding="utf-8")
    assert "agent.lod !== 1" in mouth, "mouths lost their LOD1 gate"
    assert "crowdEyes.headMatrixFor" in mouth, "mouths lost the shared head matrices"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "target.pupil = v" in app, "pupil slider lost its per-agent path"
    assert "crowdMouth.update(" in app, "paused refresh lost mouth updates"
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "crowdMouth.update(" in sim, "sim tick lost mouth updates"
    oral = (WEB_VIEWER / "static" / "js" / "eyesOral.js").read_text(encoding="utf-8")
    assert "pupil = null" in oral, "eye material lost pupil injection"
    assert "* 0.3;" in oral and "* 0.2;" in oral, "gaze-lid coupling regressed"


def test_list_skeleton_selection_both_ways():
    """List clicks reveal the driven joint; orb clicks jump the list back."""
    gizmo = (WEB_VIEWER / "static" / "js" / "gizmo.js").read_text(encoding="utf-8")
    assert "export function revealJointForParam" in gizmo, "list→skeleton reveal lost"
    assert "export function strongestJointForParam" in gizmo, "joint mapping core lost"
    assert "export function jumpToJointParam" in gizmo, "skeleton→list jump lost"
    assert "export function paramJointMode" in gizmo, "dominant-mode mapping lost"
    assert "preferredMode" in gizmo, "selectJoint lost its mode preference"
    assert "scalarOnly" in gizmo, "scale tool lost its scalar-only filter"
    assert 'if (obj.name !== "XYZ") obj.visible = false;' in gizmo, (
        "scale tool shows per-axis handles again"
    )
    assert "export function axisThroughPoints" in gizmo, "connecting-axis core lost"
    assert "affectedJointsMoveQuat" in gizmo, "move tool lost its affected-joint frame"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    select = app[app.index("function selectParameter"):app.index("function selectParameter") + 900]
    assert "revealJointForParam(index)" in select, "selectParameter lost the joint reveal"
    assert "selectParameter(Number(item.dataset.index))" in app, (
        "list clicks bypass selectParameter"
    )


def test_joint_orbs_follow_any_tab_and_anchor_clips():
    """Hinge orbs are visible + clickable on every actor tab (not pose-only),
    and the anchor clip pump refreshes the display buffer every frame so the
    overlay tracks Main's dance instead of freezing at play."""
    gizmo = (WEB_VIEWER / "static" / "js" / "gizmo.js").read_text(encoding="utf-8")
    vis = gizmo[gizmo.index("export function updateSphereVisibility"):
                gizmo.index("function onPoseEvaluated")]
    assert "currentGroup" not in vis, "orb visibility re-gated to one tab"
    up = gizmo[gizmo.index("function onPointerUp"):
               gizmo.index("TransformControls can only hide")]
    assert "currentGroup" not in up, "orb clicks re-gated to one tab"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    pump = app[app.index("function pumpAnchorClip"):app.index("function syncSimPlayButton")]
    assert "syncDisplayedSkeleton()" in pump, (
        "anchor clip pump must refresh the display buffer per frame"
    )


def test_locked_params_reject_writes():
    """Locked params (zero-span or backend-managed) refuse every UI write:
    slider, numeric commit, all resets, and the actor choke point."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert app.count("isLocked(parameter)") >= 3, "slider/commit lost lock guards"
    assert "!isLocked(item)" in app, "resets lost their lock filter"
    actor = (WEB_VIEWER / "static" / "js" / "actor.js").read_text(encoding="utf-8")
    assert "isLocked(param)" in actor, "setParamValue lost its lock guard"
    gizmo = (WEB_VIEWER / "static" / "js" / "gizmo.js").read_text(encoding="utf-8")
    assert "isLocked(param)" in gizmo, "gizmo primaries lost their lock filter"


def test_expression_vectors_present_and_wired():
    """Min-to-max travel lines per expression + mirror-pair hover highlight."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert html.index('id="skeleton-toggle"') < html.index('id="expr-vectors-toggle"'), (
        "vectors toggle should sit with the overlay toggles"
    )
    assert (WEB_VIEWER / "static" / "js" / "exprVectors.js").is_file(), (
        "missing exprVectors.js"
    )
    vec = (WEB_VIEWER / "static" / "js" / "exprVectors.js").read_text(encoding="utf-8")
    for ref in ("mirrorPairs", "strongestVertex", "initExprVectors", "pickExprVector"):
        assert ref in vec, f"exprVectors.js missing: {ref}"
    assert "boneInverses" in vec, "lines lost their bind-space correction"
    assert "bs.totalBs" in vec and "bs.idCount" in vec, "delta layout regressed to metadata"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("initExprVectors", "pickExprVector", "setExprVectorHover"):
        assert ref in app, f"vectors not wired: {ref}"
    params = (WEB_VIEWER / "static" / "js" / "params.js").read_text(encoding="utf-8")
    assert "setExprVectorSelected" in params, "selection highlight not wired"
    state = (WEB_VIEWER / "static" / "js" / "state.js").read_text(encoding="utf-8")
    assert "expr-vectors-toggle" in state, "toggle element not in state.js"


def test_expr_vectors_fat_lines_and_tooltip():
    """Fat lines (WebGPU node variant) + click-select with transform tooltip."""
    vec = (WEB_VIEWER / "static" / "js" / "exprVectors.js").read_text(encoding="utf-8")
    assert "lines/webgpu/LineSegments2.js" in vec, "not the WebGPU fat-line class"
    assert "Line2NodeMaterial" in vec, "fat lines need the TSL node material"
    assert "LineBasicMaterial" not in vec, "thin-line material still present"
    assert "linewidth: LINE_WIDTH_PX" in vec, "pixel width not configured"
    assert "this._ray.camera = camera" in vec, "fat-line pick misses raycaster.camera"
    assert "faceIndex" in vec, "fat-line pick must read the segment faceIndex"
    assert "hits[0].index / 2" not in vec, "thin-line segment math still present"
    assert "describe(" in vec and "travelMm" in vec, "tooltip detail missing"
    rebuild = vec[vec.index("rebuild() {"):]
    box_at = rebuild.index("computeBoundingBox()")
    sph_at = rebuild.index("computeBoundingSphere()")
    assert box_at < sph_at, (
        "rebuild must refresh the box before the sphere (sphere reuses box center)"
    )
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("describeExprVector", "setExprVectorSelected", "showCanvasTooltip"):
        assert ref in app, f"line click-select not wired: {ref}"
    assert "selectParameter(hit)" in app, "3D line select does not sync the list"
    assert "exprVectorWorldSegment" in vec, "drag scrub lost its world-segment query"
    assert "exprVectorWorldSegment" in app, "drag scrub not wired in app.js"
    for ref in ("updateLineDrag", "writeExprDragValue", "commitExprDragValue"):
        assert ref in app, f"line scrub drag missing: {ref}"
    assert "distanceSqToSegment" in app, "scrub lost its ray-to-segment mapping"
    assert "controls.enabled = false" in app, "line drag never captures OrbitControls"
    assert 'syncBlendshapeServer("expression", i, param.value)' in app, (
        "drag release lost its slider-change parity"
    )
    assert "deltaMm" in vec, "translation vector missing from describe()"
    math = (WEB_VIEWER / "static" / "js" / "exprVectorMath.js").read_text(encoding="utf-8")
    assert "formatDeltaMm" in math, "translation formatter missing"
    assert "translation ${formatDeltaMm(d.deltaMm)} mm" in app, (
        "translation row missing from the tooltip"
    )
    assert "pinCanvasTooltip()" in app, "single click does not pin the tooltip"
    assert "dismissCanvasTooltip" in app, "no dismiss path wired in app.js"
    assert 'if (event.key === "Escape") {' in app, "Escape does not dismiss the tooltip"
    scene = (WEB_VIEWER / "static" / "js" / "scene.js").read_text(encoding="utf-8")
    assert "export function showCanvasTooltip" in scene, "shared tooltip missing"
    assert "export function pinCanvasTooltip" in scene, "tooltip pin missing"
    assert "export function dismissCanvasTooltip" in scene, "tooltip dismiss missing"
    assert "if (canvasTooltipPinned) return;" in scene, "hover ignores the pin"


def test_export_and_global_lod_ui_removed():
    """Export UI and the global LOD dropdown were deliberately deleted.

    Main stays at the server default (LOD 1); backend /api/export/* and
    /api/configure endpoints remain for the server test-suite.
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in (
        "export-format",
        "export-ground-toggle",
        "export-ground-state",
        "export-button",
        "lod-select",
        "lod-summary",
        "lod-help",
    ):
        assert f'id="{elem_id}"' not in html, f"removed UI still present: #{elem_id}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for token in ("exportCurrentModel", "configureModel", "setCrowdWireframe", "snapToGround"):
        assert token not in app, f"removed UI ref still in app.js: {token}"
    for rel in ("static/js/ui.js", "static/js/state.js", "static/js/crowd.js"):
        src = (WEB_VIEWER / rel).read_text(encoding="utf-8")
        for token in ("exportFormat", "exportButton", "exportGround", "#lod-select", "setCrowdWireframe"):
            assert token not in src, f"removed UI ref still in {rel}: {token}"
    css = (WEB_VIEWER / "static" / "styles.css").read_text(encoding="utf-8")
    for token in (".export-option", ".export-button", ".lod-field", ".lod-summary", "model-note"):
        assert token not in css, f"removed UI rule still in styles.css: {token}"


def test_world_pane_tabs_and_panels():
    """World sidepane: tab strip + world/performance/crowd/actors panels."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "<summary>Appearance</summary>" not in html, "Appearance was renamed to World"
    strip = html.index('id="world-tabs"')
    panels = {name: html.index(f'id="wtab-{name}"') for name in ("world", "performance", "crowd", "actors")}
    assert strip < min(panels.values()), "world tab strip must lead its panels"
    assert panels["world"] < panels["performance"] < panels["crowd"] < panels["actors"], (
        "world panels out of order"
    )
    world = html[panels["world"]:panels["performance"]]
    assert 'id="background-color"' in world, "scene globals escaped the World tab"
    toolbar = html[html.index('class="viewport-toolbar"'):html.index('id="joint-toolbar"')]
    for elem_id in ("grid-toggle", "axes-toggle", "dev-tools"):
        assert f'id="{elem_id}"' in toolbar, f"#{elem_id} is not in the viewport toolbar"
    perf = html[panels["performance"]:panels["crowd"]]
    for elem_id in ("correctives-toggle", "frustum-toggle", "bench-details"):
        assert f'id="{elem_id}"' in perf, f"performance option escaped: #{elem_id}"
    crowd = html[panels["crowd"]:panels["actors"]]
    for elem_id in ("crowd-card", "crowd-sim-spawn", "crowd-sim-summary"):
        assert f'id="{elem_id}"' in crowd, f"crowd control escaped: #{elem_id}"
    assert 'id="crowd-stats-details"' not in crowd, "crowd stats belong in telemetry"
    assert 'id="bench-details"' not in crowd, "benchmark belongs in Performance"
    actors = html[panels["actors"]:]
    assert 'id="actors-list"' in actors, "actors list missing"
    assert 'id="crowd-sim-actor-select"' not in html, "actor dropdown should be replaced by the list"
    body = html.index('id="telemetry-body"')
    tele = html[body:html.index("</main>", body)]
    shell = html.index('class="sidebar"')
    assert shell < html.index('class="world-pane"') < html.index('class="model-settings"'), (
        "sidebar shell must hold the world section above the actor section"
    )
    assert 'id="perf-card"' in tele and 'id="crowd-stats-details"' in tele, (
        "telemetry strip lost the perf card or crowd stats"
    )
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "switchWorldTab" in app and "mhr.worldTab" in app, "world tabs not wired"
    assert "telemetryCollapse" in app, "telemetry collapse not wired"
    assert "renderActorsList" in app, "actors list render missing"
    assert 'elements.actorsList.addEventListener("click"' in app, "actors list click not wired"
    assert "refreshActorDropdown" not in app, "dropdown refresh should be replaced"


def test_clone_and_instance_tiers_removed():
    """Clone/instance stress tiers were deleted; agents are the only crowd."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    for snippet in ("Normal copies", "Instanced meshes", "crowd-clone-", "crowd-inst-"):
        assert snippet not in html, f"removed tier UI still present: {snippet}"
    for rel in ("static/app.js", "static/js/crowd.js", "static/js/state.js"):
        src = (WEB_VIEWER / rel).read_text(encoding="utf-8")
        for token in ("spawnClones", "spawnInstances", "CROWD_LIMITS", "takeAutoDirty", "rebakeInstances"):
            assert token not in src, f"removed tier ref still in {rel}: {token}"


def test_eyes_mouth_card_present_and_wired():
    """Eyes & Mouth detail controls live in the Actor card Face group."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="eyes-mouth-card"' not in html, (
        "Eyes card should be folded into the Actor card"
    )
    # The Eyes & mouth master toggle was removed: all eyes/mouths stay on.
    for elem_id in ("eyes-toggle", "eyes-toggle-state"):
        assert f'id="{elem_id}"' not in html, f"removed toggle still present: #{elem_id}"
    app0 = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "elements.eyesToggle" not in app0, "removed toggle still wired"
    state0 = (WEB_VIEWER / "static" / "js" / "state.js").read_text(encoding="utf-8")
    assert 'querySelector("#eyes-toggle")' not in state0, "removed toggle still queried"
    for elem_id in (
        "eyes-autoblink",
        "eyes-autoblink-state",
        "eyes-blink",
        "gaze-yaw",
        "gaze-yaw-val",
        "gaze-pitch",
        "gaze-pitch-val",
        "eyes-pupil",
        "eyes-pupil-val",
        "eyes-iris",
        "mouth-open",
        "mouth-open-val",
        "teeth-count",
        "teeth-crooked",
        "teeth-crooked-val",
        "teeth-gap",
        "teeth-gap-val",
        "teeth-overbite",
        "teeth-overbite-val",
        "teeth-fangs",
        "teeth-fangs-val",
        "tongue-out",
        "tongue-out-val",
        "tongue-curl",
        "tongue-curl-val",
        "tongue-width",
        "tongue-width-val",
        "tongue-roll",
        "tongue-roll-val",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("eyesOral", "wireEyesMouth", "setGaze", "setTeeth", "setTongue"):
        assert ref in app, f"eyes & mouth not wired: {ref}"
    for elem_id in (
        "crowd-eyes-toggle",
        "crowd-eyes-toggle-state",
        "crowd-blink-toggle",
        "crowd-blink-toggle-state",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    for ref in ("crowdEyes", "crowdEyesToggle", "crowdBlinkToggle"):
        assert ref in app, f"crowd eyes not wired: {ref}"
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "crowdEyes.update(" in sim, "sim tick never updates crowd eyes"
    crowd = (WEB_VIEWER / "static" / "js" / "crowdEyes.js").read_text(encoding="utf-8")
    assert "InstancedMesh" in crowd, "crowd eyes lost instancing"
    assert "eyesClosed_L" in crowd, "crowd eyes lost the GPU blink channel"
    eyes_at = html.index('id="atab-eyes"')
    mouth_at = html.index('id="atab-mouth"')
    materials_at = html.index('id="atab-materials"')
    for face_id in ("gaze-yaw", "eyes-blink", "eyes-autoblink",
                    "eyes-pupil", "eyes-iris"):
        face_at = html.index(f'id="{face_id}"')
        assert eyes_at < face_at < mouth_at, f"#{face_id} is not in the Eyes tab"
    for face_id in ("mouth-open",
                    "teeth-count", "teeth-fangs",
                    "tongue-out", "tongue-roll"):
        face_at = html.index(f'id="{face_id}"')
        assert mouth_at < face_at < materials_at, f"#{face_id} is not in the Mouth tab"


def test_anchor_renders_under_actor_group():
    """Main body + garments parent under the anchor group so Scale works."""
    rig = (WEB_VIEWER / "static" / "js" / "rig.js").read_text(encoding="utf-8")
    assert "actor0.group.add(rootBone, skinnedMesh)" in rig, (
        "Main meshes escaped the actor group again (Scale goes dead)"
    )
    assert "scene.add(actor0.group)" in rig, "anchor group never joins the scene"
    actor = (WEB_VIEWER / "static" / "js" / "actor.js").read_text(encoding="utf-8")
    setter = actor[actor.index("setScale(scale) {"):]
    setter = setter[:setter.index("\n  }\n")]
    assert "syncDisplayedSkeleton()" in setter, (
        "setScale leaves the skeleton lines at the old scale"
    )
    assert "state.skinnedMesh.parent?.remove(state.skinnedMesh)" in rig, (
        "re-rig removal assumes scene-root parenting"
    )
    cloth = (WEB_VIEWER / "static" / "js" / "clothing.js").read_text(encoding="utf-8")
    assert 'actorManager.getActor(0)?.group' in cloth, (
        "garments escaped the anchor group (Scale tears clothes off)"
    )
    assert "this.mesh.parent?.remove(this.mesh)" in cloth, (
        "garment clear assumes scene-root parenting"
    )


def test_clothing_card_present_and_wired():
    """Garment controls live in the Actor card Body group (card was folded)."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="cloth-card"' not in html, "Clothing card should be folded into Actor"
    for elem_id in (
        "cloth-toggle",
        "cloth-toggle-state",
        "cloth-thickness",
        "cloth-thickness-val",
        "cloth-refit",
        "cloth-color",
        "cloth-shorts-toggle",
        "cloth-shorts-toggle-state",
        "cloth-shorts-color",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("shirt", "shorts", "wireClothing", "setThickness", "clothRefit"):
        assert ref in app, f"clothing not wired: {ref}"
    module = (WEB_VIEWER / "static" / "js" / "clothing.js").read_text(encoding="utf-8")
    assert "SkinnedMesh" in module, "shirt lost its skinned bind"
    assert "bind(state.skeleton, body.bindMatrix)" in module, (
        "shirt bind must pass an explicit matrix (see SkinnedMesh.bind)"
    )


def test_skin_and_panorama_present_and_wired():
    """Procedural skin body material + 360/HDRI background controls."""
    scene = (WEB_VIEWER / "static" / "js" / "scene.js").read_text(encoding="utf-8")
    assert "createSkinMaterial(elements.bodyColor.value)" in scene, (
        "body material is not the procedural skin"
    )
    skin = (WEB_VIEWER / "static" / "js" / "skin.js").read_text(encoding="utf-8")
    assert "materialColor" in skin, "skin lost its per-material tint node"
    assert "MeshStandardNodeMaterial" in skin, "skin is not a node material"
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in (
        "background-file",
        "background-env",
        "background-env-val",
        "hdri-grid",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("loadBackgroundPanorama", "clearBackgroundPanorama", "setEnvironmentIntensity"):
        assert ref in app, f"panorama not wired: {ref}"
    assert "hdri-upload" in app, "HDRI grid lost its upload tile"
    for ref in ("/api/hdri/list", "/api/hdri/thumb/", "/api/hdri/file/"):
        assert ref in app, f"HDRI grid not wired: {ref}"
    assert "EquirectangularReflectionMapping" in scene, (
        "panorama does not set an equirect mapping"
    )
    assert "EXRLoader" in scene, "panorama cannot load .exr environments"
    module = (WEB_VIEWER / "static" / "js" / "eyesOral.js").read_text(encoding="utf-8")
    assert "MeshBasicNodeMaterial" in module, (
        "eye shader is not a TSL node material"
    )
    assert "ShaderMaterial" not in module.replace("MeshBasicNodeMaterial", ""), (
        "GLSL ShaderMaterial cannot run on WebGPURenderer"
    )
    for token in ("c_head", "c_jaw", "l_eye", "r_eye", "jawDrop", "eyesClosed_L"):
        assert token in module, f"eyesOral lost its rig anchor: {token}"


def test_face_view_follows_posed_head_rotation():
    """Face camera aims along the posed face-forward, not world front."""
    scene = (WEB_VIEWER / "static" / "js" / "scene.js").read_text(encoding="utf-8")
    assert "direction: faceDirection(headPos)" in scene, (
        "faceMetrics lost its posed-facing direction"
    )
    assert "metrics.direction ?? VIEW_DIRECTIONS.front" in scene, (
        "fitCamera face branch ignores the posed facing"
    )
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    focus = app[app.index("function activeActorFocus"):app.index("function setDisabledWithReason")]
    assert "syncDisplayedSkeleton();" in focus and "return null;" in focus, (
        "agent face view must measure live joints instead of a fixed-height box"
    )
    assert "1.62" not in focus, "fixed head-height guess is still framing agent faces"
    changed = app[app.index("function onActiveActorChanged"):app.index("function refreshSimMetrics")]
    assert changed.index("syncDisplayedSkeleton();") < changed.index("fitCamera("), (
        "camera refit must run after the skeleton sync on actor switch"
    )




def test_actor_tabs_wired_and_param_component_shared():
    """Actor tab strip switches panels; one param component is reparented."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="actor-tabs"' in html, "actor tab strip missing"
    assert html.count("data-atab=") == 9, "actor strip should hold 9 tabs"
    assert 'id="param-component"' in html, "shared param component missing"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "function switchActorTab" in app, "switchActorTab missing"
    assert 'localStorage.getItem("mhr.actorTab")' in app, "actor tab not restored"
    assert 'localStorage.setItem("mhr.actorTab", next)' in app, "actor tab not persisted"
    assert "param-mount-" in app, "param mounts not referenced"
    assert "elements.paramComponent" in app, "shared component not reparented"
    assert "groupTabs" not in app, "stale groupTabs ref survived in app.js"
    params = (WEB_VIEWER / "static" / "js" / "params.js").read_text(encoding="utf-8")
    assert "groupTabs" not in params, "selectGroup still touches tab DOM"
    state = (WEB_VIEWER / "static" / "js" / "state.js").read_text(encoding="utf-8")
    assert 'querySelector("#actor-tabs")' in state, "actorTabs not queried"
    assert 'querySelector("#param-component")' in state, "paramComponent not queried"


def test_agent_console_present_and_wired():
    """Floating agent console: panel markup + dispatcher bindings + console plug."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in (
        "agent-panel",
        "agent-head",
        "agent-status",
        "agent-collapse",
        "agent-body",
        "agent-settings",
        "agent-key",
        "agent-model",
        "agent-url",
        "agent-actions",
        "agent-actions-log",
        "agent-log",
        "agent-form",
        "agent-input",
        "agent-send",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    panel = html.index('id="agent-panel"')
    assert panel > html.index("</main>"), "agent console should float outside the layout grid"
    for mod in ("actions.js", "agent.js", "agentPanel.js"):
        assert (WEB_VIEWER / "static" / "js" / mod).is_file(), f"missing agent module: {mod}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("createDispatcher", "createActionLog", "initAgentPanel", "createAgentBindings",
                "getAgentSnapshot", "setActiveGaze", "window.mhr"):
        assert ref in app, f"agent console not wired: {ref}"
    assert "initAgentPanel({" in app, "console never initialized with dispatcher"
    crowd = (WEB_VIEWER / "static" / "js" / "crowd.js").read_text(encoding="utf-8")
    for ref in ("export function listAgents", "export function getShowLodColors",
                "export function setOneAgentBehavior"):
        assert ref in crowd, f"crowd.js lost agent façade: {ref}"
    panel_js = (WEB_VIEWER / "static" / "js" / "agentPanel.js").read_text(encoding="utf-8")
    for ref in ("runAgentTurn", "requestApproval", "mhr.agentKey", "agent-msg",
                "agent-actions-log", "log.subscribe"):
        assert ref in panel_js, f"console panel lost: {ref}"


def test_dockview_shell_present_and_wired():
    """Phase-2 dockview: vendor delivery, dock root, 16 section panes, persistence."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert '"dockview": "/vendor/dockview/dist/package/main.esm.mjs"' in html, (
        "importmap lost the dockview entry"
    )
    assert '"dockview-core": "/vendor/dockview-core/dist/package/main.esm.mjs"' in html, (
        "importmap lost the dockview-core entry (bare re-export breaks loading)"
    )
    assert 'href="/vendor/dockview/dist/styles/dockview.css"' in html, (
        "dockview stylesheet not linked"
    )
    for elem_id in ("dock-root",):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    for act in ("header", "layout"):
        assert f'data-act="{act}"' in html, f"View menu lost: {act}"
    assert 'id="layout-reset"' not in html and 'id="topbar-toggle"' not in html, (
        "header buttons survived the menubar move"
    )
    assert 'class="status-stats"' in html, "model stats lost their footer home"
    assert (WEB_VIEWER / "static" / "js" / "dock.js").is_file(), "missing dock.js"
    dock = (WEB_VIEWER / "static" / "js" / "dock.js").read_text(encoding="utf-8")
    for ref in ("DockviewComponent", "themeDark", "createComponent", "fromJSON",
                "toJSON", "onDidLayoutChange", "mhr.dockLayout", "is-docked",
                "singleTabMode", "onPanelActivated", "onDidActivePanelChange",
                "resetDockLayout",
                "dockApi"):
        assert ref in dock, f"dock shell lost: {ref}"
    for pane in ("viewport", "world", "performance", "crowd", "actors", "pose",
                 "identity", "expression", "eyes", "mouth", "materials",
                 "clothing", "animation", "ai", "telemetry", "crowdstats"):
        assert f'id: "{pane}"' in dock, f"default layout lost pane: {pane}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "actor-selected" in dock and ".statusbar" in dock, (
        "dock must rescue the actor header + status bar before dropping main.workspace"
    )
    assert 'await import("./js/dock.js")' in app, "dock boot is not a guarded dynamic import"
    assert "initDock({" in app, "dock boot lost the onPanelActivated callback"
    for ref in ("showParamGroup", "showActorTab", "getPanel("):
        assert ref in app, f"dock/param integration lost: {ref}"
    css = (WEB_VIEWER / "static" / "styles.css").read_text(encoding="utf-8")
    assert '#dock-root [id^="wtab-"]' in css, "moved world sections lost dock fill rules"
    assert '#dock-root [id^="atab-"]' in css, "moved actor sections lost dock fill rules"
    server = (WEB_VIEWER / "server.py").read_text(encoding="utf-8")
    assert "/vendor/dockview/<path:filename>" in server, "vendor route missing"
    assert "DOCKVIEW_ROOT" in server, "vendor root missing"
    assert "/vendor/dockview-core/<path:filename>" in server, "core vendor route missing"
    assert "DOCKVIEW_CORE_ROOT" in server, "core vendor root missing"
    assert '"text/javascript" if filename.endswith((".js", ".mjs"))' in server, (
        "vendor route must pin the JS MIME type (.mjs breaks module loading)"
    )
    package = (WEB_VIEWER / "package.json").read_text(encoding="utf-8")
    assert '"dockview"' in package, "dockview not in package.json"


def test_command_menu_present_and_wired():
    """Ctrl+K palette markup, entries-from-catalog, and dispatcher binding."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in (
        "cmdk-open",
        "cmdk-overlay",
        "cmdk-input",
        "cmdk-list",
        "cmdk-hint",
    ):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    menu = (WEB_VIEWER / "static" / "js" / "commandMenu.js").read_text(encoding="utf-8")
    for ref in ("TOOL_SCHEMAS", "buildEntries", "filterEntries", "initCommandMenu"):
        assert ref in menu, f"palette lost: {ref}"
    assert 'event.key.toLowerCase() === "k"' in menu, "palette lost the Ctrl+K binding"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert 'initCommandMenu({' in app, "palette not wired in app.js"
    assert '"palette"' in app, "palette dispatches lack the source tag"
    css = (WEB_VIEWER / "static" / "styles.css").read_text(encoding="utf-8")
    assert "#cmdk-overlay" in css, "palette overlay lost its styles"


def test_pose_presets_present_and_wired():
    """BVH pose grid in the Pose tab, hold/release wiring, license popup."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    pose_at = html.index('id="atab-pose"')
    ident_at = html.index('id="atab-identity"')
    for elem_id in ("pose-grid", "pose-filter", "pose-release", "pose-preset-count"):
        at = html.index(f'id="{elem_id}"')
        assert pose_at < at < ident_at, f"#{elem_id} is not in the Pose tab"
    poses = (WEB_VIEWER / "static" / "js" / "poses.js").read_text(encoding="utf-8")
    for ref in ("MH_CHAINS", "fetchPoseList", "loadPosePreset", "retargetPoseToTheta", "encodePoseId"):
        assert ref in poses, f"poses.js lost: {ref}"
    mixamo = (WEB_VIEWER / "static" / "js" / "mixamo.js").read_text(encoding="utf-8")
    assert "getMixamoTargetRig" in mixamo, "poses lost the shared target rig"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("refreshPoseGrid", "renderPoseTiles", "applyPosePreset", "releasePosePreset",
                "syncPoseGridSelection", "showPoseLicense", "heldPose", "applyHeldPose"):
        assert ref in app, f"pose preset lost: {ref}"
    actor = (WEB_VIEWER / "static" / "js" / "actor.js").read_text(encoding="utf-8")
    assert "_mergeHeldPose" in actor, "actor lost the hold merge"
    gpu = (WEB_VIEWER / "static" / "js" / "gpu" / "GpuAgent.js").read_text(encoding="utf-8")
    assert "applyHeldPose" in gpu, "GPU agents cannot hold presets"
    server = (WEB_VIEWER / "server.py").read_text(encoding="utf-8")
    for ref in ("/api/pose/list", "/api/pose/file/", "/api/pose/thumb/", "list_pose_files",
                "resolve_pose_file", "parse_pose_meta"):
        assert ref in server, f"pose route lost: {ref}"
    css = (WEB_VIEWER / "static" / "styles.css").read_text(encoding="utf-8")
    for ref in (".pose-badge", "#pose-license-pop"):
        assert ref in css, f"pose style lost: {ref}"


def test_reset_buttons_release_pose_overrides():
    """Reset current/group/all drops a held pose or playing clip first.

    Both re-apply every tick, so a pose reset that only tweens param values
    leaves the character visibly stuck (neutral values even trip the
    "already neutral" early return)."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    head = app[app.index("function animateReset"):app.index("function easeOutCubic")]
    assert "releasePosePreset()" in head, "pose reset does not release the held pose"
    assert "stopMixamoOnSelected()" in head, "pose reset does not stop a playing clip"


def test_fitted_teeth_size_wired():
    """Mouth tab grows the fitted arches: slider -> setFittedTeethScale."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in ("teeth-size", "teeth-size-val"):
        assert f'id="{elem_id}"' in html, f"missing element #{elem_id}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "setFittedTeethScale" in app, "teeth size slider not bound"
    oral = (WEB_VIEWER / "static" / "js" / "eyesOral.js").read_text(encoding="utf-8")
    assert "FITTED_TEETH_SCALE" in oral, "fitted teeth lost their size constant"
    assert "setFittedTeethScale" in oral, "fitted teeth lost setFittedTeethScale"


def test_sim_transport_gated_behind_dev_tools():
    """Play/step/skip/speed hide until Developer tools are enabled."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    dev = app[app.index("function applyDevTools"):app.index("function showWelcomeIfFirstRun")]
    for ref in ("crowdSimPlay", "crowdSimStep", "crowdSimSkip", "crowdSimSpeed"):
        assert ref in dev, f"transport not dev-gated: {ref}"


def test_viewport_top_zones_wired():
    """Viewport top: camera left, brand center, capture/scene right, no THREE badge."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "engine-badge" not in html, "THREE.JS badge survived"
    left = html[html.index('<div class="viewport-toolbar-left">'):html.index('<div class="viewport-brand"')]
    for snippet in ('data-view="front"', 'data-view="face"', 'id="camera-reset"',
                    'id="projection-toggle"'):
        assert snippet in left, f"{snippet} is not in the left toolbar"
    brand = html[html.index('<div class="viewport-brand"'):html.index('<div class="viewport-toolbar">')]
    assert "neohuman-icon.svg" in brand and "NeoHuman" in brand, "brand overlay incomplete"
    right = html[html.index('<div class="viewport-toolbar">'):html.index('id="joint-toolbar"')]
    for snippet in ('id="screenshot-button"', 'id="grid-toggle"', 'id="dev-tools"'):
        assert snippet in right, f"{snippet} is not in the right toolbar"


def test_dock_single_sidebar_identity_first():
    """Dock default: one right-hand tab group opening on Identity."""
    dock = (WEB_VIEWER / "static" / "js" / "dock.js").read_text(encoding="utf-8")
    assert '"identity", "Identity", "viewport", "right"' in dock, "sidebar not rooted at Identity"
    assert 'getPanel("identity")?.api.setActive()' in dock, "Identity not activated"


def test_identity_wipe_smile_wired():
    """Identity tab offers the one-click clean face; handler resets face state."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="identity-wipe-smile"' in html, "wipe-smile button missing"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "function wipeLandingLook" in app, "wipeLandingLook missing"
    assert 'getElementById("identity-wipe-smile")' in app, "wipe-smile not bound"


def test_tab_intros_present():
    """Every tab opens with a one-paragraph orientation."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    panels = ["wtab-world", "wtab-camera", "wtab-performance", "wtab-crowd",
              "wtab-actors", "atab-pose", "atab-identity", "atab-expression",
              "atab-eyes", "atab-mouth", "atab-materials", "atab-clothing",
              "atab-animation", "atab-ai"]
    starts = [html.index(f'id="{p}"') for p in panels]
    assert starts == sorted(starts), "tab order drifted; update the intro test"
    starts.append(len(html))
    for pid, begin, end in zip(panels, starts, starts[1:]):
        assert html[begin:end].count('class="tab-intro"') == 1, f"#{pid} lacks its intro"


def test_tabs_auto_toggle_overlays():
    """Pose reveals the skeleton, Expression the vectors; leaving restores."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "function syncTabAutoOverlays" in app, "tab overlay sync missing"
    strip = app[app.index("function switchActorTab"):app.index("function showActorTab")]
    assert "syncTabAutoOverlays(next)" in strip, "static strip skips overlay sync"
    assert "syncTabAutoOverlays(id)" in app, "dock activation skips overlay sync"


def test_ai_tab_dev_gated():
    """AI tab hides until Developer tools are enabled (strip + dock)."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert 'const DEV_ATABS = ["ai"]' in app, "AI not listed as a dev actor tab"
    dev = app[app.index("function applyDevTools"):app.index("function showWelcomeIfFirstRun")]
    assert "actorTabs" in dev, "dev toggle ignores the actor strip"
    assert 'switchActorTab("identity")' in dev, "active AI tab not bounced to Identity"
    dock = (WEB_VIEWER / "static" / "js" / "dock.js").read_text(encoding="utf-8")
    panes = [line for line in dock.splitlines() if "DEV_PANES = " in line][0]
    assert '"ai"' in panes, "AI not listed as a dev dock pane"
    assert 'ai: ["identity", "within"]' in dock, "AI has no dock home"


def test_benchmark_url_trigger_wired():
    """?bench=1 autostarts the benchmark with optional overrides."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "function triggerUrlBenchmark" in app, "URL bench trigger missing"
    assert "URLSearchParams" in app, "no query parsing for bench trigger"
    assert "benchRun?.click()" in app, "trigger bypasses the Run button path"
    init = app[app.index("async function initialize"):app.index("const ACTOR_TABS")]
    assert "triggerUrlBenchmark()" in init, "trigger never runs at boot"


def test_benchmark_ramp_mode_wired():
    """Ramp mode grows agents until fps breaks; fixed mode kept."""
    bench = (WEB_VIEWER / "static" / "js" / "bench.js").read_text(encoding="utf-8")
    assert "export async function runRampBenchmark" in bench, "ramp runner missing"
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in ("bench-mode", "bench-step", "bench-interval", "bench-minfps"):
        assert f'id="{elem_id}"' in html, f"ramp control missing: #{elem_id}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "runRampBenchmark(" in app, "Run button cannot reach ramp mode"
    trig = app[app.index("function triggerUrlBenchmark"):app.index("function wireBenchmark")]
    assert '"ramp"' in trig, "URL trigger ignores ramp mode"


def test_mixamo_stop_restores_base():
    """Stopping a clip restores the authored base (anchor has no tick)."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    body = app[app.index("function stopActorClip"):app.index("function jointIndexByName")]
    assert "restoreBasePose" in body, "stop leaves the anchor frozen mid-pose"
    actor = (WEB_VIEWER / "static" / "js" / "actor.js").read_text(encoding="utf-8")
    assert "restoreBasePose()" in actor, "Actor.restoreBasePose missing"
    assert "this.thetaJ.set(this._baseThetaJ)" in actor


def test_mixamo_next_wired():
    """Motion clips offer Next: advance the clip and play it."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="mixamo-next"' in html, "mixamo Next button missing"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "function playNextMixamo" in app, "playNextMixamo missing"
    assert "elements.mixamoNext" in app, "Next button not bound"


def test_noui_flag_wired():
    """?noui hides every overlay for clean scene recording."""
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert '"noui"' in app, "noui param not parsed"
    assert 'classList.add("noui")' in app, "noui class never applied"
    assert 'panel.id !== "viewport"' in app, "dock panels not closed for noui"
    css = (WEB_VIEWER / "static" / "styles.css").read_text(encoding="utf-8")
    assert "body.noui #viewport" in css, "viewport not freed for noui"
    assert "body.noui .viewport-toolbar" in css, "toolbar not hidden for noui"


def test_selection_ring_toggle_wired():
    """Viewport toolbar toggles the selection ring under the active agent."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    right = html[html.index('<div class="viewport-toolbar">'):html.index('id="joint-toolbar"')]
    assert 'id="ring-toggle"' in right, "ring toggle not in the viewport toolbar"
    sim = (WEB_VIEWER / "static" / "js" / "sim" / "AgentSimulation.js").read_text(encoding="utf-8")
    assert "setSelectionRingVisible" in sim, "sim cannot toggle the ring"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    assert "elements.ring" in app, "ring toggle not bound"


def test_avatar_code_wired():
    """Avatar codes: copy/import UI, codec module, ?avatar= autoload."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    for elem_id in ("identity-avatar-copy", "identity-avatar-import",
                    "dlg-avatar", "avatar-code-text"):
        assert f'id="{elem_id}"' in html, f"avatar code UI missing: #{elem_id}"
    codec = (WEB_VIEWER / "static" / "js" / "avatarCodec.js").read_text(encoding="utf-8")
    for ref in ("AVATAR_CODE_VERSION", "packAvatarData", "unpackAvatarCode"):
        assert ref in codec, f"codec missing: {ref}"
    app = (WEB_VIEWER / "static" / "app.js").read_text(encoding="utf-8")
    for ref in ("collectAvatarState", "applyAvatarState", "avatarCodec.js"):
        assert ref in app, f"avatar flow missing: {ref}"
    init = app[app.index("async function initialize"):app.index("const ACTOR_TABS")]
    assert "applyUrlAvatar" in init, "?avatar= never applied at boot"
    assert '.get("avatar")' in app, "?avatar= never read"
