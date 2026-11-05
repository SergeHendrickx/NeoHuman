/**
 * Model Catalog & Schema Registry
 * Defines model nodes, input/output contracts, socket types, icons, and hardware residency.
 */

import { svgPlaceholder } from "./util.js";

export const MODEL_CATALOG = {
  "action-meta-sam": {
    id: "action-meta-sam",
    nodeType: "action",
    styleKind: "meta-api",
    family: "meta-ai",
    familyLabel: "Meta AI Foundation · Segment Anything",
    title: "API: Meta SAM 3.1",
    subtitle: "sam-3.1 · Media Segmentation",
    category: "Segmentation Model",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>`,
    acceptsInput: ["image", "crop"],
    param: "people",
    consumes: ["Image Input", "Concept: 'people'"],
    produces: ["Bounding-box list", "Mask list", "Confidence list", "Label list"],
    inputs: [
      { id: "in-image", name: "Image Input", type: "image", color: "#10b981", badge: "RGB" },
      { id: "in-prompt", name: "Concept Prompt", type: "text", color: "#ec4899", badge: "'people'" }
    ],
    outputs: [
      { id: "out-boxes", name: "Bounding Boxes", type: "boxes", badge: "list" },
      { id: "out-masks", name: "Masks", type: "masks", color: "#10b981", badge: "list" },
      { id: "out-confs", name: "Confidences", type: "confidences", color: "#f59e0b", badge: "list" },
      { id: "out-labels", name: "Labels", type: "labels", color: "#ec4899", badge: "list" }
    ],
    description: "Segment Anything Model 3.1 (sam-3.1) segments a noun-phrase concept into typed instance lists: boxes, masks, confidences, labels. Right-click any filled list pin to split it into item pins.",
    latency: "1.85s",
    status: "Ready to Run",
    runtime: "Meta Model API (sam-3.1) + meta-sam parser",
    residency: "Cloud API"
  },
  "action-describe": {
    id: "action-describe",
    nodeType: "action",
    styleKind: "meta-api",
    family: "meta-ai",
    familyLabel: "Meta AI Foundation · Responses API",
    title: "API: Meta Describe",
    subtitle: "describe · Whole-Scene Read",
    category: "Vision Description",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>`,
    acceptsInput: ["image", "crop"],
    consumes: ["Image Input"],
    produces: ["Scene Description Text"],
    inputs: [
      { id: "in-image", name: "Image Input", type: "image", color: "#10b981", badge: "RGB", accepts: ["image", "box", "mask", "crop"] }
    ],
    outputs: [
      { id: "out-text", name: "Description", type: "text", color: "#ec4899", badge: "—" }
    ],
    description: "Describes the whole photo in one shot — scene, subjects, appearance, composition. No boxes, no masks, just the read.",
    latency: "~4s",
    status: "Ready to Run",
    runtime: "Meta Model API (describe) + Responses trace",
    residency: "Cloud API"
  },
  "action-apply-mask": {
    id: "action-apply-mask",
    nodeType: "action",
    styleKind: "render",
    family: "utility",
    familyLabel: "Utility · Local Mask Crop",
    title: "LOCAL: Apply Mask",
    subtitle: "mask × photo · Region Crop",
    category: "Mask Region",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" stroke-dasharray="3 2"/><circle cx="12" cy="12" r="4"/></svg>`,
    acceptsInput: ["mask"],
    consumes: ["Mask Item Pin"],
    produces: ["Masked Region Image"],
    inputs: [
      { id: "in-mask", name: "Mask", type: "mask", color: "#10b981", badge: "Mask", accepts: ["mask", "masks", "list"] }
    ],
    outputs: [
      { id: "out-image", name: "Region Image", type: "image", color: "#10b981", badge: "—" }
    ],
    description: "Crops the photo to one SAM mask: tight crop around the mask itself, masked pixels only (everything outside the mask is black). Wire a Mask pin and get a plain RGB image to feed Describe, Spark or Sapiens.",
    latency: "~1s",
    status: "Ready to Run",
    runtime: "Local OpenCV crop (no model)",
    residency: "Local CPU"
  },
  "action-image-diff": {
    id: "action-image-diff",
    nodeType: "action",
    styleKind: "render",
    family: "utility",
    familyLabel: "Utility · Local Pixel Diff",
    title: "LOCAL: Image Difference",
    subtitle: "photo_a − photo_b · Δ map",
    category: "Image Compare",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="8" height="18" rx="1"/><rect x="13" y="3" width="8" height="18" rx="1"/></svg>`,
    acceptsInput: ["image"],
    consumes: ["Two Images"],
    produces: ["Amplified Δ Map"],
    inputs: [
      { id: "in-a", name: "A", type: "image", color: "#10b981", badge: "A", accepts: ["image"] },
      { id: "in-b", name: "B", type: "image", color: "#10b981", badge: "B", accepts: ["image"] }
    ],
    outputs: [
      { id: "out-image", name: "Difference", type: "image", color: "#10b981", badge: "—" }
    ],
    description: "Amplified absolute pixel difference of two photos (B resampled to A when sizes differ): mean and peak Δ plus a Δ map. Wire an overlay against its photo to quantify a fit.",
    latency: "local",
    status: "Ready to Run",
    runtime: "Local canvas 2D (no model)",
    residency: "Local CPU"
  },
  "action-meta-muse": {
    id: "action-meta-muse",
    nodeType: "action",
    styleKind: "meta-api",
    family: "meta-ai",
    familyLabel: "Meta AI Foundation · Responses API",
    title: "API: Meta Muse Image",
    subtitle: "muse-image-1.0 · Generation & Editing",
    category: "Image Generation",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M20.4 14.5L16 10 4 20"/></svg>`,
    acceptsInput: ["text", "image", "box", "mask", "crop"],
    param: "",
    consumes: ["Prompt Text", "Images (optional)"],
    produces: ["Generated Image"],
    inputs: [
      { id: "in-prompt", name: "Prompt", type: "text", color: "#ec4899", badge: "Text" },
      { id: "in-images", name: "Images", type: "image", color: "#10b981", badge: "Optional", accepts: ["image", "box", "mask", "crop"] }
    ],
    outputs: [
      { id: "out-image", name: "Image", type: "image", color: "#10b981", badge: "—" }
    ],
    description: "Generic Muse Image call (muse-image-1.0): prompt alone generates (POST /v1/images/generations), prompt plus images edits (POST /v1/images/edits, 1–10 refs).",
    latency: "—",
    status: "Ready to Run",
    runtime: "Meta Images API (muse-image-1.0)",
    residency: "Cloud API"
  },
  "action-spark": {
    id: "action-spark",
    nodeType: "action",
    styleKind: "meta-api",
    family: "meta-ai",
    familyLabel: "Meta AI Foundation · Muse Spark",
    title: "API: Meta Spark",
    subtitle: "muse-spark-1.3-contributor · Multimodal Chat",
    category: "Multimodal Chat",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`,
    acceptsInput: ["text", "image", "box", "mask", "crop"],
    param: "",
    consumes: ["Prompt Text", "Images (optional)"],
    produces: ["Reply Text"],
    inputs: [
      { id: "in-prompt", name: "Prompt", type: "text", color: "#ec4899", badge: "Text" },
      { id: "in-images", name: "Images", type: "image", color: "#10b981", badge: "Optional", accepts: ["image", "box", "mask", "crop"] }
    ],
    outputs: [
      { id: "out-text", name: "Reply", type: "text", color: "#ec4899", badge: "—" }
    ],
    description: "Generic Muse Spark call (muse-spark-1.3-contributor): prompt text plus optional images in, reply text out. Mirrors POST /v1/responses — one input_text block, one input_image block per wired image.",
    latency: "—",
    status: "Ready to Run",
    runtime: "Meta Responses API (muse-spark-1.3-contributor)",
    residency: "Cloud API"
  },
  "action-sapiens-pose": {
    id: "action-sapiens-pose",
    nodeType: "action",
    styleKind: "meta-oss",
    family: "sapiens",
    familyLabel: "Sapiens2 Family · 0.4B Foundation",
    title: "LOCAL: SapiensPose",
    subtitle: "sapiens_0.4b_pose",
    category: "Model Action",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="5" r="2"/><path d="m9 20 3-6 3 6"/><path d="m6 8 6 2 6-2"/><path d="M12 10v4"/></svg>`,
    acceptsInput: ["image", "crop"],
    consumes: ["Image"],
    produces: ["308 keypoints (x, y, score)"],
    inputs: [
      { id: "in-image", name: "Image", type: "image", color: "#10b981", badge: "RGB", accepts: ["image", "box", "mask", "crop"] }
    ],
    outputs: [
      { id: "out-kpts", name: "308 Full Keypoints", type: "pointcloud", color: "#06b6d4", badge: "308 pts" },
      { id: "out-face", name: "Face Keypoints", type: "pointcloud", badge: "238 pts" },
      { id: "out-body", name: "Torso & Limbs", type: "pointcloud", color: "#10b981", badge: "70 pts" }
    ],
    description: "Runs Sapiens 0.4B Pose vision transformer to detect dense 308 anatomical keypoints across face, hands, feet, and torso.",
    latency: "8.87s",
    status: "Ready to Run",
    runtime: "Local CUDA 12.8 (RTX 4090)",
    residency: "Resident ModelSlot (~2.4GB VRAM)"
  },
  "action-sapiens-normal": {
    id: "action-sapiens-normal",
    nodeType: "action",
    styleKind: "meta-oss",
    family: "sapiens",
    familyLabel: "Sapiens2 Family · 0.4B Foundation",
    title: "LOCAL: SapiensNormal",
    subtitle: "sapiens_0.4b_normal",
    category: "Model Action",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 14.14 14.14"/></svg>`,
    acceptsInput: ["image", "crop"],
    consumes: ["Image", "Seg labels"],
    produces: ["Surface normals (.npy)"],
    inputs: [
      { id: "in-image", name: "Image", type: "image", color: "#10b981", badge: "RGB", accepts: ["image", "box", "mask", "crop"] },
      { id: "in-kpts", name: "308 Keypoints", type: "pointcloud", color: "#06b6d4", badge: "Pose" }
    ],
    outputs: [
      { id: "out-normals", name: "Surface Normal Map", type: "normalmap", badge: "Normals" }
    ],
    description: "Dense surface orientation estimator predicting unit surface normal vectors [Nx, Ny, Nz] for skin and clothing.",
    latency: "6.52s",
    status: "Ready to Run",
    runtime: "Local CUDA 12.8 (RTX 4090)",
    residency: "Sequential ModelSlot eviction"
  },
  "action-sapiens-depth": {
    id: "action-sapiens-depth",
    nodeType: "action",
    styleKind: "meta-oss",
    family: "sapiens",
    familyLabel: "Sapiens2 Family · 0.4B Foundation",
    title: "LOCAL: SapiensDepth",
    subtitle: "sapiens_0.4b_depth",
    category: "Model Action",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12h20"/><path d="M20 12v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-6"/></svg>`,
    acceptsInput: ["image", "crop"],
    consumes: ["Image", "Seg labels"],
    produces: ["Metric depth in metres (.npy)"],
    inputs: [
      { id: "in-image", name: "Image", type: "image", color: "#10b981", badge: "RGB", accepts: ["image", "box", "mask", "crop"] },
      { id: "in-kpts", name: "308 Keypoints", type: "pointcloud", color: "#06b6d4", badge: "Pose" }
    ],
    outputs: [
      { id: "out-depth", name: "Metric Depth Map", type: "depthmap", color: "#06b6d4", badge: "Metres" }
    ],
    description: "Monocular metric depth estimation model outputting Euclidean pointmap distance for body surfaces in meters.",
    latency: "6.84s",
    status: "Ready to Run",
    runtime: "Local CUDA 12.8 (RTX 4090)",
    residency: "Sequential ModelSlot eviction"
  },
  "action-sapiens-seg": {
    id: "action-sapiens-seg",
    nodeType: "action",
    styleKind: "meta-oss",
    family: "sapiens",
    familyLabel: "Sapiens2 Family · 0.4B Foundation",
    title: "LOCAL: SapiensSeg",
    subtitle: "sapiens_0.4b_seg",
    category: "Model Action",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 14.93V17a1 1 0 0 1-2 0v-.07A8 8 0 0 1 4.07 11H5a1 1 0 0 1 0 2H4.07A8 8 0 0 1 11 4.07V5a1 1 0 0 1 2 0v-.93A8 8 0 0 1 19.93 11H19a1 1 0 0 1 0-2h.93A8 8 0 0 1 13 16.93z"/></svg>`,
    acceptsInput: ["image", "crop"],
    consumes: ["Image"],
    produces: ["28-class labels", "Skin mask", "Clothing mask"],
    inputs: [
      { id: "in-image", name: "Image", type: "image", color: "#10b981", badge: "RGB", accepts: ["image", "box", "mask", "crop"] }
    ],
    outputs: [
      { id: "out-parts", name: "28-Class Segmentation", type: "image", color: "#10b981", badge: "28 Classes" },
      { id: "out-skin", name: "Skin Mask", type: "mask", color: "#ec4899", badge: "Face & Body" },
      { id: "out-clothing", name: "Clothing Mask", type: "mask", color: "#3b82f6", badge: "Garments" }
    ],
    description: "28-class body part parser segmenting face, hair, arms, torso, legs, and apparel.",
    latency: "6.25s",
    status: "Ready to Run",
    runtime: "Local CUDA 12.8 (RTX 4090)",
    residency: "Sequential ModelSlot eviction"
  },
  "action-body-fit": {
    id: "action-body-fit",
    nodeType: "action",
    styleKind: "meta-oss",
    family: "avatar-solver",
    familyLabel: "Parametric Avatar · SAM 3D Suite",
    title: "LOCAL: SAM 3D Body & Joint Solver",
    subtitle: "smpl_mhr_fitter",
    category: "Solver Action",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
    acceptsInput: ["image", "crop"],
    consumes: ["Image", "Sapiens2 308 keypoints", "Sapiens2 seg mask", "Sapiens2 normals", "Sapiens2 depth"],
    produces: ["MHR avatar mesh (18,439 verts)", "Identity beta (45)", "Expression (72)", "Camera"],
    inputs: [
      { id: "in-image", name: "Image", type: "image", color: "#10b981", badge: "RGB", accepts: ["image", "box", "mask", "crop"] },
      { id: "in-pose", name: "308 Keypoints", type: "pointcloud", color: "#06b6d4", badge: "Sapiens2 pose", accepts: ["pointcloud"] },
      { id: "in-seg", name: "Seg Mask", type: "mask", color: "#ec4899", badge: "Sapiens2 seg", accepts: ["mask", "image"] },
      { id: "in-normal", name: "Normals", type: "normalmap", color: "#0866FF", badge: "Sapiens2 normal", accepts: ["normalmap"] },
      { id: "in-depth", name: "Depth", type: "depthmap", color: "#06b6d4", badge: "Sapiens2 depth", accepts: ["depthmap"] },
      { id: "in-init", name: "Warm Start", type: "mesh", color: "#f59e0b", badge: "previous stage", accepts: ["mesh"] }
    ],
    outputs: [
      { id: "out-mesh", name: "3D Parametric Mesh", type: "mesh", color: "#3b82f6", badge: "18,439 verts" },
      { id: "out-beta", name: "Shape Beta Vector (β)", type: "vector", color: "#f59e0b", badge: "45 coeffs" },
      { id: "out-expr", name: "Expression Vector (72)", type: "vector", color: "#ec4899", badge: "72 · jaw/lips" }
    ],
    description: "Fits the MHR base mesh to one image. Wire the Sapiens2 pose/seg/normal/depth cards into it to keep every Meta model invocation separate (unwired inputs fall back to running Sapiens2 internally). 45 identity coefficients, global/body pose, scale state, camera AND 72 expression coefficients (jaw/lips) solved jointly against Sapiens2 308 keypoints + normals + depth re-derived internally. Outputs the mesh plus identity/beta and expression vectors for downstream voice coupling. Toggle the expression solve (ON: jaw/lips fitted from 0, OFF: neutral) on the card.",
    latency: "62.27s",
    status: "Ready to Run",
    runtime: "CUDA 12.8 & SciPy Joint Solver",
    residency: "Resident ModelSlot (~4.8GB VRAM)",
    param: "on"
  },
  "action-body-fit-raw": {
    id: "action-body-fit-raw",
    nodeType: "action",
    styleKind: "meta-oss",
    family: "avatar-solver",
    familyLabel: "Parametric Avatar · SAM 3D Body",
    title: "LOCAL: SAM 3D Body · HMR Baseline",
    subtitle: "hmr_map_only · image -> mesh, nothing else",
    category: "Baseline Solver",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 2v20M2 12h20"/><circle cx="12" cy="12" r="7"/></svg>`,
    acceptsInput: ["image"],
    consumes: ["Image only"],
    produces: ["Parametric MHR mesh (raw HMR)"],
    inputs: [
      { id: "in-image", name: "Image", type: "image", color: "#10b981", badge: "RGB", accepts: ["image", "crop", "box", "mask"] }
    ],
    outputs: [
      { id: "out-mesh", name: "Raw HMR Mesh", type: "mesh", color: "#3b82f6", badge: "18,439 verts" }
    ],
    description: "Stage 1 baseline: one photo in, SAM 3D Body single forward pass (HMR mapping) out. No Sapiens2 cards, no identity refine, no joint solve, no expression. This is the honest floor the later stages improve on.",
    latency: "~15s",
    status: "Ready to Run",
    runtime: "Local CUDA 12.8 (SAM 3D Body only)",
    residency: "Resident ModelSlot (~3.6GB VRAM)"
  },
  "action-mhr-match": {
    id: "action-mhr-match",
    nodeType: "action",
    styleKind: "meta-oss",
    family: "avatar-solver",
    familyLabel: "Parametric Avatar · SAM 3D Suite",
    title: "LOCAL: MHR Multi-View Match",
    subtitle: "per-panel MHR fits",
    category: "Validator Action",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="9" cy="9" r="3.2"/><circle cx="15" cy="9" r="3.2"/><circle cx="9" cy="15" r="3.2"/><circle cx="15" cy="15" r="3.2"/><path d="M11.4 8.4 12.6 8.4M11.4 15.6 12.6 15.6M8.4 11.4 8.4 12.6M15.6 11.4 15.6 12.6"/></svg>`,
    acceptsInput: ["image", "crop"],
    consumes: ["View sheet (2x2 / 2x3 / 2x4)"],
    produces: ["Aligned MHR head cloud (PLY)", "Cross-view residual (mm)"],
    inputs: [
      { id: "in-image", name: "View Sheet", type: "image", color: "#10b981", badge: "Grid", accepts: ["image", "crop", "box", "mask"] }
    ],
    outputs: [
      { id: "out-cloud", name: "Aligned MHR Cloud", type: "image", color: "#06b6d4", badge: "PLY + RGB" }
    ],
    description: "Fits the MHR parametric body independently to every panel, then aligns the fits by head landmarks (Procrustes) and reports the residual in mm. This is the strong consistency authority: independent MHR fits align at a few millimetres on the scalp, roughly 5x better than VGGT's monocular estimates. Heavier than the VGGT card (~90-150s for 8 panels) because each panel is a full identity + pose + expression solve. Hair is not MHR geometry, so judge the scalp and face, not the hair.",
    latency: "~2 min",
    status: "Ready to Run",
    runtime: "Local CUDA 12.8 (RTX 4090)",
    residency: "Sequential ModelSlot eviction"
  },
  "action-vggt-check": {
    id: "action-vggt-check",
    nodeType: "action",
    styleKind: "meta-oss",
    family: "avatar-solver",
    familyLabel: "Parametric Avatar · SAM 3D Suite",
    title: "LOCAL: VGGT Multi-View Check",
    subtitle: "vggt_omega_1b",
    category: "Validator Action",
    badgeVariant: "meta-light",
    iconVariant: "meta",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="2"/><circle cx="5" cy="6" r="1.6"/><circle cx="19" cy="6" r="1.6"/><circle cx="5" cy="18" r="1.6"/><circle cx="19" cy="18" r="1.6"/><path d="M6.4 7.2 10.6 10.9M17.6 7.2 13.4 10.9M6.4 16.8 10.6 13.1M17.6 16.8 13.4 13.1"/></svg>`,
    acceptsInput: ["image", "crop"],
    consumes: ["View sheet (2x2 / 2x3 / 1x6)"],
    produces: ["Cameras + fused point cloud", "Cross-view consistency (mm)"],
    inputs: [
      { id: "in-image", name: "View Sheet", type: "image", color: "#10b981", badge: "Grid", accepts: ["image", "crop", "box", "mask"] }
    ],
    outputs: [
      { id: "out-cloud", name: "Fused Point Cloud", type: "image", color: "#06b6d4", badge: "PLY + RGB" }
    ],
    description: "Runs VGGT-Omega on a view sheet: recovers cameras and depth, lifts Sapiens2 person masks into one fused point cloud and reports cross-view consistency in mm. Feed it the textured/photoreal turnaround, not a clay render - featureless grey geometry under-matches (measured: clay 19.7 mm vs textured 17.4 mm vs photoreal Muse grid 10.4 mm). Use it as the QA gate before texturing.",
    latency: "~20s",
    status: "Ready to Run",
    runtime: "Local CUDA 12.8 (RTX 4090)",
    residency: "Sequential ModelSlot eviction"
  },
  "action-texture": {
    id: "action-texture",
    nodeType: "action",
    styleKind: "render",
    family: "avatar-solver",
    familyLabel: "Parametric Avatar · SAM 3D Suite",
    title: "LOCAL: Texture Bake",
    subtitle: "photo_projection_bake",
    category: "Texture Action",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 4h16v16H4z"/><path d="M4 9h16M9 4v16"/><circle cx="15" cy="15" r="2.4"/></svg>`,
    acceptsInput: ["mesh"],
    consumes: ["Fitted MHR mesh", "Source photo"],
    produces: ["Textured GLB", "UV texture (1024)", "Textured preview"],
    inputs: [
      { id: "in-mesh", name: "Fitted Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh", accepts: ["mesh"] },
      { id: "in-photo", name: "Source Photo", type: "image", color: "#10b981", badge: "RGB", accepts: ["image", "crop", "mask", "box"] }
    ],
    outputs: [
      { id: "out-glb", name: "Textured GLB", type: "mesh", color: "#10b981", badge: "GLB" },
      { id: "out-sheet", name: "Textured Turnaround", type: "image", color: "#10b981", badge: "6 views" }
    ],
    description: "Projects the source photo through the fitted camera onto the MHR mesh (z-buffered, front-facing only), fills unseen verts from their nearest visible neighbours, bakes a 1024 MHR-UV texture and exports a textured GLB with vertex colours. Closes the graph with a textured asset, not just geometry.",
    latency: "~25s",
    status: "Ready to Run",
    runtime: "Local CUDA 12.8 + numpy rasteriser",
    residency: "Resident ModelSlot (~4.8GB VRAM)"
  },
  "action-apose": {
    id: "action-apose",
    nodeType: "action",
    styleKind: "render",
    family: "avatar-solver",
    familyLabel: "Parametric Avatar · SAM 3D Suite",
    title: "LOCAL: Canonical A-Pose",
    subtitle: "canonical_apose_export",
    category: "Export Action",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="5" r="2"/><path d="M12 7v6"/><path d="M5 9h14"/><path d="M12 13l-3 7M12 13l3 7"/></svg>`,
    acceptsInput: ["mesh"],
    consumes: ["Fitted MHR mesh + identity JSON"],
    produces: ["A-pose mesh (PLY/OBJ)", "6-view turnaround sheet"],
    inputs: [
      { id: "in-mesh", name: "Fitted Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh", accepts: ["mesh"] }
    ],
    outputs: [
      { id: "out-mesh", name: "A-Pose Mesh", type: "mesh", color: "#3b82f6", badge: "PLY/OBJ" },
      { id: "out-sheet", name: "Turnaround Sheet", type: "image", color: "#10b981", badge: "6 views" }
    ],
    description: "Rebuilds the fitted MHR avatar in a canonical A-pose (identity + scale kept, body euler zeroed, neutral expression) and renders a 6-view turnaround sheet. This is the comparable, reusable asset for texturing and multi-view validation.",
    latency: "~8s",
    status: "Ready to Run",
    runtime: "CUDA 12.8 + Open3D offscreen",
    residency: "Resident ModelSlot (~4.8GB VRAM)"
  },
  "action-export": {
    id: "action-export",
    nodeType: "action",
    styleKind: "render",
    family: "utility",
    familyLabel: "Character Pipeline Exporter",
    title: "LOCAL: Character Exporter",
    subtitle: "glb_pack_exporter",
    category: "Export Action",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>`,
    acceptsInput: ["mesh", "geometry"],
    consumes: ["Fitted mesh (+ fit stats)"],
    produces: ["avatar_character.glb"],
    inputs: [
      { id: "in-mesh", name: "3D Parametric Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh" }
    ],
    outputs: [
      { id: "out-glb", name: "avatar_character.glb", type: "mesh", color: "#10b981", badge: "Binary 3D" }
    ],
    description: "Bakes materials, verifies non-biometric governance compliance, and packages character assets into portable GLB.",
    latency: "0.12s",
    status: "Ready to Run",
    runtime: "Local CPU export",
    residency: "Disk storage"
  },
  "action-mesh-render": {
    id: "action-mesh-render",
    nodeType: "action",
    styleKind: "render",
    family: "mesh",
    familyLabel: "Mesh Render · open3d (server)",
    title: "LOCAL: Mesh Renderer",
    subtitle: "mesh_still_render",
    category: "Render Action",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><circle cx="12" cy="12" r="3"/></svg>`,
    acceptsInput: ["mesh"],
    consumes: ["Mesh"],
    produces: ["Shaded Still Image"],
    inputs: [
      { id: "in-mesh", name: "Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh" }
    ],
    outputs: [
      { id: "out-image", name: "Shaded Still", type: "image", color: "#10b981", badge: "Still" }
    ],
    description: "Renders the wired mesh to a shaded 512px still on the server (open3d offscreen, same 3/4 view as the 3D stage). CPU call, no GPU slot.",
    latency: "~5s",
    status: "Ready to Run",
    runtime: "Server CPU (open3d offscreen)",
    residency: "None — no GPU slot"
  },
  "action-mesh-depth": {
    id: "action-mesh-depth",
    nodeType: "action",
    styleKind: "render",
    family: "mesh",
    familyLabel: "Mesh Render · open3d (server)",
    title: "LOCAL: Mesh Depth",
    subtitle: "mesh_depth_map",
    category: "Render Action",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 15l6-6 4 4 3-3 5 5"/></svg>`,
    acceptsInput: ["mesh"],
    consumes: ["Mesh"],
    produces: ["White-Near Depth Map"],
    inputs: [
      { id: "in-mesh", name: "Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh" }
    ],
    outputs: [
      { id: "out-depth", name: "Depth Map", type: "depthmap", color: "#06b6d4", badge: "Depth" }
    ],
    description: "Renders the wired mesh to a 512px near-white depth map on the server (open3d depth buffer, contrast-stretched over the body). CPU call, no GPU slot.",
    latency: "~5s",
    status: "Ready to Run",
    runtime: "Server CPU (open3d offscreen)",
    residency: "None — no GPU slot"
  },
  "action-mesh-normal": {
    id: "action-mesh-normal",
    nodeType: "action",
    styleKind: "render",
    family: "mesh",
    familyLabel: "Mesh Render · open3d (server)",
    title: "LOCAL: Mesh Normals",
    subtitle: "mesh_normal_map",
    category: "Render Action",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>`,
    acceptsInput: ["mesh"],
    consumes: ["Mesh"],
    produces: ["View-Space Normal Map"],
    inputs: [
      { id: "in-mesh", name: "Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh" }
    ],
    outputs: [
      { id: "out-normal", name: "Normal Map", type: "normalmap", color: "#0866FF", badge: "Normals" }
    ],
    description: "Renders the wired mesh to a 512px view-space normal map on the server (open3d, RGB = XYZ). CPU call, no GPU slot.",
    latency: "~5s",
    status: "Ready to Run",
    runtime: "Server CPU (open3d offscreen)",
    residency: "None — no GPU slot"
  },
  "action-mesh-multiview": {
    id: "action-mesh-multiview",
    nodeType: "action",
    styleKind: "render",
    family: "mesh",
    familyLabel: "Mesh Render · open3d (server)",
    title: "LOCAL: Multi-View Renderer",
    subtitle: "mesh_multiview_sheet",
    category: "Render Action",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/></svg>`,
    acceptsInput: ["mesh"],
    consumes: ["Mesh"],
    produces: ["Labelled multi-view sheet"],
    param: "4",
    framing: "body",
    inputs: [
      { id: "in-mesh", name: "Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh" }
    ],
    outputs: [
      { id: "out-image", name: "View Sheet", type: "image", color: "#10b981", badge: "Sheet" }
    ],
    description: "Photographs the wired mesh from 2, 4, 6 or 8 even azimuths into one labelled contact sheet on the server (open3d offscreen), framed on the full body, waist-up or face. CPU call, no GPU slot.",
    latency: "~25s",
    status: "Ready to Run",
    runtime: "Server CPU (open3d offscreen)",
    residency: "None — no GPU slot"
  },
  "action-fit-overlay": {
    id: "action-fit-overlay",
    nodeType: "action",
    styleKind: "render",
    family: "avatar-solver",
    familyLabel: "Parametric Avatar · SAM 3D Suite",
    title: "LOCAL: Fit Overlay",
    subtitle: "fit_matched_overlay",
    category: "Fit Check Action",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>`,
    acceptsInput: ["image", "mesh"],
    consumes: ["Photo", "Fitted Mesh"],
    produces: ["Overlay Blend", "Fit-Frame Photo", "Matched Render"],
    inputs: [
      { id: "in-photo", name: "Photo", type: "image", color: "#10b981", badge: "RGB", accepts: ["image"] },
      { id: "in-mesh", name: "Fit", type: "mesh", color: "#3b82f6", badge: "Fit" }
    ],
    outputs: [
      { id: "out-overlay", name: "Overlay 50/50", type: "image", color: "#10b981", badge: "Blend" },
      { id: "out-photo", name: "Fit-Frame Photo", type: "image", color: "#10b981", badge: "Photo" },
      { id: "out-render", name: "Matched Render", type: "image", color: "#10b981", badge: "Render" }
    ],
    description: "Re-renders the fitted mesh in the photo frame (same matched-frame painter body-fit uses) and blends it 50/50 over the photo. Three separate frames out: fit-frame photo, matched render, blend. Wire the photo and the body-fit (or avatar) card.",
    latency: "local CPU",
    status: "Ready to Run",
    runtime: "Local CPU (Open3D offscreen)",
    residency: "None — no GPU slot"
  },
  "asset-photo-viewer": {
    id: "asset-photo-viewer",
    nodeType: "asset",
    assetKind: "image",
    family: "utility",
    familyLabel: "Utility · Viewer",
    title: "Asset: Photo Viewer",
    subtitle: "viewer",
    category: "Viewer",
    badgeVariant: "success-light",
    iconVariant: "outline",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>`,
    acceptsInput: ["image", "crop", "box", "mask"],
    imgUrl: svgPlaceholder("Wire an image to view"),
    imgFallback: "",
    imgBadge: "unwired",
    description: "Viewer card: shows whatever image, crop, box or mask is wired into its input, and passes the pixels through its output. Drop it anywhere mid-flow to inspect passing images.",
    latency: "—",
    status: "Waiting for Input",
    tensorShape: "RGB image (? × ?)",
    format: "RGB Bitmap",
    inputs: [
      { id: "in-view", name: "View", type: "image", color: "#10b981", badge: "View", accepts: ["image", "box", "mask", "crop"] }
    ],
    outputs: [
      { id: "out-view", name: "Passthrough", type: "image", color: "#10b981", badge: "RGB" }
    ]
  },
  "asset-note": {
    id: "asset-note",
    nodeType: "note",
    family: "utility",
    familyLabel: "Notes & Annotations",
    title: "Note",
    subtitle: "post-it",
    category: "Annotation",
    badgeVariant: "warning-light",
    iconVariant: "outline",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 4h16v12l-4 4H4z"/><path d="M16 20v-4h4"/></svg>`,
    description: "Free-form note card for explaining the graph on the board - drop it anywhere and type. Saves with the flow.",
    noteText: "Double-click to edit. Use notes to explain the pipeline, assumptions and numbers.",
    noteWidth: 320,
    latency: "-",
    status: "Note",
    tensorShape: "-",
    format: "Text",
    inputs: [],
    outputs: []
  },
  "asset-photo": {
    id: "asset-photo",
    nodeType: "asset",
    assetKind: "image",
    family: "utility",
    familyLabel: "Utility · Photo",
    title: "Asset: Source Photo",
    subtitle: "photo",
    category: "Image Asset",
    badgeVariant: "success-light",
    iconVariant: "outline",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
    acceptsInput: ["image", "crop", "box", "mask"],
    imgUrl: svgPlaceholder("Drop an image"),
    imgFallback: "",
    imgBadge: "unwired",
    description: "Source photo card: drop an image onto it (or wire a viewer pin in) and feed its Photo Output into any model. The preset flow starts with one; add more from the Nodes gallery.",
    latency: "—",
    status: "Waiting for Input",
    tensorShape: "RGB image (? × ?)",
    format: "RGB Bitmap / JPEG",
    inputs: [
      { id: "in-view", name: "View", type: "image", color: "#10b981", badge: "View", accepts: ["image", "box", "mask", "crop"] }
    ],
    outputs: [
      { id: "out-photo", name: "Photo Output", type: "image", color: "#10b981", badge: "RGB" }
    ]
  }
};
