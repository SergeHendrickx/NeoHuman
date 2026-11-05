# NeoHuman — one photo, one parameter vector: a character you can see and hear

## The gap

Single-photo human reconstruction has become impressive and almost useless in the same breath. A model
returns a mesh — a beautiful, frozen shell with no controls. You cannot open its jaw, change its
expression, pose it, or hand it to an animator; you can only look at it. And even when the mesh is
riggable, the character still has no voice. Visual tools build a head, audio tools pick a preset from a
list, and nothing connects them.

NeoHuman takes a different output seriously: not a mesh, but the **parameters that produced it**. From
one photograph we recover a metric, animation-ready character on Meta's **MHR** parametric body/head
model — 45 identity, 130 pose, 28 scale and 72 expression coefficients — and we treat that coefficient
vector as the character's interface. The same 72 expression numbers that open the mesh's jaw are the
articulation input for a speech decoder, so a jaw edit is simultaneously a geometry edit and a sound
edit. The visual pipeline that recovers those numbers is what we ship in this repository; the voice
side consumes the vector in the demo.

## Supervision: Sapiens2

Everything starts with one photo. **Sapiens2** supplies the dense supervision that makes a specific
face possible: 308 keypoints across body, face, ears and hands (we use 272; the iris rows are excluded),
a 29-class segmentation, surface normals, a metric point map and matting. In one local pass we get a
skeleton, a mask, skin normals and depth — the measurements our solver is graded against.

## Identity, pose and expression: SAM 3D Body on MHR

A **SAM 3D Body** forward pass gives a first estimate on the MHR base mesh. The released head, however,
zeroes the 72 expression coefficients and jaw rotation — by design, that head is an identity/pose
solver. We re-enable those 72 numbers and solve identity, pose, scale, expression and camera **jointly**
against the Sapiens2 keypoints, weighting the ears and the dense mouth rows far above the body and
adding a skin-normal consistency term. The gain is measurable against the unmodified fit: lip error
10.8 → 3.3 px, face error 8.18 → 4.26 px, ears 10.03 → 8.75 px. That is the difference between a
generic face and the person in the photograph.

The same solve produces a **canonical A-pose** export, a six-view turnaround sheet, and — by projecting
the photo through the solved camera, z-buffering and baking onto the MHR UV layout — a **textured GLB**
with a single 1024² atlas. One photo in; a riggable, metric, textured character out, in about 20–60
seconds on one RTX 4090.

## Metric geometry: HyDen-MoGeV2

Monocular scale is where attractive reconstructions usually fall over. **HyDen-MoGeV2** (FAIR) adds
metric point maps; its dense depth agrees with our solved camera at **0.990** scale, with a median
dense-depth residual of 14.07 mm (p90 59.17 mm). That is what puts the character in a scene instead of
floating above one.

## Checking the result instead of adjectives

We validate with **VGGT-Ω**. It takes a view sheet, recovers cameras and depth without intrinsics, lifts
the Sapiens2 person masks into a fused point cloud, and reports cross-view consistency. On our
photoreal 2×4 sheets it reports a median of **14.6–15.1 mm** (best pairs 4.1–7.1 mm, worst ~40 mm); a
photoreal 2×2 grid reaches **10.4 mm**, while untextured clay renders sit at **19.7 mm**. Those negative
numbers mattered: they told us that featureless geometry under-matches, and that the turnaround has to
be textured or photoreal before it is worth validating.

For independent per-view fits we do something stronger: fit MHR separately to every panel, align the
fits by head landmarks with Procrustes, and measure the residual. That reaches **3.3 mm** on the scalp
across six views. It also exposes the honest limit — hair has no MHR
geometry, so the scalp and face are the test, not the hair.

## Directed character states: Meta Image

**muse-image-1.0** is used as a director, not a filter. Five calls per full demo run: a photoreal 2×4
six-view turnaround generated from a textured MHR guide plus the source photo; a hat-off edit; neutral;
sad; and a chained sad → bald edit that holds the character while changing state. Each call is one
explicit card. The policy boundary is real
and documented: public figures and identity-transfer phrasing are refused, so the demo uses a private
consented photo and character-creation wording.

## The demo is the system

The node graph is not a video trick — it is how the pipeline is composed. Every model invocation is a
card: bright blue for a Meta cloud API call, blue-edged with yellow interior for a local Meta open
model, violet for local render utilities. A judge can watch Sapiens2 produce keypoints, SAM 3D Body
produce the mesh, texture bake to GLB, Meta Image produce the turnaround, and VGGT or the MHR matcher
report millimetres. Graphs save server-side, artifacts land in one folder per input image, and the 3D
viewers open GLB, OBJ and PLY inline.

## Impact

Character creation stops being a modelling job. An indie studio can take one consented photo and get a
riggable character with 72 face controls; a dubbing team can drive a performance
without a scan; an accessibility tool can give someone an avatar that looks like them and can speak. The
frame is fictional characters, not inference about real people — no medical, diagnostic or demographic
claims.

## Limitations, stated plainly

Single-view identity is ambiguous. Hair, hats and occlusions are the dominant failure modes, and hair is
outside MHR's geometry. Generated turnarounds are not ground truth, and Muse edits can drift in pose or
framing. The expression-to-voice coupling is demonstrated on designed characters, not as an inference
about any individual. We publish these limits because they are what the next iteration is for.

## What is next

A textured multi-view bake driven by the validated turnaround; a richer articulation bridge so every
expression handle has an acoustic consequence;.
The ambition stays the same: a character whose voice is a property of its body.
