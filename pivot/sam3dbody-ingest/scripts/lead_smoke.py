import os, sys, time, json
os.environ.setdefault("TORCH_HOME", r"A:\dev\GNM-V\.cache\torch")
sys.path.insert(0, r"A:\dev\GNM-V\pivot\sam-3d-body")
import numpy as np, cv2, torch
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
ckpt = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\model.ckpt"
mhr = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\assets\mhr_model.pt"
t=time.time(); model,cfg = load_sam_3d_body(ckpt, device="cuda", mhr_path=mhr); load=time.time()-t
est = SAM3DBodyEstimator(model,cfg,human_detector=None,human_segmentor=None,fov_estimator=None)
img = cv2.cvtColor(cv2.imread(r"A:\dev\GNM-V\pivot\teenage-woman-long-wavy-white-gray__01971-2947889296.png"), cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]; print("img", img.shape)
for it in ("body","full"):
    t=time.time(); out = est.process_one_image(img, bboxes=np.array([[0,0,w,h]],dtype=np.float32), inference_type=it); dt=time.time()-t
    o=out[0]; print(it, "sec", round(dt,3), "n", len(out))
    print("  keys/shapes:", {k:(np.asarray(v).shape if not np.isscalar(v) else 'scalar') for k,v in o.items() if v is not None})
    print("  expr max", float(np.abs(np.asarray(o["expr_params"])).max()), "|beta|", round(float(np.linalg.norm(o["shape_params"])),3))
    print("  verts min", np.round(o["pred_vertices"].min(0),3), "max", np.round(o["pred_vertices"].max(0),3), "bbox", np.round(o["bbox"],1))
    if it=="full": print("  lhand", np.round(o.get("lhand_bbox"),1), "rhand", np.round(o.get("rhand_bbox"),1))
torch.cuda.synchronize(); print("gpu peak MB", round(torch.cuda.max_memory_allocated()/2**20,1))
print("load", round(load,2), "cfg IMAGE_SIZE", cfg.MODEL.IMAGE_SIZE)
