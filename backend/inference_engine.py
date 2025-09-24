import io, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Transforms and Encoder (from your script) ===
eval_tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])  # :contentReference[oaicite:3]{index=3}

class Encoder(nn.Module):
    def __init__(self, emb_dim=128, train_backbone=False):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        if not train_backbone:
            for p in backbone.parameters():
                p.requires_grad = False
        self.backbone = backbone
        self.head = nn.Linear(512, emb_dim)
    def forward(self, x): return self.head(self.backbone(x))  # :contentReference[oaicite:4]{index=4}

def _mask_to_bbox(mask):
    ys, xs = mask.nonzero(as_tuple=True)
    return xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()

def _crop_plant_only(orig_img: Image.Image, mask_tensor):
    mh, mw = mask_tensor.shape[0], mask_tensor.shape[1]
    resized = orig_img.resize((mw, mh))
    x1, y1, x2, y2 = _mask_to_bbox(mask_tensor)
    crop = resized.crop((x1, y1, x2, y2))
    m = mask_tensor[y1:y2, x1:x2].cpu().numpy().astype(bool)
    arr = np.array(crop); arr[~m] = 0
    return Image.fromarray(arr), (x1, y1, x2, y2), (mw, mh)  # :contentReference[oaicite:5]{index=5}

class PhaseModel:
    """Holds frozen-prototype MAML checkpoint for a phase (VG/BP/RM)."""
    def __init__(self, ckpt_path, tau_override=None, tau_scale=1.0):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        emb_dim = int(ckpt['emb_dim']); P_obj = ckpt['fixed_proto']
        tau = float(ckpt['fixed_tau'])
        if tau_override is not None: tau = float(tau_override)
        tau *= float(tau_scale)
        self.enc = Encoder(emb_dim=emb_dim).to(device).eval()
        self.enc.load_state_dict(ckpt['state_dict'])
        P = (torch.from_numpy(P_obj) if isinstance(P_obj, np.ndarray)
             else (P_obj.detach() if torch.is_tensor(P_obj) else torch.tensor(P_obj)))
        self.P = F.normalize(P.float().to(device), dim=1)
        self.tau = tau

    def classify_crop(self, crop_img: Image.Image):
        xb = eval_tfm(crop_img).unsqueeze(0).to(device)
        with torch.no_grad():
            z = F.normalize(self.enc(xb), dim=1)
            d2 = (z - self.P).pow(2).sum(1).item()
        is_normal = d2 <= self.tau
        return {"d2": float(d2), "tau": float(self.tau), "label": "Normal" if is_normal else "Abnormal"}

class InferenceEngine:
    """
    Loads YOLO once; supports:
    - Empty-bag detection (always)
    - Phase classifier via MAML for VG/BP/RM
    """
    def __init__(self, yolo_w, phase_models: dict, conf_thresh=0.50, area_thresh=500, box_width=6):
        self.det = YOLO(yolo_w); self.det.model.to(device).eval()
        self.phase_models = phase_models
        self.conf_thresh = float(conf_thresh)
        self.area_thresh = int(area_thresh)
        self.box_width = int(box_width)
        try: self.font = ImageFont.truetype("arial.ttf", 18)
        except: self.font = ImageFont.load_default()

    def _intersects(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a; bx1, by1, bx2, by2 = box_b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    def run(self, image_bytes: bytes, phase: str):
        orig = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        w, h = orig.size; canvas = orig.copy(); draw = ImageDraw.Draw(canvas)

        res = self.det(orig, augment=False, stream=False)[0]
        boxes = res.boxes.xyxy.detach().cpu().numpy() if res.boxes is not None else np.zeros((0,4))
        clses = res.boxes.cls.detach().cpu().numpy() if res.boxes is not None else np.zeros((0,))
        confs = res.boxes.conf.detach().cpu().numpy() if res.boxes is not None else np.zeros((0,))
        masks = res.masks.data.cpu() if getattr(res, 'masks', None) is not None else None

        # Collect detections by class
        plant_idxs = [i for i,(c,cf) in enumerate(zip(clses, confs)) if int(c)==0 and cf>=self.conf_thresh]
        bag_idxs   = [i for i,(c,cf) in enumerate(zip(clses, confs)) if int(c)==1 and cf>=self.conf_thresh]

        dets = []
        # 1) Empty-bag detection (for every phase)
        for bi in bag_idxs:
            bx1,by1,bx2,by2 = boxes[bi].astype(int).tolist()
            # Is there any plant inside/intersecting this bag?
            plant_inside = any(self._intersects((bx1,by1,bx2,by2), boxes[pi].astype(int).tolist())
                               for pi in plant_idxs)
            label = "Abnormal: Empty Bag" if not plant_inside else "Has Plant"
            color = (255,0,0) if not plant_inside else (0,255,0)
            draw.rectangle([bx1,by1,bx2,by2], outline=color, width=self.box_width)
            draw.text((bx1, max(0,by1-20)), label, fill=color, font=self.font)
            dets.append({"type":"bag", "bbox":[bx1,by1,bx2,by2], "label":label,
                         "conf": float(confs[bi])})

        # 2) Plant classification (VG/BP/RM only)
        if phase not in ("EarlySprouting",):
            model = self.phase_models.get(phase)
            for i in plant_idxs:
                x1,y1,x2,y2 = boxes[i].astype(int).tolist()
                if (x2-x1)*(y2-y1) < self.area_thresh: continue
                # prefer mask crop if available
                if masks is not None and masks.shape[0] > i:
                    crop, (mx1,my1,mx2,my2), (mw,mh) = _crop_plant_only(orig, masks[i])
                else:
                    crop = orig.crop((x1,y1,x2,y2))
                out = model.classify_crop(crop)
                is_norm = (out["label"]=="Normal")
                color = (0,0,255) if is_norm else (255,0,0)  # BLUE=Normal, RED=Abnormal (same as your script) :contentReference[oaicite:6]{index=6}
                txt = f'{out["label"]} d2={out["d2"]:.4f} τ={out["tau"]:.4f} conf={float(confs[i]):.2f}'
                draw.rectangle([x1,y1,x2,y2], outline=color, width=self.box_width)
                draw.text((x1, max(0,y1-20)), txt, fill=color, font=self.font)
                dets.append({"type":"plant", "bbox":[x1,y1,x2,y2], "label":out["label"],
                             "conf": float(confs[i]), "d2": out["d2"], "tau": out["tau"]})

        # Verdict rule:
        #  - Any empty bag => Abnormal
        #  - Else if any plant Abnormal => Abnormal
        #  - Else if ≥1 plant Normal OR bags have plant => Normal
        has_empty_bag = any(d["type"]=="bag" and "Empty" in d["label"] for d in dets)
        has_abn_plant = any(d["type"]=="plant" and d["label"]=="Abnormal" for d in dets)
        verdict = "Abnormal" if (has_empty_bag or has_abn_plant) else ("Normal" if dets else "NoDetections")

        # Render annotated
        buf = io.BytesIO(); canvas.save(buf, format="PNG"); annotated_png = buf.getvalue()
        return {"verdict": verdict, "detections": dets, "annotated_png": annotated_png}
