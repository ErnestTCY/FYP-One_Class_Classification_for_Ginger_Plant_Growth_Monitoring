#!/usr/bin/env python3
import os, random, argparse, csv, time
from glob import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', default='dataset/train/Normal')
    ap.add_argument('--val_normal', default='dataset/val/Normal')
    ap.add_argument('--val_anom',   default='dataset/val/Abnormal')

    ap.add_argument('--emb_dim', type=int, default=128)

    # TRAINING EPISODES (no K-shot concept here)
    ap.add_argument('--episodes', type=int, default=800)
    ap.add_argument('--batch_images', type=int, default=32, help='distinct images sampled per episode')
    ap.add_argument('--pos_per_image', type=int, default=1, help='positive pairs per image')
    ap.add_argument('--neg_per_image', type=int, default=4, help='negative pairs per image')

    # OPTIM
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--train_backbone', action='store_true', help='fine-tune ResNet if set')

    # EVAL (K-shot ONLY used here)
    ap.add_argument('--k_shot_eval', type=int, default=5)
    ap.add_argument('--eval_runs', type=int, default=20)

    ap.add_argument('--checkpoint', default='checkpoints/RM_relational.pth')
    ap.add_argument('--log_csv',    default='logs/RM_train_log.csv')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--light_aug', action='store_true')
    return ap.parse_args()

# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────
def make_train_tfm(light=False):
    if light:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.4,0.4,0.4,0.2),
        transforms.GaussianBlur(3, sigma=(0.1,2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.2)),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

eval_tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────
def list_imgs(root):
    if not os.path.isdir(root): return []
    return sorted([p for p in glob(os.path.join(root, '*')) if os.path.isfile(p)])

def load_img(p, tfm):
    return tfm(Image.open(p).convert('RGB'))

def sample_paths(paths, k):
    if len(paths) < k: return random.choices(paths, k=k)
    return random.sample(paths, k)

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """ResNet18 trunk + projection head to emb_dim; L2-normalized output."""
    def __init__(self, emb_dim=128, train_backbone=False):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        if not train_backbone:
            for p in backbone.parameters(): p.requires_grad = False
        self.backbone = backbone
        self.head = nn.Linear(512, emb_dim)

    def forward(self, x):
        z = self.backbone(x)
        z = self.head(z)
        return F.normalize(z, dim=1)

class RelationHead(nn.Module):
    """MLP over pair-encoding: [|z1−z2|, z1⊙z2] → sim∈[0,1]."""
    def __init__(self, emb_dim=128, hidden=256):
        super().__init__()
        in_dim = 2*emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden//2), nn.ReLU(inplace=True),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, z1, z2):
        diff = torch.abs(z1 - z2)
        had  = z1 * z2
        x = torch.cat([diff, had], dim=1)
        return torch.sigmoid(self.mlp(x).squeeze(1))

# ──────────────────────────────────────────────────────────────────────────────
# Pair builder (1-class)
# ──────────────────────────────────────────────────────────────────────────────
def build_pairs_oneclass(paths, tfm, batch_images=32, pos_per_image=1, neg_per_image=4):
    sel = sample_paths(paths, batch_images)
    # Preload one view for negatives
    views = [load_img(p, tfm) for p in sel]

    x1, x2, y = [], [], []

    # Positives: two views of the SAME image
    for p in sel:
        for _ in range(pos_per_image):
            v1 = load_img(p, tfm); v2 = load_img(p, tfm)
            x1.append(v1); x2.append(v2); y.append(1)

    # Negatives: view of image i vs view of image j≠i
    n = len(sel)
    for i in range(n):
        a = views[i]
        for _ in range(neg_per_image):
            j = random.randrange(n)
            while j == i:
                j = random.randrange(n)
            b = views[j]
            x1.append(a); x2.append(b); y.append(0)

    x1 = torch.stack(x1, 0)
    x2 = torch.stack(x2, 0)
    y  = torch.tensor(y, dtype=torch.float32)
    return x1, x2, y

# ──────────────────────────────────────────────────────────────────────────────
# Logging helper
# ──────────────────────────────────────────────────────────────────────────────
def count_params(module, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())

class CSVLogger:
    def __init__(self, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(header); self.f.flush()
    def log(self, row):
        self.w.writerow(row); self.f.flush()
    def close(self):
        try: self.f.close()
        except: pass

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────
def train_rn(args):
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    train_paths = list_imgs(args.train_dir)
    assert len(train_paths) >= max(8, args.batch_images//2), "Need more normal images in train_dir."

    train_tfm = make_train_tfm(args.light_aug)
    enc = Encoder(emb_dim=args.emb_dim, train_backbone=args.train_backbone).to(device)
    rel = RelationHead(emb_dim=args.emb_dim).to(device)

    # Parameters & optimizer
    if args.train_backbone:
        params = list(enc.parameters()) + list(rel.parameters())
    else:
        params = list(enc.head.parameters()) + list(rel.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # Print a one-shot config banner
    total_params = count_params(enc, False) + count_params(rel, False)
    trainable_params = count_params(enc, True) + count_params(rel, True)
    print("──────────────────────────────── CONFIG ────────────────────────────────")
    print(f"Device={device}  emb_dim={args.emb_dim}  train_backbone={args.train_backbone}")
    print(f"episodes={args.episodes}  batch_images={args.batch_images}  pos/img={args.pos_per_image}  neg/img={args.neg_per_image}")
    print(f"lr={args.lr}  weight_decay={args.weight_decay}")
    print(f"Params: total={total_params:,}  trainable={trainable_params:,}")
    print(f"Checkpoint: {args.checkpoint}")
    print("────────────────────────────────────────────────────────────────────────")

    logger = CSVLogger(args.log_csv, header=[
        "episode","loss","ema_loss","pos_acc","neg_acc","pairs",
        "lr","batch_images","pos_per_image","neg_per_image","emb_dim",
        "train_backbone","trainable_params","device","time_sec"
    ])

    ema = None
    best_loss = float('inf')
    t0 = time.time()

    for ep in range(1, args.episodes+1):
        opt.zero_grad()
        x1, x2, y = build_pairs_oneclass(
            train_paths, train_tfm,
            batch_images=args.batch_images,
            pos_per_image=args.pos_per_image,
            neg_per_image=args.neg_per_image
        )
        x1 = x1.to(device); x2 = x2.to(device); y = y.to(device)

        z1 = enc(x1); z2 = enc(x2)
        s  = rel(z1, z2)  # similarity ∈ [0,1]
        loss = F.binary_cross_entropy(s, y)

        # Simple accuracy split (threshold 0.5)
        with torch.no_grad():
            preds = (s >= 0.5).float()
            pos_mask = (y == 1); neg_mask = (y == 0)
            pos_acc = (preds[pos_mask] == 1).float().mean().item() if pos_mask.any() else float('nan')
            neg_acc = (preds[neg_mask] == 0).float().mean().item() if neg_mask.any() else float('nan')

        loss.backward()
        opt.step()

        ema = loss.item() if ema is None else 0.9*ema + 0.1*loss.item()

        # Current LR (no scheduler -> single group)
        cur_lr = opt.param_groups[0]['lr']

        if ep % 50 == 0 or ep == 1:
            pairs = s.numel()
            elapsed = time.time() - t0
            print(f"Ep {ep:04d}/{args.episodes} | loss={loss.item():.4f} ema={ema:.4f} "
                  f"| pos_acc={pos_acc:.3f} neg_acc={neg_acc:.3f} | pairs={pairs} | lr={cur_lr:g}")
            logger.log([
                ep, f"{loss.item():.6f}", f"{ema:.6f}",
                f"{pos_acc:.4f}", f"{neg_acc:.4f}", pairs,
                f"{cur_lr:g}", args.batch_images, args.pos_per_image, args.neg_per_image,
                args.emb_dim, int(args.train_backbone), trainable_params, device, f"{elapsed:.2f}"
            ])

        # Save best by instantaneous loss (or use ema if you prefer)
        if loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
            torch.save({'enc': enc.state_dict(), 'rel': rel.state_dict(),
                        'emb_dim': args.emb_dim, 'cfg': vars(args)}, args.checkpoint)

    logger.close()
    print(f"Done. CSV log at {args.log_csv}")

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (K-shot ONLY here)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_paths(paths, tfm, enc):
    if len(paths) == 0: return torch.empty(0, enc.head.out_features, device=device)
    xb = torch.stack([load_img(p, tfm) for p in paths], 0).to(device)
    return enc(xb)

@torch.no_grad()
def evaluate(args):
    normal_paths = list_imgs(args.val_normal)
    anom_paths   = list_imgs(args.val_anom)
    if len(normal_paths) < args.k_shot_eval or len(anom_paths) == 0:
        print("Skip eval: not enough val data.")
        return

    # safe torch.load
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location='cpu')

    enc = Encoder(emb_dim=ckpt['emb_dim']).to(device)
    rel = RelationHead(emb_dim=ckpt['emb_dim']).to(device)
    enc.load_state_dict(ckpt['enc']); rel.load_state_dict(ckpt['rel'])
    enc.eval(); rel.eval()

    aucs, aps = [], []
    best = (-1, None, None)

    for _ in range(max(1, args.eval_runs)):
        # K-shot support (normals)
        sup_paths = sample_paths(normal_paths, args.k_shot_eval)
        sup_Z = embed_paths(sup_paths, eval_tfm, enc)  # [K, D]

        # Queries = remaining normals + anomalies
        rem_norm_paths = [p for p in normal_paths if p not in sup_paths]
        Xn = [load_img(p, eval_tfm) for p in rem_norm_paths] if rem_norm_paths else []
        Xa = [load_img(p, eval_tfm) for p in anom_paths]
        X = Xn + Xa
        if len(X) == 0: continue
        Zq = enc(torch.stack(X, 0).to(device))  # [Nq, D]

        scores = []
        for i in range(Zq.size(0)):
            zq = Zq[i].unsqueeze(0).repeat(sup_Z.size(0), 1)  # [K, D]
            sim = rel(zq, sup_Z)                              # [K]
            scores.append(1.0 - sim.max().item())            # anomaly score

        s = np.array(scores)
        y_true = np.array(([0]*len(Xn)) + ([1]*len(Xa)))
        auc = roc_auc_score(y_true, s); ap = average_precision_score(y_true, s)
        aucs.append(auc); aps.append(ap)
        if auc > best[0]: best = (auc, y_true, s)

    print(f"[val] RN — AUC mean={np.mean(aucs):.4f} ± {np.std(aucs):.4f} | "
          f"AP mean={np.mean(aps):.4f} ± {np.std(aps):.4f}  (runs={args.eval_runs})")

    # Best-run plots
    y, s = best[1], best[2]
    fpr, tpr, _ = roc_curve(y, s)
    ap_best  = average_precision_score(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    os.makedirs('figs', exist_ok=True)
    plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'RN ROC (AUC={best[0]:.3f})'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/rn_roc_curve.png', dpi=220); plt.close()

    plt.figure(); plt.plot(rec,prec)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'RN PR (AP={ap_best:.3f})'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/rn_pr_curve.png', dpi=220); plt.close()

    plt.figure()
    plt.hist(s[y==0], bins=40, alpha=0.6, label='Normal')
    plt.hist(s[y==1], bins=40, alpha=0.6, label='Anomaly')
    plt.legend(); plt.title('RN Score distributions (1 - max relation)'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/rn_score_hist.png', dpi=220); plt.close()
    print("Saved: figs/rn_roc_curve.png, figs/rn_pr_curve.png, figs/rn_score_hist.png")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = get_args()
    train_rn(args)
    evaluate(args)
