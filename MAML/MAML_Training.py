#!/usr/bin/env python3
import os, random, math, argparse
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

    # MAML (Reptile-style) meta-training
    ap.add_argument('--episodes', type=int, default=500, help='meta-iterations (outer steps)')
    ap.add_argument('--tasks_per_meta_batch', type=int, default=2, help='tasks per outer step')
    ap.add_argument('--k_shot', type=int, default=5, help='1..5')
    ap.add_argument('--q_queries', type=int, default=32)

    # Inner-loop (support adaptation)
    ap.add_argument('--inner_steps', type=int, default=1)
    ap.add_argument('--inner_lr', type=float, default=5e-3)

    # Outer meta update (Reptile)
    ap.add_argument('--meta_lr', type=float, default=1e-3)

    # Regularizers / robustness
    ap.add_argument('--var_gamma', type=float, default=0.3, help='variance regularizer for inner loss (0 to disable)')
    ap.add_argument('--trim', type=float, default=0.0, help='trim ratio for robust prototype in eval (0..0.4)')

    # Eval
    ap.add_argument('--eval_runs', type=int, default=20, help='repeat eval with different support draws')

    ap.add_argument('--checkpoint', default='checkpoints/VG_maml_5shots.pth')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--light_aug', action='store_true', help='lighter train augs (debug)')

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

def load_imgs(paths, tfm):
    return torch.stack([tfm(Image.open(p).convert('RGB')) for p in paths], 0)

def sample_support(paths, k):
    if len(paths) < k: return random.choices(paths, k=k)
    return random.sample(paths, k)

def sample_episode(paths, k, q):
    if len(paths) < k + q:
        S = random.choices(paths, k=k)
        Q = random.choices(paths, k=q)
    else:
        idx = random.sample(range(len(paths)), k+q)
        S = [paths[i] for i in idx[:k]]
        Q = [paths[i] for i in idx[k:]]
    return S, Q

# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    """Frozen ResNet18 trunk + small trainable head. Head is adapted in the inner loop."""
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

    def forward(self, x, head_weight=None, head_bias=None):
        feat = self.backbone(x)
        if head_weight is None:
            return self.head(feat)
        return F.linear(feat, head_weight, head_bias)

# ──────────────────────────────────────────────────────────────────────────────
# Loss bits
# ──────────────────────────────────────────────────────────────────────────────
def variance_regularizer(z, gamma=0.3, eps=1e-4):
    if gamma <= 0: return z.new_tensor(0.0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return gamma * torch.relu(1.0 - std).mean()

def proto_from_support_z(z_sup):
    # z_sup already normalized
    p = z_sup.mean(0, keepdim=True)
    return F.normalize(p, dim=1)

def center_loss_from_proto(z, p):
    # both normalized
    return (z - p).pow(2).sum(1).mean()

# ──────────────────────────────────────────────────────────────────────────────
# Inner loop (support adaptation) — updates only the head (w,b)
# Reptile-style: returns adapted weights (no higher-order gradients)
# ──────────────────────────────────────────────────────────────────────────────
def inner_adapt_support(encoder, x_sup, inner_lr=5e-3, inner_steps=1, var_gamma=0.3):
    w = encoder.head.weight.detach().clone().requires_grad_(True)
    b = encoder.head.bias.detach().clone().requires_grad_(True)

    for _ in range(inner_steps):
        z_sup = F.normalize(encoder(x_sup.to(device), head_weight=w, head_bias=b), dim=1)
        p = proto_from_support_z(z_sup)
        loss = center_loss_from_proto(z_sup, p) + variance_regularizer(z_sup, gamma=var_gamma)
        gw, gb = torch.autograd.grad(loss, [w, b], create_graph=False)
        w = (w - inner_lr * gw).detach().requires_grad_(True)
        b = (b - inner_lr * gb).detach().requires_grad_(True)
    return w.detach(), b.detach()

# ──────────────────────────────────────────────────────────────────────────────
# Meta-train (Reptile update)
# ──────────────────────────────────────────────────────────────────────────────
def meta_train(args):
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    train_paths = list_imgs(args.train_dir)
    assert len(train_paths) >= max(args.k_shot, 1), "Need more normal images in train_dir."

    train_tfm = make_train_tfm(args.light_aug)
    enc = Encoder(emb_dim=args.emb_dim, train_backbone=False).to(device)

    print(f"Training MAML/Reptile (1-way {args.k_shot}-shot)… episodes={args.episodes}, tasks/mb={args.tasks_per_meta_batch}")
    best_seen = float('inf')
    ema = None

    for ep in range(1, args.episodes+1):
        # Accumulate deltas over a meta-batch of tasks
        dw_acc = torch.zeros_like(enc.head.weight.data)
        db_acc = torch.zeros_like(enc.head.bias.data)

        pull_log, var_log = 0.0, 0.0

        for _ in range(args.tasks_per_meta_batch):
            S_paths, Q_paths = sample_episode(train_paths, args.k_shot, args.q_queries)
            x_sup = load_imgs(S_paths, train_tfm).to(device)
            x_qry = load_imgs(Q_paths, train_tfm).to(device)

            # 1) Copy current head; inner adapt on support
            w0 = enc.head.weight.data.detach().clone()
            b0 = enc.head.bias.data.detach().clone()
            w_adapt, b_adapt = inner_adapt_support(enc, x_sup,
                                                   inner_lr=args.inner_lr,
                                                   inner_steps=args.inner_steps,
                                                   var_gamma=args.var_gamma)
            # 2) (optional) log query loss after adapt
            with torch.no_grad():
                z_sup = F.normalize(enc(x_sup, head_weight=w_adapt, head_bias=b_adapt), dim=1)
                p = proto_from_support_z(z_sup)
                z_q = F.normalize(enc(x_qry, head_weight=w_adapt, head_bias=b_adapt), dim=1)
                pull = center_loss_from_proto(z_q, p).item()
                var  = variance_regularizer(z_q, gamma=1.0).item()
                pull_log += pull; var_log += var

            # 3) Reptile meta-update direction: move (w0,b0) toward (w_adapt,b_adapt)
            dw_acc += (w_adapt - w0)
            db_acc += (b_adapt - b0)

        # Outer step (averaged over tasks)
        enc.head.weight.data += args.meta_lr * (dw_acc / args.tasks_per_meta_batch)
        enc.head.bias.data   += args.meta_lr * (db_acc / args.tasks_per_meta_batch)

        # A simple scalar to track progress
        val = (pull_log / args.tasks_per_meta_batch)
        ema = val if ema is None else 0.9*ema + 0.1*val
        if ep % 50 == 0:
            print(f"Episode {ep:05d}/{args.episodes}  query_pull≈{ema:.4f}  (after adapt)   "
                  f"mean_var={var_log/args.tasks_per_meta_batch:.4f}")

        # Save best (by moving average of query pull)
        if ema < best_seen:
            best_seen = ema
            torch.save({'state_dict': enc.state_dict(),
                        'emb_dim': args.emb_dim,
                        'cfg': vars(args)}, args.checkpoint)

    print("Done.")

# ──────────────────────────────────────────────────────────────────────────────
# Evaluation (meta-adapt on K shots, then score queries; repeat eval_runs times)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def robust_proto_from_support(enc, x_sup, w, b, trim=0.0):
    z_sup = F.normalize(enc(x_sup.to(device), head_weight=w, head_bias=b), dim=1)
    if z_sup.size(0) == 1 or trim <= 0:
        p = z_sup.mean(0, keepdim=True)
        return F.normalize(p, dim=1)
    mu = z_sup.mean(0, keepdim=True)
    d  = ((z_sup - mu)**2).sum(1)
    k_keep = max(1, int(z_sup.size(0) * (1 - min(trim, 0.49))))
    keep = d.topk(k_keep, largest=False).indices
    p = z_sup[keep].mean(0, keepdim=True)
    return F.normalize(p, dim=1)

@torch.no_grad()
def score_queries(enc, P, xb, w, b):
    z = F.normalize(enc(xb.to(device), head_weight=w, head_bias=b), dim=1)
    d2 = (z - P).pow(2).sum(1)
    return d2.cpu().numpy()  # higher = more anomalous

def evaluate(args):
    normal_paths = list_imgs(args.val_normal)
    anom_paths   = list_imgs(args.val_anom)
    if len(normal_paths) < args.k_shot or len(anom_paths) == 0:
        print("Skip eval: not enough val data.")
        return

    # safe torch.load
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location='cpu')

    enc = Encoder(emb_dim=ckpt['emb_dim']).to(device)
    enc.load_state_dict(ckpt['state_dict'])
    enc.eval()

    aucs, aps = [], []
    best = (-1, None, None)  # (auc, y, s) for plotting

    runs = max(1, args.eval_runs)
    for r in range(runs):
        # 1) Sample K support and adapt head on those K images (inner loop)
        sup_paths = sample_support(normal_paths, args.k_shot)
        x_sup = load_imgs(sup_paths, eval_tfm).to(device)
        w_adapt, b_adapt = inner_adapt_support(enc, x_sup,
                                               inner_lr=args.inner_lr,
                                               inner_steps=args.inner_steps,
                                               var_gamma=args.var_gamma)

        # 2) Build prototype from adapted support
        P = robust_proto_from_support(enc, x_sup, w_adapt, b_adapt, trim=args.trim)

        # 3) Score queries (remaining normals + all anomalies)
        rem_norm_paths = [p for p in normal_paths if p not in sup_paths]
        x_n = load_imgs(rem_norm_paths, eval_tfm).to(device) if rem_norm_paths else None
        x_a = load_imgs(anom_paths, eval_tfm).to(device)

        y_true, y_score = [], []
        if x_n is not None and x_n.size(0) > 0:
            s_n = score_queries(enc, P, x_n, w_adapt, b_adapt)
            y_true += [0]*len(s_n); y_score += list(s_n)
        s_a = score_queries(enc, P, x_a, w_adapt, b_adapt)
        y_true += [1]*len(s_a); y_score += list(s_a)

        y = np.array(y_true); s = np.array(y_score)
        auc = roc_auc_score(y, s)
        ap  = average_precision_score(y, s)
        aucs.append(auc); aps.append(ap)
        if auc > best[0]: best = (auc, y, s)

    print(f"[val] MAML — AUC mean={np.mean(aucs):.4f} ± {np.std(aucs):.4f} | "
          f"AP mean={np.mean(aps):.4f} ± {np.std(aps):.4f}  (runs={runs})")

    # plots from best run
    y, s = best[1], best[2]
    fpr, tpr, _ = roc_curve(y, s)
    ap_best  = average_precision_score(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    os.makedirs('figs', exist_ok=True)
    plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC (AUC={best[0]:.3f})'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/roc_curve.png', dpi=220); plt.close()

    plt.figure(); plt.plot(rec,prec)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR (AP={ap_best:.3f})'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/pr_curve.png', dpi=220); plt.close()

    plt.figure(); 
    plt.hist(s[y==0], bins=40, alpha=0.6, label='Normal')
    plt.hist(s[y==1], bins=40, alpha=0.6, label='Anomaly')
    plt.legend(); plt.title('Score distributions'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/score_hist.png', dpi=220); plt.close()
    print("Saved plots: figs/roc_curve.png, figs/pr_curve.png, figs/score_hist.png")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = get_args()
    meta_train(args)
    evaluate(args)
