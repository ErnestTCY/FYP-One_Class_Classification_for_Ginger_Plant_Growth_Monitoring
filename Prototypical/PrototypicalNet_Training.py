#!/usr/bin/env python3
import os, random, math
from glob import glob
from PIL import Image
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# ─────────────────────────────────────
# CLI
# ─────────────────────────────────────
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_dir', default='dataset/train/Normal')
    ap.add_argument('--val_normal', default='dataset/val/Normal')
    ap.add_argument('--val_anom',   default='dataset/val/Abnormal')  # fixed default
    ap.add_argument('--emb_dim', type=int, default=128)

    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--k_shot',   type=int, default=5, help='1..5 typical')
    ap.add_argument('--q_queries',type=int, default=32)

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)

    ap.add_argument('--checkpoint', default='checkpoints/RM_5shot.pth')
    ap.add_argument('--seed', type=int, default=0)

    # Anti-collapse & robust proto
    ap.add_argument('--var_gamma', type=float, default=0.5, help='variance regularizer weight; 0 to disable')
    ap.add_argument('--use_margin', action='store_true', help='enable two-prototype margin term')
    ap.add_argument('--margin_gamma', type=float, default=0.5, help='weight for margin loss')
    ap.add_argument('--margin', type=float, default=0.2, help='margin for two-prototype loss')
    ap.add_argument('--trim', type=float, default=0.0, help='trim ratio for robust prototype (0..0.4 recommended)')

    # Optional: lighter aug for debugging separation
    ap.add_argument('--light_aug', action='store_true', help='use lighter train augmentations')

    return ap.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ─────────────────────────────────────
# Transforms
# ─────────────────────────────────────
def make_train_tfm(light=False):
    if light:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

eval_tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ─────────────────────────────────────
# Data helpers
# ─────────────────────────────────────
def list_imgs(root):
    if not os.path.isdir(root): return []
    return sorted([p for p in glob(os.path.join(root, '*')) if os.path.isfile(p)])

def load_imgs(paths, tfm):
    return torch.stack([tfm(Image.open(p).convert('RGB')) for p in paths], 0)

# ─────────────────────────────────────
# Encoder (frozen ResNet18 + small head)
# ─────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, emb_dim=128, train_backbone=False):
        super().__init__()
        # Handle torchvision API changes gracefully
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

    def forward(self, x):
        feat = self.backbone(x)
        z = self.head(feat)
        return z

# ─────────────────────────────────────
# Few-shot helpers
# ─────────────────────────────────────
def sample_episode(normal_paths, k_shot, q_queries):
    # sample support (K) and query (Q)
    if len(normal_paths) < k_shot + q_queries:
        sup = random.choices(normal_paths, k=k_shot)
        qry = random.choices(normal_paths, k=q_queries)
    else:
        idx = random.sample(range(len(normal_paths)), k_shot + q_queries)
        sup = [normal_paths[i] for i in idx[:k_shot]]
        qry = [normal_paths[i] for i in idx[k_shot:]]
    return sup, qry

def sample_support_only(normal_paths, k_shot):
    if len(normal_paths) < k_shot:
        return random.choices(normal_paths, k=k_shot)
    return random.sample(normal_paths, k_shot)

@torch.no_grad()
def robust_proto_from_support(encoder, x_sup, trim=0.0):
    """Mean of normalized embeddings with optional trimming of farthest shots."""
    z_sup = F.normalize(encoder(x_sup.to(device)), dim=1)   # [K,d]
    if z_sup.size(0) == 1 or trim <= 0:
        p = z_sup.mean(0, keepdim=True)
        return F.normalize(p, dim=1)
    mu = z_sup.mean(0, keepdim=True)
    d = ((z_sup - mu)**2).sum(1)                            # [K]
    k_keep = max(1, int(z_sup.size(0) * (1 - min(trim, 0.49))))
    keep = d.topk(k_keep, largest=False).indices
    p = z_sup[keep].mean(0, keepdim=True)
    return F.normalize(p, dim=1)

def variance_regularizer(z, gamma=0.5, eps=1e-4):
    """Push per-dimension stddev upward to prevent collapse."""
    if gamma <= 0: return z.new_tensor(0.0)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)  # [d]
    return gamma * torch.relu(1.0 - std).mean()

# ─────────────────────────────────────
# Losses
# ─────────────────────────────────────
def proto_loss(encoder, x_sup, x_qry, var_gamma=0.5, trim=0.0):
    p = robust_proto_from_support(encoder, x_sup, trim=trim)  # [1,d]
    z_q = F.normalize(encoder(x_qry.to(device)), dim=1)
    pull = (z_q - p).pow(2).sum(dim=1).mean()
    var  = variance_regularizer(z_q, gamma=1.0)  # unscaled here; we scale outside
    loss = pull + var_gamma * var
    return loss, {'pull': pull.item(), 'var': var.item()}, (z_q, p)

def two_proto_margin_loss(encoder, x_sup_a, x_sup_b, x_q, margin=0.2, margin_gamma=0.5,
                          var_gamma=0.5, trim=0.0):
    pA = robust_proto_from_support(encoder, x_sup_a, trim=trim)
    pB = robust_proto_from_support(encoder, x_sup_b, trim=trim)
    zq = F.normalize(encoder(x_q.to(device)), dim=1)
    dA = (zq - pA).pow(2).sum(dim=1)
    dB = (zq - pB).pow(2).sum(dim=1)
    pull = dA.mean()
    rel  = torch.relu(margin - (dB - dA)).mean()
    var  = variance_regularizer(zq, gamma=1.0)  # unscaled
    loss = pull + margin_gamma * rel + var_gamma * var
    return loss, {'pull': pull.item(), 'rel': rel.item(), 'var': var.item()}, (zq, pA)

# ─────────────────────────────────────
# Train
# ─────────────────────────────────────
def train(args):
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    train_tfm = make_train_tfm(args.light_aug)
    normal_train = list_imgs(args.train_dir)
    assert len(normal_train) >= max(args.k_shot, 1), "Need more normal images in train_dir."

    encoder = Encoder(emb_dim=args.emb_dim, train_backbone=False).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Training ProtoNet (1-way {args.k_shot}-shot)… episodes={args.episodes}")
    best_loss, ema = float('inf'), None

    for ep in range(1, args.episodes+1):
        if args.use_margin:
            Sa = load_imgs(sample_support_only(normal_train, args.k_shot), train_tfm)
            Sb = load_imgs(sample_support_only(normal_train, args.k_shot), train_tfm)
            Qa = load_imgs(sample_support_only(normal_train, args.q_queries), train_tfm)
            loss, parts, (z_q, p_main) = two_proto_margin_loss(
                encoder, Sa, Sb, Qa,
                margin=args.margin, margin_gamma=args.margin_gamma,
                var_gamma=args.var_gamma, trim=args.trim
            )
        else:
            sup_paths, qry_paths = sample_episode(normal_train, args.k_shot, args.q_queries)
            x_sup = load_imgs(sup_paths, train_tfm)
            x_qry = load_imgs(qry_paths, train_tfm)
            loss, parts, (z_q, p_main) = proto_loss(
                encoder, x_sup, x_qry, var_gamma=args.var_gamma, trim=args.trim
            )

        encoder.train()
        opt.zero_grad(); loss.backward(); opt.step()

        # health metrics (every 100)
        if ep % 100 == 0:
            with torch.no_grad():
                mean_std = z_q.std(dim=0, unbiased=False).mean().item()
                cos = torch.mm(z_q, z_q.t())
                avg_offdiag = (cos.sum() - cos.diag().sum()) / (z_q.size(0)**2 - z_q.size(0))
                if args.use_margin:
                    print(f"  health: embed_std_mean={mean_std:.3f}  avg_cos_offdiag={avg_offdiag:.3f}  "
                          f"pull={parts['pull']:.4f}  rel={parts['rel']:.4f}  var={parts['var']:.4f} (γv={args.var_gamma}, γm={args.margin_gamma})")
                else:
                    print(f"  health: embed_std_mean={mean_std:.3f}  avg_cos_offdiag={avg_offdiag:.3f}  "
                          f"pull={parts['pull']:.4f}  var={parts['var']:.4f} (γv={args.var_gamma})")

        # track & save
        val = loss.item()
        ema = val if ema is None else 0.9*ema + 0.1*val
        if ep % 50 == 0:
            print(f"Episode {ep:05d}/{args.episodes}  loss={ema:.4f}")

        if val < best_loss:
            best_loss = val
            torch.save({'state_dict': encoder.state_dict(),
                        'emb_dim': args.emb_dim,
                        'cfg': vars(args)}, args.checkpoint)

    print("Done.")

# ─────────────────────────────────────
# Evaluation: AUC + ROC/PR plots + hist
# ─────────────────────────────────────
@torch.no_grad()
def score_queries(encoder, p, xb):
    z = F.normalize(encoder(xb.to(device)), dim=1)
    return (z - p).pow(2).sum(dim=1).cpu().numpy()  # higher = more anomalous

def evaluate(args):
    normal_paths = list_imgs(args.val_normal)
    anom_paths   = list_imgs(args.val_anom)
    if len(normal_paths) < args.k_shot or len(anom_paths) == 0:
        print("Skip eval: not enough val data.")
        return

    # safer torch.load; fallback for older torch
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location='cpu')

    encoder = Encoder(emb_dim=ckpt['emb_dim']).to(device)
    encoder.load_state_dict(ckpt['state_dict'])
    encoder.eval()

    # Build prototype from K support normals
    sup_paths = random.sample(normal_paths, args.k_shot)
    x_sup = load_imgs(sup_paths, eval_tfm)
    p = robust_proto_from_support(encoder, x_sup, trim=args.trim)

    # Use remaining normals as normal queries
    rem_norm_paths = [pp for pp in normal_paths if pp not in sup_paths]
    x_n = load_imgs(rem_norm_paths, eval_tfm) if rem_norm_paths else None
    x_a = load_imgs(anom_paths, eval_tfm)

    y_true, y_score = [], []
    if x_n is not None and x_n.size(0) > 0:
        s_n = score_queries(encoder, p, x_n)             # normals
        y_true += [0]*len(s_n); y_score += list(s_n)
    s_a = score_queries(encoder, p, x_a)                  # anomalies
    y_true += [1]*len(s_a); y_score += list(s_a)

    y_true = np.array(y_true); y_score = np.array(y_score)

    auc = roc_auc_score(y_true, y_score)
    auc_inv = roc_auc_score(y_true, -y_score)
    print(f"[val] AUC={auc:.4f}  AUC_inverted={auc_inv:.4f}")
    if auc_inv > auc:
        print("  Note: inverted scores perform better; consider using -score for anomaly ranking.")

    # Curves & plots
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    os.makedirs('figs', exist_ok=True)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0,1],[0,1],'--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC (AUC={max(auc,auc_inv):.3f})'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/roc_curve.png', dpi=220); plt.close()

    # PR
    plt.figure()
    plt.plot(rec, prec, lw=2)
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'PR (AP={ap:.3f})'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/pr_curve.png', dpi=220); plt.close()

    # Hist
    plt.figure()
    plt.hist(y_score[y_true==0], bins=40, alpha=0.6, label='Normal')
    plt.hist(y_score[y_true==1], bins=40, alpha=0.6, label='Anomaly')
    plt.legend(); plt.title('Score distributions'); plt.grid(True, alpha=0.3)
    plt.savefig('figs/score_hist.png', dpi=220); plt.close()

    print("Saved plots: figs/roc_curve.png, figs/pr_curve.png, figs/score_hist.png")

# ─────────────────────────────────────
if __name__ == '__main__':
    args = get_args()
    train(args)
    evaluate(args)
