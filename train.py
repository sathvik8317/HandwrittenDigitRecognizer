"""Digit Recognizer: small CNN with GPU augmentation, 3-seed ensemble.
Runs on Kaggle (reads /kaggle/input) or locally (reads ./data)."""
import glob, json, math, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

EPOCHS, BS, SEEDS, VAL_FRAC = 30, 128, [0, 1, 2], 0.05
dev = "cuda" if torch.cuda.is_available() else "cpu"


def find(name):
    for root in ("/kaggle/input", "data"):
        hits = glob.glob(f"{root}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


train, test = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
X = torch.tensor(train.drop(columns="label").values, dtype=torch.float32).view(-1, 1, 28, 28) / 255
y = torch.tensor(train.label.values)
Xt = torch.tensor(test.values, dtype=torch.float32).view(-1, 1, 28, 28) / 255

g = torch.Generator().manual_seed(42)
perm = torch.randperm(len(X), generator=g)
n_val = int(len(X) * VAL_FRAC)
vi, ti = perm[:n_val], perm[n_val:]
Xtr, ytr, Xva, yva = X[ti].to(dev), y[ti].to(dev), X[vi].to(dev), y[vi].to(dev)
Xt = Xt.to(dev)


def augment(x):
    b = x.size(0)
    a = (torch.rand(b, device=dev) * 2 - 1) * math.radians(12)
    s = 1 + (torch.rand(b, device=dev) * 2 - 1) * 0.10
    tx, ty = ((torch.rand(b, device=dev) * 2 - 1) * 0.12 for _ in range(2))
    theta = torch.stack([
        torch.stack([s * a.cos(), -s * a.sin(), tx], 1),
        torch.stack([s * a.sin(), s * a.cos(), ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def make_net():
    return nn.Sequential(
        block(1, 32), block(32, 32), nn.MaxPool2d(2), nn.Dropout(0.2),
        block(32, 64), block(64, 64), nn.MaxPool2d(2), nn.Dropout(0.2),
        block(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10)).to(dev)


def norm(x):
    return (x - 0.1307) / 0.3081


@torch.no_grad()
def probs(net, x):
    net.eval()
    return torch.cat([F.softmax(net(norm(x[i:i + 1000])), 1) for i in range(0, len(x), 1000)])


val_probs, test_probs, per_model = [], [], []
for seed in SEEDS:
    torch.manual_seed(seed)
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    steps = EPOCHS * math.ceil(len(Xtr) / BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    t0 = time.time()
    for ep in range(EPOCHS):
        net.train()
        idx = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            loss = F.cross_entropy(net(norm(augment(Xtr[b]))), ytr[b], label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if ep % 5 == 4 or ep == EPOCHS - 1:
            acc = (probs(net, Xva).argmax(1) == yva).float().mean().item()
            print(f"seed {seed} epoch {ep + 1}/{EPOCHS} val acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    pv, pt = probs(net, Xva), probs(net, Xt)
    val_probs.append(pv)
    test_probs.append(pt)
    per_model.append((pv.argmax(1) == yva).float().mean().item())

ens_val = (torch.stack(val_probs).mean(0).argmax(1) == yva).float().mean().item()
print(f"per-model val acc {per_model}, ensemble val acc {ens_val:.4f}")
pred = torch.stack(test_probs).mean(0).argmax(1).cpu().numpy()
pd.DataFrame({"ImageId": np.arange(1, len(pred) + 1), "Label": pred}).to_csv("submission.csv", index=False)
json.dump({"per_model_val_acc": per_model, "ensemble_val_acc": ens_val, "n_val": n_val}, open("results.json", "w"))
