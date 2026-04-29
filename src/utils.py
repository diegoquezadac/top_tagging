import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score

_PLOT_STYLE = {
    "font.family":       "serif",
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.titlesize":    13,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   11,
    "legend.framealpha": 0.8,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.5,
}

def train_loop(model, loader, criterion, optimizer, device, l1_lambda=None):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for X, y, w in tqdm(loader, desc="  train", leave=False, unit="batch", total=len(loader)):
        X, y, w = X.to(device), y.to(device).float(), w.to(device).float()
        optimizer.zero_grad()

        outputs = model(X).squeeze(1)

        # criterion must have reduction='none'
        loss = criterion(outputs, y)                     # shape: (batch_size,)
        weighted_loss = (loss * w).sum() / w.sum()       # weighted mean loss

        # Optional L1 regularization
        if l1_lambda:
            l1_reg = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
            weighted_loss += l1_lambda * l1_reg / len(X) # scale L1 by batch size

        weighted_loss.backward()
        optimizer.step()

        running_loss += weighted_loss.item() * X.size(0) # convert back to sum
        total_samples += X.size(0)

    return running_loss / total_samples                  # mean loss per sample


def test_loop(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y, w in tqdm(loader, desc="  val  ", leave=False, unit="batch", total=len(loader)):
            X, y = X.to(device), y.to(device).float()
            outputs = model(X).squeeze(1)

            loss = criterion(outputs, y)                 # shape: (batch_size,)
            running_loss += loss.sum().item()
            total_samples += X.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == y.long()).sum().item()

    mean_loss = running_loss / total_samples
    accuracy = correct / total_samples
    return mean_loss, accuracy


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_logger(name: str):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_metrics(y_true, y_pred, tpr_threshold=0.5):

    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float('nan')

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    idx = np.argmin(np.abs(tpr - tpr_threshold))
    threshold = thresholds[idx]
    actual_tpr = tpr[idx]
    actual_fpr = fpr[idx]

    y_pred_binary = (y_pred >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    inverse_fpr = 1.0 / actual_fpr if actual_fpr > 0 else float("inf")

    return {
        "accuracy": accuracy,
        "auc": auc,
        "recall": recall,
        "precision": precision,
        "tpr": actual_tpr,
        "fpr": actual_fpr,
        "inverse_fpr": inverse_fpr,
        "threshold": threshold,
    }

def load_weights(model, path_to_checkpoint, device):
    checkpoint = torch.load(path_to_checkpoint, map_location=device)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    return model


def plot_loss_curve(history, out_path, title=None):
    plt.rcParams.update(_PLOT_STYLE)

    train_loss = np.array(history["train_loss"])
    val_loss   = np.array(history["val_loss"])
    epochs     = np.arange(1, len(train_loss) + 1)
    best_epoch = int(np.argmin(val_loss)) + 1
    best_val   = float(np.min(val_loss))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, train_loss, color="#1f77b4", label="Training")
    ax.plot(epochs, val_loss,   color="#d62728", label="Validation")
    ax.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8,
               label=f"Best epoch {best_epoch}  (val loss = {best_val:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessed_jet_images(features, labels, out_path, nbins=64):
    """Three-panel figure: example background jet, example signal jet, ratio of averages.

    features: (N, max_constits, 7) — columns [eta, phi, log_pt, log_E, lognorm_pt, lognorm_E, radius]
    labels:   (N,) — 1=signal, 0=background
    """
    plt.rcParams.update(_PLOT_STYLE)

    eta_range = (-2.0, 2.0)
    phi_range = (-2.0, 2.0)
    extent    = [*eta_range, *phi_range]

    sig_mask = labels == 1
    bkg_mask = labels == 0

    n_real = (features[:, :, 4] != 0).sum(axis=1)
    bkg_idx = np.where(bkg_mask)[0][np.argmax(n_real[bkg_mask])]
    sig_idx = np.where(sig_mask)[0][np.argmax(n_real[sig_mask])]

    def jet_scatter(idx):
        eta = features[idx, :, 0]
        phi = features[idx, :, 1]
        lpt = features[idx, :, 4]
        valid = lpt != 0
        return eta[valid], phi[valid], np.exp(lpt[valid])

    def avg_image(mask):
        eta_f = features[mask, :, 0].ravel()
        phi_f = features[mask, :, 1].ravel()
        lpt_f = features[mask, :, 4].ravel()
        valid = lpt_f != 0
        img, _, _ = np.histogram2d(
            eta_f[valid], phi_f[valid],
            bins=nbins, range=[eta_range, phi_range],
            weights=np.exp(lpt_f[valid]),
        )
        return img / mask.sum()

    img_bkg = avg_image(bkg_mask)
    img_sig = avg_image(sig_mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((img_bkg > 0) & (img_sig > 0), img_sig / img_bkg, np.nan)

    eta_b, phi_b, pt_b = jet_scatter(bkg_idx)
    eta_s, phi_s, pt_s = jet_scatter(sig_idx)
    pt_all = np.concatenate([pt_b, pt_s])
    vmin_s, vmax_s = pt_all.min(), pt_all.max()

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    for ax, eta, phi, pt, title in [
        (ax_a, eta_b, phi_b, pt_b, "Background jet (QCD)"),
        (ax_b, eta_s, phi_s, pt_s, r"Signal jet ($Z' \to t\bar{t}$)"),
    ]:
        sc = ax.scatter(
            eta, phi, c=pt, cmap="plasma",
            norm=LogNorm(vmin=vmin_s, vmax=vmax_s),
            s=25, marker="s", linewidths=0,
        )
        ax.set_xlim(*eta_range)
        ax.set_ylim(*phi_range)
        ax.set_xlabel(r"Pre-processed $\eta$")
        ax.set_ylabel(r"Pre-processed $\phi$")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, label=r"Normalized $p_T$")

    valid_ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    r_min, r_max = valid_ratio.min(), valid_ratio.max()
    im = ax_c.imshow(
        ratio.T, origin="lower", extent=extent, aspect="auto",
        cmap="plasma", norm=LogNorm(vmin=r_min, vmax=r_max),
    )
    ax_c.set_xlabel(r"Pre-processed $\eta$")
    ax_c.set_ylabel(r"Pre-processed $\phi$")
    ax_c.set_title("Ratio: average signal / background jet image")
    fig.colorbar(im, ax=ax_c, label=r"Ratio of normalized $p_T$")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rejection_vs_variable(y_pred, y_true, variable, xlabel, out_path,
                                n_bins=10, tpr_thresholds=(0.5, 0.8)):
    """Background rejection (1/FPR) vs a binned physical variable.

    Top panel: step plot with shaded Poisson uncertainty bands for each TPR threshold.
    Bottom panel: ratio of rejection at TPR=0.5 to TPR=0.8.
    """
    plt.rcParams.update(_PLOT_STYLE)

    bin_edges   = np.percentile(variable, np.linspace(5, 95, n_bins + 1))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    colors      = ["#1f77b4", "#d62728"]

    all_rej, all_err = [], []
    for tpr_thresh in tpr_thresholds:
        rejections, rej_errs = [], []
        for i in range(len(bin_edges) - 1):
            mask  = (variable >= bin_edges[i]) & (variable < bin_edges[i + 1])
            n_bkg = int(((y_true == 0) & mask).sum())
            if mask.sum() < 20 or n_bkg < 5:
                rejections.append(np.nan)
                rej_errs.append(np.nan)
                continue
            fpr, tpr, _ = roc_curve(y_true[mask], y_pred[mask])
            idx        = np.argmin(np.abs(tpr - tpr_thresh))
            actual_fpr = max(fpr[idx], 1e-9)
            rejection  = 1.0 / actual_fpr
            rej_err    = rejection ** 1.5 / np.sqrt(n_bkg)
            rejections.append(rejection)
            rej_errs.append(rej_err)
        all_rej.append(np.array(rejections, dtype=float))
        all_err.append(np.array(rej_errs,   dtype=float))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    for tpr_thresh, color, rej, err in zip(tpr_thresholds, colors, all_rej, all_err):
        valid = ~np.isnan(rej)
        x = bin_centers[valid]
        r = rej[valid]
        e = err[valid]
        ax_top.step(x, r, where="mid", color=color, linewidth=1.5,
                    label=rf"$\varepsilon_{{sig}}$ = {tpr_thresh:.1f}", zorder=3)
        ax_top.fill_between(x, r - e, r + e, alpha=0.25, color=color,
                            step="mid", zorder=2)

    # ratio: TPR=0.5 / TPR=0.8
    r0, r1 = all_rej[0], all_rej[1]
    e0, e1 = all_err[0], all_err[1]
    valid = ~np.isnan(r0) & ~np.isnan(r1) & (r1 > 0)
    ratio     = np.where(valid, r0 / r1, np.nan)
    ratio_err = np.where(valid, ratio * np.sqrt((e0 / np.where(r0 > 0, r0, np.nan)) ** 2 +
                                                 (e1 / np.where(r1 > 0, r1, np.nan)) ** 2), np.nan)
    x_r = bin_centers[valid]
    ax_bot.step(x_r, ratio[valid], where="mid", color="black", linewidth=1.2)
    ax_bot.fill_between(x_r,
                        (ratio - ratio_err)[valid], (ratio + ratio_err)[valid],
                        alpha=0.25, color="black", step="mid")
    ax_bot.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax_bot.set_ylabel(r"$\varepsilon_{0.5}\,/\,\varepsilon_{0.8}$", fontsize=10)
    ax_bot.set_xlabel(xlabel)

    ax_top.set_ylabel(r"Background rejection $\varepsilon_{bkg}^{-1}$")
    ax_top.set_yscale("log")
    ax_top.legend()
    ax_top.tick_params(labelbottom=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)