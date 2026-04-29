"""
Exploratory data analysis for the ATLAS Top Tagging Open Data.
Generates scientific-quality plots saved to an output directory.

Usage:
    python scripts/eda.py data/test-public-small.h5
    python scripts/eda.py data/test-public.h5 --output-dir figures/eda --max-jets 50000
"""
import sys
import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import plot_preprocessed_jet_images

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          12,
    "axes.labelsize":     13,
    "axes.titlesize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "legend.framealpha":  0.8,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "lines.linewidth":    1.5,
})

SIG_COLOR = "#d62728"   # red  — signal (top quark)
BKG_COLOR = "#1f77b4"   # blue — background (QCD)
ALPHA     = 0.55
DPI       = 300


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(path, max_jets=None):
    with h5py.File(path, "r") as f:
        n = f["labels"].shape[0]
        if max_jets is not None:
            n = min(n, max_jets)
        return {key: f[key][:n] for key in f.keys()}


# ── Helpers ───────────────────────────────────────────────────────────────────
def pct_bins(values, n=60, lo=1, hi=99):
    return np.linspace(np.percentile(values, lo), np.percentile(values, hi), n)


def dual_hist(ax, sig_vals, bkg_vals, bins, xlabel, ylabel="Normalised density", legend=True):
    kw = dict(bins=bins, density=True, histtype="stepfilled", alpha=ALPHA)
    ax.hist(bkg_vals, color=BKG_COLOR, label="Background", **kw)
    ax.hist(sig_vals, color=SIG_COLOR, label="Signal (top)", **kw)
    ax.hist(bkg_vals, bins=bins, density=True, color=BKG_COLOR, histtype="step", linewidth=1.2)
    ax.hist(sig_vals, bins=bins, density=True, color=SIG_COLOR, histtype="step", linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_class_balance(data, out):
    labels = data["labels"]
    n_sig  = int((labels == 1).sum())
    n_bkg  = int((labels == 0).sum())

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Background (QCD)", "Signal (top quark)"],
        [n_bkg, n_sig],
        color=[BKG_COLOR, SIG_COLOR],
        edgecolor="black", linewidth=0.8, width=0.5,
    )
    ceiling = max(n_bkg, n_sig)
    for bar, count in zip(bars, [n_bkg, n_sig]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ceiling * 0.015,
            f"{count:,}", ha="center", va="bottom",
        )
    ax.set_ylabel("Number of jets")
    ax.set_title("Class distribution")
    ax.set_ylim(0, ceiling * 1.15)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(out / "class_balance.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  class_balance.png")


def plot_jet_kinematics(data, out):
    labels = data["labels"]
    sig, bkg = labels == 1, labels == 0

    variables = [
        ("fjet_pt",  r"Jet $p_T$ [GeV]",   1e-3),
        ("fjet_m",   r"Jet mass [GeV]",     1e-3),
        ("fjet_eta", r"Jet $\eta$",          1.0),
        ("fjet_phi", r"Jet $\phi$ [rad]",    1.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, xlabel, scale) in zip(axes.flat, variables):
        vals = data[key] * scale
        bins = pct_bins(vals)
        dual_hist(ax, vals[sig], vals[bkg], bins, xlabel)

    fig.suptitle("Jet kinematics", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "jet_kinematics.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  jet_kinematics.png")


def plot_substructure(data, out):
    labels = data["labels"]
    sig, bkg = labels == 1, labels == 0

    # (key, xlabel, scale, log10_transform)
    variables = [
        ("fjet_Tau1_wta",  r"$\tau_1^{\mathrm{WTA}}$",          1.0,  False),
        ("fjet_Tau2_wta",  r"$\tau_2^{\mathrm{WTA}}$",          1.0,  False),
        ("fjet_Tau3_wta",  r"$\tau_3^{\mathrm{WTA}}$",          1.0,  False),
        ("fjet_Tau4_wta",  r"$\tau_4^{\mathrm{WTA}}$",          1.0,  False),
        ("fjet_Split12",   r"$\sqrt{d_{12}}$ [GeV]",            1e-3, False),
        ("fjet_Split23",   r"$\sqrt{d_{23}}$ [GeV]",            1e-3, False),
        ("fjet_ECF1",      r"$\log_{10}(\mathrm{ECF}_1)$",      1.0,  True),
        ("fjet_ECF2",      r"$\log_{10}(\mathrm{ECF}_2)$",      1.0,  True),
        ("fjet_ECF3",      r"$\log_{10}(\mathrm{ECF}_3)$",      1.0,  True),
        ("fjet_C2",        r"$C_2$",                             1.0,  False),
        ("fjet_D2",        r"$D_2$",                             1.0,  False),
        ("fjet_L2",        r"$L_2$",                             1.0,  False),
        ("fjet_L3",        r"$L_3$",                             1.0,  False),
        ("fjet_Qw",        r"$Q_W$ [GeV]",                      1e-3, False),
        ("fjet_ThrustMaj", r"Thrust major",                      1.0,  False),
    ]

    n_cols = 5
    n_rows = int(np.ceil(len(variables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.3, n_rows * 3.1))

    for idx, (ax, (key, xlabel, scale, log_x)) in enumerate(zip(axes.flat, variables)):
        raw = data[key]
        if log_x:
            sig_vals = np.log10(raw[sig][raw[sig] > 0])
            bkg_vals = np.log10(raw[bkg][raw[bkg] > 0])
        else:
            sig_vals = raw[sig] * scale
            bkg_vals = raw[bkg] * scale

        bins = pct_bins(np.concatenate([sig_vals, bkg_vals]))
        ylabel = "Density" if idx % n_cols == 0 else ""
        dual_hist(ax, sig_vals, bkg_vals, bins, xlabel, ylabel=ylabel, legend=(idx == 0))

    for ax in axes.flat[len(variables):]:
        ax.set_visible(False)

    fig.suptitle("High-level substructure variables", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "substructure.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  substructure.png")


def plot_constituent_multiplicity(data, out):
    labels = data["labels"]
    sig, bkg = labels == 1, labels == 0

    mult = (data["fjet_clus_pt"] > 0).sum(axis=1)
    bins = np.arange(0, mult.max() + 2)

    fig, ax = plt.subplots(figsize=(7, 5))
    dual_hist(ax, mult[sig], mult[bkg], bins, "Number of constituents")
    ax.set_title("Constituent multiplicity per jet")
    fig.tight_layout()
    fig.savefig(out / "constituent_multiplicity.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  constituent_multiplicity.png")


def plot_leading_constituent_pt(data, out):
    labels = data["labels"]
    sig, bkg = labels == 1, labels == 0

    clus_pt  = data["fjet_clus_pt"] * 1e-3
    n_leading = 15
    ranks     = np.arange(1, n_leading + 1)

    means_sig = clus_pt[sig, :n_leading].mean(axis=0)
    means_bkg = clus_pt[bkg, :n_leading].mean(axis=0)
    stds_sig  = clus_pt[sig, :n_leading].std(axis=0)
    stds_bkg  = clus_pt[bkg, :n_leading].std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ranks, means_bkg, yerr=stds_bkg, fmt="o-", color=BKG_COLOR,
                label="Background", markersize=5, capsize=3)
    ax.errorbar(ranks, means_sig, yerr=stds_sig, fmt="s-", color=SIG_COLOR,
                label="Signal (top)", markersize=5, capsize=3)
    ax.set_xlabel(r"Constituent rank (by $p_T$)")
    ax.set_ylabel(r"Mean $p_T \pm \sigma$ [GeV]")
    ax.set_title(r"$p_T$ spectrum of leading constituents")
    ax.set_xticks(ranks)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "leading_constituent_pt.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  leading_constituent_pt.png")


def plot_jet_images(data, out):
    labels   = data["labels"]
    sig, bkg = labels == 1, labels == 0

    clus_eta = data["fjet_clus_eta"]
    clus_phi = data["fjet_clus_phi"]
    clus_pt  = data["fjet_clus_pt"] * 1e-3

    # centre on the jet axis
    deta = clus_eta - data["fjet_eta"][:, np.newaxis]
    dphi = clus_phi - data["fjet_phi"][:, np.newaxis]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

    eta_range, phi_range, nbins = (-1.2, 1.2), (-1.2, 1.2), 64

    def make_image(mask):
        eta_f = deta[mask].ravel()
        phi_f = dphi[mask].ravel()
        pt_f  = clus_pt[mask].ravel()
        valid = pt_f > 0
        img, _, _ = np.histogram2d(
            eta_f[valid], phi_f[valid],
            bins=nbins, range=[eta_range, phi_range],
            weights=pt_f[valid],
        )
        return img / mask.sum()

    img_bkg = make_image(bkg)
    img_sig = make_image(sig)

    vmin = min(img_bkg[img_bkg > 0].min(), img_sig[img_sig > 0].min())
    vmax = max(img_bkg.max(), img_sig.max())
    extent = [*eta_range, *phi_range]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, img, title in zip(axes, [img_bkg, img_sig],
                               ["Background (QCD)", "Signal (top quark)"]):
        im = ax.imshow(
            img.T, origin="lower", extent=extent,
            aspect="auto", cmap="inferno",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_xlabel(r"$\Delta\eta$")
        ax.set_ylabel(r"$\Delta\phi$ [rad]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=r"Mean $\Sigma\,p_T$ per jet [GeV]")

    fig.suptitle("Average jet images (jet-centred)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "jet_images.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  jet_images.png")


def plot_correlations(data, out):
    labels = data["labels"]

    keys = [
        "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta",
        "fjet_Split12",  "fjet_Split23",
        "fjet_C2",       "fjet_D2",
        "fjet_L2",       "fjet_L3",
        "fjet_Qw",       "fjet_ThrustMaj",
    ]
    tick_labels = [
        r"$\tau_1$", r"$\tau_2$", r"$\tau_3$",
        r"$\sqrt{d_{12}}$", r"$\sqrt{d_{23}}$",
        r"$C_2$", r"$D_2$",
        r"$L_2$", r"$L_3$",
        r"$Q_W$", "Thrust",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, mask, title in zip(
        axes,
        [labels == 0, labels == 1],
        ["Background (QCD)", "Signal (top quark)"],
    ):
        corr = np.corrcoef(np.array([data[k][mask] for k in keys]))
        im   = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticklabels(tick_labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Pearson $r$", shrink=0.82)
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(
                    j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.65 else "black",
                )

    fig.suptitle("Feature correlations", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "correlations.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  correlations.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for ATLAS Top Tagging data")
    parser.add_argument("input_path", type=str, help="Path to the HDF5 file")
    parser.add_argument("--output-dir", type=str, default="figures/eda",
                        help="Output directory for plots (default: figures/eda)")
    parser.add_argument("--max-jets", type=int, default=None,
                        help="Cap number of jets loaded (default: all)")
    parser.add_argument("--preprocessed", type=str, default=None,
                        help="Path to a preprocessed HDF5 file to generate jet image plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input_path} ...")
    data  = load_data(args.input_path, max_jets=args.max_jets)
    n     = data["labels"].shape[0]
    n_sig = int((data["labels"] == 1).sum())
    n_bkg = int((data["labels"] == 0).sum())
    print(f"  {n:,} jets  —  {n_sig:,} signal  /  {n_bkg:,} background")
    print(f"Writing plots to {out_dir}/\n")

    plot_class_balance(data, out_dir)
    plot_jet_kinematics(data, out_dir)
    plot_substructure(data, out_dir)
    plot_constituent_multiplicity(data, out_dir)
    plot_leading_constituent_pt(data, out_dir)
    plot_jet_images(data, out_dir)
    plot_correlations(data, out_dir)

    if args.preprocessed:
        import h5py
        print(f"\nLoading preprocessed file {args.preprocessed} ...")
        with h5py.File(args.preprocessed, "r") as f:
            n = min(args.max_jets, len(f["labels"])) if args.max_jets else len(f["labels"])
            features = f["features"][:n]
            labels   = f["labels"][:n]
        plot_preprocessed_jet_images(features, labels, out_dir / "jet_images_preprocessed.png")
        print("  jet_images_preprocessed.png")

    print("\nDone.")
