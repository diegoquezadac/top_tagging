# Data

## Source

This project uses the [ATLAS Top Tagging Open Data](https://opendata.cern.ch/record/15013) from CERN OpenData. Each entry is a large-radius jet labelled as top quark (`1`) or background (`0`).

## Datasets

The following table describes the relevant datasets:

| Dataset | Size | [OpenData](https://opendata.cern.ch/record/15013) | tarron | euler |
|---------|------|---------------------------------------------------|--------|-------|
| `train-public.h5` | 130G | Yes | `/home/rafa/respaldoRaquel/train-public.h5` | `/mnt/storage/rpezoa/train-public.h5` |
| `test-public.h5` | 7.6G | Yes | `/home/raquel/data/test-public.h5` | `/mnt/storage/rpezoa/test-public.h5` |
| `raw_string.h5` | 19G | No | `/home/raquel/data/raw_string.h5` | `/mnt/storage/rpezoa/raw_string.h5` |
| `raw_cluster.h5` | 19G | No | `/home/raquel/data/raw_cluster.h5` | `/mnt/storage/rpezoa/raw_cluster.h5` |
| `raw_angular.h5` | 19G | No | `/home/raquel/data/raw_angular.h5` | `/mnt/storage/rpezoa/raw_angular.h5` |
| `raw_dipole.h5` | — | No | `/home/raquel/data/raw_dipole.h5` | `/mnt/storage/rpezoa/raw_dipole.h5` |


## Variables

Each jet contains three groups of variables:

1. **Jet four-vector** — one scalar per jet (`fjet_eta`, `fjet_phi`, `fjet_pt`, `fjet_m`)

2. **Constituent four-vectors** — up to 200 constituents per jet, sorted by pt descending and zero-padded:

   | Variable | Description | Datasets |
   |----------|-------------|----------|
   | `fjet_clus_eta` | Pseudorapidity | all |
   | `fjet_clus_phi` | Azimuthal angle | all |
   | `fjet_clus_pt` | Transverse momentum | all |
   | `fjet_clus_E` | Energy | all |
   | `fjet_clus_taste` | Cluster type flag (integer) | `raw_*.h5` only |

   The actual number of real constituents varies per jet (the rest are zeros). In practice only the leading **80** are used, and **3 derived features** are added during preprocessing, giving **7 features per constituent**.

3. **High-level substructure quantities** — one scalar per jet:

   | Variable | Description |
   |----------|-------------|
   | `fjet_Tau1_wta` – `fjet_Tau4_wta` | N-subjettiness (WTA axes) |
   | `fjet_Split12`, `fjet_Split23` | Splitting scales |
   | `fjet_ECF1` – `fjet_ECF3` | Energy correlation functions |
   | `fjet_C2`, `fjet_D2` | ECF ratios |
   | `fjet_L2`, `fjet_L3` | Generalised angularities |
   | `fjet_Qw` | W-boson momentum fraction |
   | `fjet_ThrustMaj` | Thrust major |


## Preprocessing

The preprocessing follows the same logic described in the [original ATLAS implementation](https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/blob/master/utils.py?ref_type=heads), but adapted to work in batches (default size: 100,000) so it can handle the large training dataset on machines with standard memory.

The variables `lognorm_pt` and `lognorm_energy` require global sums over the full dataset. The `compute_stats` function pre-computes these before batched processing begins, avoiding the need to load the entire column at once.
