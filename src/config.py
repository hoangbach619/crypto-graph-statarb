"""Project-wide configuration. All scripts import from here."""
from pathlib import Path

SEED = 42

# Data window
DATA_START = "2020-01-01"
OOS_START = "2022-01-01"

# Portfolio construction
N_LONGS = 20
N_SHORTS = 20
FORWARD_HORIZON_DAYS = 21
REBALANCE_FREQ = "W-FRI"

# Costs (basis points)
TAKER_FEE_BPS = 2   # one-way; round-trip = 4 bps
SLIPPAGE_BPS = 2    # per leg

# Graph / ML
KNN_K = 5
LASSO_CV_FOLDS = 3
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 5
RF_MAX_FEATURES = 0.6
LAPLACIAN_ALPHA = 0.7
CORR_WINDOW = 60

# Annualisation factor (crypto trades 365 days)
ANNUALISATION_FACTOR = 365

# Universe construction
UNIVERSE_SIZE = 30
LISTING_GRACE_DAYS = 365  # accept symbols that listed within N days of DATA_START
EXCLUDED_BASES = {
    "BUSD", "USDC", "FDUSD", "TUSD", "DAI", "USDP", "PYUSD", "USDD", "USTC",
    "WBTC", "WETH", "WBNB", "WSOL",
}
EXCLUDED_PREFIXES = ("1000",)

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TBL_DIR = RESULTS_DIR / "tables"

for _d in [RAW_DIR, PROCESSED_DIR, FIG_DIR, TBL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
