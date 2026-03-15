import os
import glob
import time
import hashlib
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =========================================================
# SETTINGS
# =========================================================
USE_SAVED_MODELS = True

# If you want to force retrain, set True
FORCE_RETRAIN = False

# Cache the final feature table (245k rows) to avoid re-reading 113M rows every run
USE_FEATURE_CACHE = False

BASE_DIR = os.getcwd()
IN_DIR = os.path.join(BASE_DIR, "내국수단")
ADMI_PATH = os.path.join(BASE_DIR, "ADMI_202601.csv")

PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_LIN_PATH = os.path.join(BASE_DIR, "linear_model.pkl")
MODEL_RF_PATH  = os.path.join(BASE_DIR, "random_forest_model.pkl")
COLS_PATH      = os.path.join(BASE_DIR, "feature_columns.pkl")
META_PATH      = os.path.join(BASE_DIR, "artifacts_meta.pkl")
METRICS_PATH   = os.path.join(BASE_DIR, "metrics.pkl")

FEATURES_CACHE_PATH = os.path.join(BASE_DIR, "features_cache.pkl")
FEATURES_CACHE_META = os.path.join(BASE_DIR, "features_cache_meta.pkl")

print("[INFO] BASE_DIR:", BASE_DIR)
print("[INFO] IN_DIR exists?:", os.path.isdir(IN_DIR))
print("[INFO] ADMI exists?:", os.path.isfile(ADMI_PATH))

FEATURE_VERSION = "v5_lag_roll_shift1_sin_cos_weekend_plots"

# =========================================================
# UTILS
# =========================================================
def dataset_fingerprint(file_paths):
    h = hashlib.sha256()
    for fp in sorted(file_paths):
        st = os.stat(fp)
        h.update(fp.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(int(st.st_mtime)).encode("utf-8"))
    return h.hexdigest()

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)

def safe_smape(y_true, y_pred):
    # kept as-is (not printed anymore), since you asked not to change unrelated code
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-6)) * 100)

def evaluate_return(y_true, y_pred):
    """
    Performance identifiers reduced to: MAE, MAPE, R2 only.
    """
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    return {
        "mae": mae,
        "mape_pct": mape,
        "r2": r2
    }

def print_metrics(title, m):
    """
    Print only: MAE, MAPE, R2.
    """
    print(f"\n{title}")
    print(f"MAE  : {m['mae']:.2f}")
    print(f"MAPE : {m['mape_pct']:.2f}%")
    print(f"R2   : {m['r2']:.4f}")

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

# =========================================================
# 1) FILE LIST + DATA FINGERPRINT
# =========================================================
files = sorted(glob.glob(os.path.join(IN_DIR, "in_*.csv")))
if not files:
    raise FileNotFoundError(f"[ERROR] No in_*.csv files found in: {IN_DIR}")

data_fp = dataset_fingerprint(files)
print(f"[INFO] in_*.csv files: {len(files)}")
print(f"[INFO] dataset fingerprint: {data_fp[:12]}...")

# =========================================================
# 2) LOAD ADMI + optionally label mapping (if columns exist)
# =========================================================
admi = pd.read_csv(ADMI_PATH, dtype="string")

if "SIDO_NM" not in admi.columns or "ADMI_CD" not in admi.columns:
    raise ValueError("[ERROR] ADMI file must include at least ADMI_CD and SIDO_NM columns.")

seoul_admi = set(admi.loc[admi["SIDO_NM"].str.contains("서울", na=False), "ADMI_CD"].dropna())

# For labeling (optional)
label_col = None
for cand in ["FULL_NM", "ADMI_NM", "SGG_NM"]:
    if cand in admi.columns:
        label_col = cand
        break

admi_label = None
if label_col is not None:
    admi_label = (
        admi[["ADMI_CD", label_col]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"ADMI_CD": "d_admdong_cd", label_col: "d_label"})
    )

# =========================================================
# 3) BUILD (or LOAD) FEATURE TABLE
# =========================================================
def build_features_from_raw():
    t0 = time.time()
    dfs = []
    for fp in files:
        tmp = pd.read_csv(
            fp,
            dtype={
                "d_admdong_cd": "string",
                "fns_time_cd": "string",
                "total_cnt": "float",
                "etl_ymd": "string",
            },
            usecols=["d_admdong_cd", "fns_time_cd", "total_cnt", "etl_ymd"]
        )
        dfs.append(tmp)

    df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] loaded rows: {len(df):,} ({time.time()-t0:.1f}s)")

    # Seoul destination only
    df = df[df["d_admdong_cd"].isin(seoul_admi)].copy()

    # time -> hour
    def time_to_hour(x):
        x = str(x)
        if len(x) == 2 and x.isdigit():
            h = int(x)
            return h if 0 <= h <= 23 else np.nan
        if len(x) == 4 and x.isdigit():
            h = int(x[:2])
            return h if 0 <= h <= 23 else np.nan
        return np.nan

    df["hour"] = df["fns_time_cd"].map(time_to_hour)
    df = df.dropna(subset=["hour"]).copy()
    df["hour"] = df["hour"].astype(int)

    # date/dow/weekend
    dt = pd.to_datetime(df["etl_ymd"], format="%Y%m%d", errors="raise")
    df["dow"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # aggregate to (date, dest, hour)
    agg = (
        df.groupby(["etl_ymd", "d_admdong_cd", "hour", "dow", "is_weekend"], as_index=False)["total_cnt"]
          .sum()
          .rename(columns={"total_cnt": "total_cnt_sum"})
    )

    # time-series features (NO leakage)
    agg = agg.sort_values(["d_admdong_cd", "hour", "etl_ymd"]).copy()
    g = agg.groupby(["d_admdong_cd", "hour"])["total_cnt_sum"]
    agg["lag_1"] = g.shift(1)
    agg["lag_7"] = g.shift(7)
    agg["rolling_mean_3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    agg["trend_1_7"] = agg["lag_1"] - agg["lag_7"]

    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour"] / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour"] / 24)

    before = len(agg)
    agg = agg.dropna().copy()
    print(f"[INFO] rows after feature+dropna: {before:,} -> {len(agg):,}")

    return agg

if USE_FEATURE_CACHE and os.path.exists(FEATURES_CACHE_PATH) and os.path.exists(FEATURES_CACHE_META):
    meta = joblib.load(FEATURES_CACHE_META)
    if meta.get("dataset_fingerprint") == data_fp and meta.get("feature_version") == FEATURE_VERSION:
        print("[INFO] Loading cached feature table:", FEATURES_CACHE_PATH)
        agg = pd.read_pickle(FEATURES_CACHE_PATH)
    else:
        print("[INFO] Feature cache version mismatch -> rebuild.")
        agg = build_features_from_raw()
        agg.to_pickle(FEATURES_CACHE_PATH)
        joblib.dump({"dataset_fingerprint": data_fp, "feature_version": FEATURE_VERSION}, FEATURES_CACHE_META)
        print("[INFO] Saved feature cache:", FEATURES_CACHE_PATH)
else:
    agg = build_features_from_raw()
    if USE_FEATURE_CACHE:
        agg.to_pickle(FEATURES_CACHE_PATH)
        joblib.dump({"dataset_fingerprint": data_fp, "feature_version": FEATURE_VERSION}, FEATURES_CACHE_META)
        print("[INFO] Saved feature cache:", FEATURES_CACHE_PATH)

# Optional labels
if admi_label is not None:
    agg = agg.merge(admi_label, on="d_admdong_cd", how="left")

# =========================================================
# 4) DATE SPLIT (80/20)
# =========================================================
days = sorted(agg["etl_ymd"].unique())
split_idx = int(len(days) * 0.8)
train_days = set(days[:split_idx])
test_days  = set(days[split_idx:])

train_df = agg[agg["etl_ymd"].isin(train_days)].copy()
test_df  = agg[agg["etl_ymd"].isin(test_days)].copy()

print(f"[INFO] days: {len(days)} | train days: {len(train_days)} | test days: {len(test_days)}")
print(f"[INFO] train rows: {len(train_df):,} | test rows: {len(test_df):,}")

# =========================================================
# 5) FEATURE MATRIX
# =========================================================
feature_cols = [
    "d_admdong_cd", "hour", "dow",
    "is_weekend",
    "lag_1", "lag_7",
    "rolling_mean_3",
    "trend_1_7",
    "hour_sin", "hour_cos"
]

X_train = pd.get_dummies(train_df[feature_cols], columns=["d_admdong_cd", "hour", "dow"])
X_test  = pd.get_dummies(test_df[feature_cols], columns=["d_admdong_cd", "hour", "dow"])
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

y_train = train_df["total_cnt_sum"].values
y_test  = test_df["total_cnt_sum"].values

print("[INFO] X_train:", X_train.shape, "X_test:", X_test.shape)

# =========================================================
# 6) LOAD OR TRAIN MODELS
# =========================================================
def artifacts_exist():
    return (os.path.exists(MODEL_LIN_PATH) and os.path.exists(MODEL_RF_PATH)
            and os.path.exists(COLS_PATH) and os.path.exists(META_PATH))

loaded = False
trained_now = False

if USE_SAVED_MODELS and (not FORCE_RETRAIN) and artifacts_exist():
    meta = joblib.load(META_PATH)
    same_data = (meta.get("dataset_fingerprint") == data_fp)
    same_feat = (meta.get("feature_version") == FEATURE_VERSION)

    if same_data and same_feat:
        print("[INFO] Loading saved models (data+feature match)...")
        lin = joblib.load(MODEL_LIN_PATH)
        rf  = joblib.load(MODEL_RF_PATH)
        saved_cols = joblib.load(COLS_PATH)
        X_train = X_train.reindex(columns=saved_cols, fill_value=0)
        X_test  = X_test.reindex(columns=saved_cols, fill_value=0)
        loaded = True
    else:
        print("[INFO] Saved artifacts mismatch -> retrain.")
        print("       same_data:", same_data, "| same_feat:", same_feat)

if not loaded:
    print("[INFO] Training new models...")
    trained_now = True

    lin = LinearRegression()

    # Heavy RF (your “previous” setup)
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=28,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    t0 = time.time()
    lin.fit(X_train, y_train)
    print(f"[INFO] Linear fit done in {time.time()-t0:.1f}s")

    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f"[INFO] RF fit done in {time.time()-t0:.1f}s")

    joblib.dump(lin, MODEL_LIN_PATH)
    joblib.dump(rf, MODEL_RF_PATH)
    joblib.dump(X_train.columns.tolist(), COLS_PATH)
    joblib.dump({"dataset_fingerprint": data_fp, "feature_version": FEATURE_VERSION}, META_PATH)
    print("[INFO] Saved models + feature columns + meta")

# =========================================================
# 7) PREDICTIONS + METRICS (TRAIN & TEST)
# =========================================================
lin_train_pred = lin.predict(X_train)
rf_train_pred  = rf.predict(X_train)

lin_test_pred = lin.predict(X_test)
rf_test_pred  = rf.predict(X_test)

m_lin_train = evaluate_return(y_train, lin_train_pred)
m_rf_train  = evaluate_return(y_train, rf_train_pred)
m_lin_test  = evaluate_return(y_test, lin_test_pred)
m_rf_test   = evaluate_return(y_test, rf_test_pred)

print_metrics("Linear Regression (TRAIN)", m_lin_train)
print_metrics("Random Forest (TRAIN)", m_rf_train)
print_metrics("Linear Regression (TEST)", m_lin_test)
print_metrics("Random Forest (TEST)", m_rf_test)

joblib.dump({
    "loaded_models": loaded,
    "trained_now": trained_now,
    "dataset_fingerprint": data_fp,
    "feature_version": FEATURE_VERSION,
    "linear_train": m_lin_train,
    "rf_train": m_rf_train,
    "linear_test": m_lin_test,
    "rf_test": m_rf_test,
}, METRICS_PATH)

# =========================
# 8) PLOTS FOR SLIDES (clean + non-redundant, ONE FIGURE EACH)
# =========================
import matplotlib
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
print("[INFO] Generating slide plots into:", PLOT_DIR)

def set_korean_font():
    """
    Windows: Malgun Gothic (맑은 고딕) 있으면 한글 깨짐 방지
    없으면 기본 폰트로 진행 (경고는 뜰 수 있음)
    """
    try:
        matplotlib.rcParams["font.family"] = "Malgun Gothic"
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def add_bar_labels(ax, bars, fmt="{:.2f}", pad=3):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h),
                    (b.get_x() + b.get_width()/2, h),
                    ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, pad),
                    fontsize=10)

set_korean_font()

# ---------------------------------------------------------
# Naive baseline (TEST): y_hat = lag_7
# ---------------------------------------------------------
y_naive = test_df["lag_7"].values
m_naive = evaluate_return(y_test, y_naive)
print_metrics("Naive Baseline (TEST) y_hat = lag_7", m_naive)

# ---------------------------------------------------------
# 3-model comparison bars (TEST)
# ---------------------------------------------------------
plt.figure(figsize=(7.2, 4.0))
ax = plt.gca()
vals = [m_naive["mae"], m_lin_test["mae"], m_rf_test["mae"]]
bars = ax.bar(["Naive", "Linear", "Random Forest"], vals)
ax.set_title("Test MAE")
ax.set_ylabel("MAE")
ax.grid(True, axis="y", alpha=0.2)
add_bar_labels(ax, bars, fmt="{:.2f}")
savefig(os.path.join(PLOT_DIR, "S0_test_mae_3models.png"))

plt.figure(figsize=(7.2, 4.0))
ax = plt.gca()
vals = [m_naive["mape_pct"], m_lin_test["mape_pct"], m_rf_test["mape_pct"]]
bars = ax.bar(["Naive", "Linear", "Random Forest"], vals)
ax.set_title("Test MAPE")
ax.set_ylabel("MAPE")
ax.grid(True, axis="y", alpha=0.2)
add_bar_labels(ax, bars, fmt="{:.2f}%")
savefig(os.path.join(PLOT_DIR, "S0_test_mape_3models.png"))

# ---------------------------------------------------------
# (S1) TEST MAE - bar (Linear vs RF)
# ---------------------------------------------------------
plt.figure(figsize=(6.5, 4))
ax = plt.gca()
vals = [m_lin_test["mae"], m_rf_test["mae"]]
bars = ax.bar(["Linear", "Random Forest"], vals)
ax.set_title("Test MAE")
ax.set_ylabel("MAE")
ax.grid(True, axis="y", alpha=0.2)
add_bar_labels(ax, bars, fmt="{:.2f}")
savefig(os.path.join(PLOT_DIR, "S1_test_mae.png"))

# ---------------------------------------------------------
# (S2) TEST MAPE - bar (Linear vs RF)
# ---------------------------------------------------------
plt.figure(figsize=(6.5, 4))
ax = plt.gca()
vals = [m_lin_test["mape_pct"], m_rf_test["mape_pct"]]
bars = ax.bar(["Linear", "Random Forest"], vals)
ax.set_title("Test MAPE")
ax.set_ylabel("MAPE")
ax.grid(True, axis="y", alpha=0.2)
add_bar_labels(ax, bars, fmt="{:.2f}%")
savefig(os.path.join(PLOT_DIR, "S2_test_mape.png"))

# ---------------------------------------------------------
# (S3) TEST R^2 - bar (Linear vs RF)  (kept for compatibility; you can omit in slides)
# ---------------------------------------------------------
plt.figure(figsize=(6.5, 4))
ax = plt.gca()
vals = [m_lin_test["r2"], m_rf_test["r2"]]
bars = ax.bar(["Linear", "Random Forest"], vals)
ax.set_title("Test R2")
ax.set_ylabel("R2")
ax.set_ylim(0, 1.02)
ax.grid(True, axis="y", alpha=0.2)
add_bar_labels(ax, bars, fmt="{:.4f}")
savefig(os.path.join(PLOT_DIR, "S3_test_r2.png"))

# ---------------------------------------------------------
# (S4) Predicted vs Actual (RF) - Test
# ---------------------------------------------------------
plt.figure(figsize=(6.5, 6.5))
ax = plt.gca()
actual = y_test
pred = rf_test_pred

ax.scatter(actual, pred, s=10, alpha=0.12)
mx = float(np.nanmax([actual.max(), pred.max()]))
ax.plot([0, mx], [0, mx], linewidth=2)
ax.set_title("Predicted vs Actual Random Forest")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.grid(True, alpha=0.15)
savefig(os.path.join(PLOT_DIR, "S4_pred_vs_actual_rf.png"))

# ---------------------------------------------------------
# (S4b) Predicted vs Actual (Linear) - Test
# ---------------------------------------------------------
plt.figure(figsize=(6.5, 6.5))
ax = plt.gca()
actual = y_test
pred = lin_test_pred

ax.scatter(actual, pred, s=10, alpha=0.12)
mx = float(np.nanmax([actual.max(), pred.max()]))
ax.plot([0, mx], [0, mx], linewidth=2)
ax.set_title("Predicted vs Actual Linear Regression")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.grid(True, alpha=0.15)
savefig(os.path.join(PLOT_DIR, "S4b_pred_vs_actual_lin.png"))

# ---------------------------------------------------------
# (S5) Residual histogram (RF)
# ---------------------------------------------------------
res = (y_test - rf_test_pred).astype(float)
lo, hi = np.quantile(res, [0.01, 0.99])
res_zoom = res[(res >= lo) & (res <= hi)]

plt.figure(figsize=(7.5, 4.2))
ax = plt.gca()
ax.hist(res_zoom, bins=60, alpha=0.9)
ax.axvline(0, linestyle="--", linewidth=2)
ax.set_title("Residuals Random Forest")
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.grid(True, axis="y", alpha=0.2)
savefig(os.path.join(PLOT_DIR, "S5_residual_hist_rf.png"))

# ---------------------------------------------------------
# (S5b) Residual histogram (Linear)
# ---------------------------------------------------------
res_lin = (y_test - lin_test_pred).astype(float)
lo, hi = np.quantile(res_lin, [0.01, 0.99])
res_lin_zoom = res_lin[(res_lin >= lo) & (res_lin <= hi)]

plt.figure(figsize=(7.5, 4.2))
ax = plt.gca()
ax.hist(res_lin_zoom, bins=60, alpha=0.9)
ax.axvline(0, linestyle="--", linewidth=2)
ax.set_title("Residuals Linear Regression")
ax.set_xlabel("Residual")
ax.set_ylabel("Count")
ax.grid(True, axis="y", alpha=0.2)
savefig(os.path.join(PLOT_DIR, "S5b_residual_hist_lin.png"))

# =========================
# MAE by hour (Test) - Linear + RF
# =========================
y_true = test_df["total_cnt_sum"].values
hours  = test_df["hour"].values

abs_err_lin = np.abs(y_true - lin_test_pred)
abs_err_rf  = np.abs(y_true - rf_test_pred)

mae_lin_by_hour = (
    pd.DataFrame({"hour": hours, "abs_err": abs_err_lin})
      .groupby("hour")["abs_err"].mean()
      .reindex(range(24))
)

mae_rf_by_hour = (
    pd.DataFrame({"hour": hours, "abs_err": abs_err_rf})
      .groupby("hour")["abs_err"].mean()
      .reindex(range(24))
)

plt.figure(figsize=(12, 5))
plt.plot(mae_lin_by_hour.index, mae_lin_by_hour.values, marker="o", linewidth=2, label="Linear")
plt.plot(mae_rf_by_hour.index,  mae_rf_by_hour.values,  marker="o", linewidth=2, label="Random Forest")
plt.title("MAE by Hour")
plt.xlabel("Hour")
plt.ylabel("MAE")
plt.xticks(range(0, 24, 1))
plt.grid(True, alpha=0.25)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "S4_mae_by_hour_models.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================
# MAPE by hour (Test) - Linear + RF
# =========================
ape_lin = np.abs((y_true - lin_test_pred) / (y_true + 1e-6)) * 100.0
ape_rf  = np.abs((y_true - rf_test_pred)  / (y_true + 1e-6)) * 100.0

mape_lin_by_hour = (
    pd.DataFrame({"hour": hours, "ape": ape_lin})
      .groupby("hour")["ape"].mean()
      .reindex(range(24))
)

mape_rf_by_hour = (
    pd.DataFrame({"hour": hours, "ape": ape_rf})
      .groupby("hour")["ape"].mean()
      .reindex(range(24))
)

plt.figure(figsize=(12, 5))
plt.plot(mape_lin_by_hour.index, mape_lin_by_hour.values, marker="o", linewidth=2, label="Linear")
plt.plot(mape_rf_by_hour.index,  mape_rf_by_hour.values,  marker="o", linewidth=2, label="Random Forest")
plt.title("MAPE by Hour")
plt.xlabel("Hour")
plt.ylabel("MAPE")
plt.xticks(range(0, 24, 1))
plt.grid(True, alpha=0.25)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "S4b_mape_by_hour_models.png"), dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# Citywide heatmap: avg inflow by (dow x hour)
# ---------------------------------------------------------
city = agg.copy()
heat = city.groupby(["dow", "hour"])["total_cnt_sum"].mean().unstack("hour").sort_index()

plt.figure(figsize=(10.5, 4.5))
ax = plt.gca()
im = ax.imshow(heat.values, aspect="auto")
ax.set_title("Citywide Inflow Heatmap")
ax.set_xlabel("Hour")
ax.set_ylabel("Day of Week")
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([str(h) for h in range(0, 24, 2)])
ax.set_yticks(range(7))
ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
plt.colorbar(im, ax=ax, label="Avg total_cnt_sum")
savefig(os.path.join(PLOT_DIR, "S7_city_heatmap_dow_hour.png"))

# ---------------------------------------------------------
# RF importance: lag_7 share vs all others
# ---------------------------------------------------------
if hasattr(rf, "feature_importances_"):
    imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    total_imp = float(imp.sum())
    lag7 = float(imp.get("lag_7", 0.0))
    rest = max(total_imp - lag7, 0.0)

    lag7_pct = 100.0 * lag7 / max(total_imp, 1e-12)
    rest_pct = 100.0 - lag7_pct

    plt.figure(figsize=(7.5, 4.2))
    ax = plt.gca()
    bars = ax.bar(["lag_7", "all other features"], [lag7_pct, rest_pct])
    ax.set_title("RF Lag7 Share")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.2)
    add_bar_labels(ax, bars, fmt="{:.1f}%")
    savefig(os.path.join(PLOT_DIR, "S8_rf_importance_share.png"))

    # ---------------------------------------------------------
    # Top RF features including lag_7 (relative importance % of total)
    # ---------------------------------------------------------
    top_rf = imp.head(12)
    top_rf_pct = (top_rf / max(total_imp, 1e-12)) * 100.0

    plt.figure(figsize=(8.8, 5.2))
    ax = plt.gca()
    ax.barh(top_rf_pct.index[::-1], top_rf_pct.values[::-1])
    ax.set_title("Top RF Features")
    ax.set_xlabel("Percent")
    ax.grid(True, axis="x", alpha=0.2)
    savefig(os.path.join(PLOT_DIR, "S9_rf_top_features_relative.png"))

# ---------------------------------------------------------
# Linear Regression coefficient magnitude share: lag_7 vs all others
# ---------------------------------------------------------
coef_abs = pd.Series(np.abs(lin.coef_), index=X_train.columns)
total_coef = float(coef_abs.sum())
lag7_c = float(coef_abs.get("lag_7", 0.0))
rest_c = max(total_coef - lag7_c, 0.0)

lag7_pct = 100.0 * lag7_c / max(total_coef, 1e-12)
rest_pct = 100.0 - lag7_pct

plt.figure(figsize=(7.5, 4.2))
ax = plt.gca()
bars = ax.bar(["lag_7", "all other features"], [lag7_pct, rest_pct])
ax.set_title("Linear Lag7 Share")
ax.set_ylabel("Percent")
ax.set_ylim(0, 100)
ax.grid(True, axis="y", alpha=0.2)
add_bar_labels(ax, bars, fmt="{:.1f}%")
savefig(os.path.join(PLOT_DIR, "S8b_lin_coef_share.png"))

# ---------------------------------------------------------
# Top Linear Regression features (relative importance % of total |coef|)
# ---------------------------------------------------------
top_lin = coef_abs.sort_values(ascending=False).head(12)
top_lin_pct = (top_lin / max(total_coef, 1e-12)) * 100.0

plt.figure(figsize=(8.8, 5.2))
ax = plt.gca()
ax.barh(top_lin_pct.index[::-1], top_lin_pct.values[::-1])
ax.set_title("Top Linear Features")
ax.set_xlabel("Percent")
ax.grid(True, axis="x", alpha=0.2)
savefig(os.path.join(PLOT_DIR, "S9b_lin_top_features_relative.png"))

print("[INFO] Slide plot files saved:")
print(" - S0_test_mae_3models.png")
print(" - S0_test_mape_3models.png")
print(" - S1_test_mae.png")
print(" - S2_test_mape.png")
print(" - S3_test_r2.png")
print(" - S4_pred_vs_actual_rf.png")
print(" - S4b_pred_vs_actual_lin.png")
print(" - S5_residual_hist_rf.png")
print(" - S5b_residual_hist_lin.png")
print(" - S4_mae_by_hour_models.png")
print(" - S4b_mape_by_hour_models.png")
print(" - S7_city_heatmap_dow_hour.png")
print(" - S8_rf_importance_share.png")
print(" - S9_rf_top_features_relative.png")
print(" - S8b_lin_coef_share.png")
print(" - S9b_lin_top_features_relative.png")