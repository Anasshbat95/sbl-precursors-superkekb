"""
pipeline_config.py - All tunable parameters of the 5-step SBL pipeline in one place.

Override any of them by passing a dict to run_pipeline(...), e.g.
    run_pipeline(blocks, sbl_times, {"window": "30min", "methods": ("iforest",)})
"""

DEFAULTS = dict(
    # ---- step 1: preprocessing -------------------------------------------------------
    ts_col="timestamp",
    block_tz=None,                 # e.g. "Europe/Berlin" if block timestamps are local and tz-naive
    sbl_tz=None,                   # same for the SBL timestamps
    use_operation_filter=False,     # keep only BEAM_ON rows (needs the beam block + current columns)
    beam_block="beam",
    current_cols=("her_current", "ler_current"),
    on_thr=50.0,                   # beam current above this = beam present
    ffill_limit=3,                 # forward-fill at most this many rows of NaN, then drop the row
    auto_log=True,                 # log10-transform columns that span several orders of magnitude
    use_injection_veto=False,      # suppress alarms right after injections (check the reported
    injection_block="injection",   #   injection-active fraction first: if injection is nearly
    injection_rate_cols=None,      #   always on, a veto would remove almost everything)
    inj_veto="2min",

    # ---- step 2: feature engineering -------------------------------------------------
    window="10min",                # backward-looking rolling window (auto-widened for sparse blocks)
    min_rows_per_window=5,         # window is at least this many median row spacings wide
    min_periods=3,                 # rolling statistics need at least this many rows in the window
    feature_gap="2h",              # larger gaps start a new feature segment

    # ---- steps 3 + 4: model + training strategy --------------------------------------
    methods=("zscore", "iforest"), # "zscore" = simple baseline, "iforest" = Isolation Forest
    normal_margin="15min",         # "normal" = farther than this from EVERY SBL
    train_frac=0.6,                # chronological split: train | calibration | test
    calib_frac=0.2,                   # calibration period is this fraction of the train span (e.g. 0.2 = 20% of train)
    n_estimators=200,
    max_train_rows=200000,
    min_train_rows=100,
    random_state=0,

    # ---- step 5: validation ----------------------------------------------------------
    grid="auto",                   # common time grid for fusing block scores ("auto" = from the data)
    lookback="10min",              # an alarm counts as detection if it falls in [SBL-lookback, SBL-buffer]
    buffer="30s",
    cooldown="10min",              # alarms closer together than this count as one false-alarm episode
    max_gap="30min",               # cap on the time one score point "covers" when computing hours
    thr_quantiles=(0.99, 0.995, 0.999, 0.9999),   # thresholds = quantiles of calibration-normal scores
    n_boot=500,                    # bootstrap resamples for the lead-time confidence interval
)
