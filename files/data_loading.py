"""
data_loading.py - Read the raw CSV files and rename their columns to
consistent, readable names (e.g. "BMHDCCT:CURRENT" -> "her_current").

Usage:
    from data_loading import load_all_blocks, load_sbl_timestamps

    blocks = load_all_blocks("Dataset")
    beam, background, injection, vacuum, position = (
        blocks["beam"], blocks["background"], blocks["injection"],
        blocks["vacuum"], blocks["position"],
    )
    sbl_times = load_sbl_timestamps("Dataset/SBL_timestamps_Germany.csv")
"""
import pandas as pd

# raw EPICS/detector variable name -> readable column name, per block
RENAME_MAPS = {
    "background": {
        "Timestamp": "timestamp",
        "CGHOPT:GATED_TUNE_X:MEAS": "her_tune_x",
        "CGHOPT:GATED_TUNE_Y:MEAS": "her_tune_y",
        "CGLOPT:GATED_TUNE_X:MEAS": "ler_tune_x",
        "CGLOPT:GATED_TUNE_Y:MEAS": "ler_tune_y",
        "MGLPS:QKSD5TLP1:KMON": "dispersion_correction",
        "B2_nsm:get:MONZDLMINT:ZDLM_INTVAL:value": "mag_current_z_corr",
        "B2_nsm:get:ECL_LUM_MON:lum_acc_20": "det_luminosity",
        "B2_nsm:get:ECLTRG_FAM:rate_fw": "det_trigger_rate_fw",
        "B2_nsm:get:TRGOSCILLO1:lff_her": "her_oscillator_freq",
        "B2_nsm:get:ECLTRG_FAM:rate_bw": "det_trigger_rate_bw",
        "B2_nsm:get:TRGOSCILLO0:lff_ler": "ler_oscillator_freq",
    },
    "beam": {
        "Timestamp": "timestamp",
        "BMHDCCT:CURRENT": "her_current",
        "BMHXRM:BEAM:SIGMAXatIP": "her_sigma_x",
        "BMHXRM:BEAM:SIGMAYatIP": "her_sigma_y",
        "BMLDCCT:CURRENT": "ler_current",
        "BMLXRM:BEAM:SIGMAXatIP": "ler_sigma_x",
        "BMLXRM:BEAM:SIGMAYatIP": "ler_sigma_y",
    },
    "injection": {
        "Timestamp": "timestamp",
        "CGeBT:DPP0": "her_disp_matching",
        "CGpBT:DPP0": "ler_disp_matching",
        "BTePS:VM27E:KRB": "her_rf_volt_vm27e",
        "BTePS:VM28E:KRB": "her_rf_volt_vm28e",
        "BTpPS:VM32P:KRB": "ler_rf_volt_vm32p",
        "BTpPS:VM33P:KRB": "ler_rf_volt_vm33p",
        "CGHINJ:RATE_AVE": "her_inj_rate",
        "CGLINJ:EFFICIENCY": "ler_inj_efficiency",
        "CGHINJ:EFFICIENCY": "her_inj_efficiency",
    },
    "vacuum": {
        "Timestamp": "timestamp",
        "B2_TOP:L1:SCALER_RATE_TYPE_2_AVERAGE_PER_PMT": "det_event_rate_scaler",
        "B2_VXD:Rad:Res:BPAtDCU3:A1:DoseRate": "det_vxd_dose_rate",
        "VALCCG:LER:PRES:AVG": "ler_vac_pres_global",
        "VALCCG:D02_L25:PRES": "ler_vac_pres_local_d02",
        "VAHCCG:HER:PRES:AVG": "her_vac_pres_global",
        "VAHCCG:D01_HER:PRES:AVG": "her_vac_pres_local_d01",
    },
    "position": {
        "Timestamp": "timestamp",
        "BMH:MQLC2LE:POS.PXP": "her_mag_mqlc2le_pos_x",
        "BMH:MQLC2LE:POS.PYP": "her_mag_mqlc2le_pos_y",
        "BMH:MQLC2RE:POS.PXP": "her_mag_mqlc2re_pos_x",
        "BMH:MQLC3LE:POS.PXP": "her_mag_mqlc3le_pos_x",
        "BMH:MQLC3RE:POS.PXP": "her_mag_mqlc3re_pos_x",
        "BMH:MQLC6RE:POS.PYP": "her_mag_mqlc6re_pos_y",
        "BMH:MQLC7LE:POS.PYP": "her_mag_mqlc7le_pos_y",
        "BMH:MQLC7RE:POS.PYP": "her_mag_mqlc7re_pos_y",
        "BML:MQC2LP:POS.PXP": "ler_mag_mqc2lp_pos_x",
        "BML:MQC2LP:POS.PYP": "ler_mag_mqc2lp_pos_y",
        "BML:MQC2RP:POS.PXP": "ler_mag_mqc2rp_pos_x",
        "BML:MQC2RP:POS.PYP": "ler_mag_mqc2rp_pos_y",
    },
}

# default raw file name per block (adjust to match your Dataset folder)
DEFAULT_FILES = {
    "beam": "02_ms_beam.csv",
    "background": "01_ms_background.csv",
    "injection": "04_ms_injection.csv",
    "vacuum": "05_ms_vacuum.csv",
    "position": "03_ms_position.csv",
}


def to_timestamp(df: pd.DataFrame, column_name: str = "timestamp") -> pd.DataFrame:
    """Convert a timestamp column to a real pandas datetime dtype (in place, returns df)."""
    df[column_name] = pd.to_datetime(df[column_name], format="ISO8601")
    return df


def rename_and_clean(df: pd.DataFrame, block_name: str) -> pd.DataFrame:
    """Rename a raw block's columns using RENAME_MAPS[block_name] and parse its timestamp."""
    if block_name not in RENAME_MAPS:
        raise KeyError(f"Unknown block '{block_name}'. Known blocks: {list(RENAME_MAPS)}")
    df = df.rename(columns=RENAME_MAPS[block_name])
    return to_timestamp(df, "timestamp")


def load_all_blocks(dataset_dir: str, files: dict | None = None) -> dict[str, pd.DataFrame]:
    """Load and rename all 5 blocks. `files` overrides DEFAULT_FILES per block name."""
    files = {**DEFAULT_FILES, **(files or {})}
    blocks = {}
    for name, filename in files.items():
        df = pd.read_csv(f"{dataset_dir}/{filename}")
        blocks[name] = rename_and_clean(df, name)
    return blocks


def load_sbl_timestamps(path: str) -> pd.DataFrame:
    """Load the SBL event timestamps (single unnamed column) as a DataFrame with column 'timestamp'."""
    df = pd.read_csv(path, header=None, names=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")
    return df
