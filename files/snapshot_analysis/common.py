import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
ROOT = "/Users/anasshbat/Desktop/Uni-mac/Masterthesis_26"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/files"); sys.path.insert(0, ROOT + "/files_pip")
os.chdir(ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
FIG = ROOT + "/MasterArbeit/figures/"
OUT = os.path.dirname(__file__) + "/out/"; os.makedirs(OUT, exist_ok=True)
_names = []
def save_next(*names):
    """the next plt.show() calls save the open figures under these names"""
    _names.extend(names)
def _show(*a, **k):
    for n in plt.get_fignums():
        fig = plt.figure(n)
        name = _names.pop(0) if _names else None
        if name:
            fig.savefig(FIG + name, dpi=200, bbox_inches="tight"); print("saved", name)
        plt.close(fig)
plt.show = _show

from data_loading import load_all_blocks, load_sbl_timestamps
def load():
    p = OUT + "data.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    blocks = load_all_blocks("Dataset")
    sbl = load_sbl_timestamps("Dataset/SBL_timestamps_Germany.csv")
    pickle.dump((blocks, sbl), open(p, "wb"))
    return blocks, sbl
