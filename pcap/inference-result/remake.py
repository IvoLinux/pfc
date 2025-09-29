import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ==== EDITABLE SETTINGS ====
CAPTION_TEXT = "Random Forest"
LABELS = ["Benigno", "Malicioso"]
FONT_BASE = 14
TITLE_SIZE = 16
ANNOT_SIZE = 14
SAVE_PATH = "./cm_am_1.png"
# ===========================

# Provided confusion matrix (counts)
cm_counts = np.array([[454142, 604],
                      [131, 108905]], dtype=float)

total = cm_counts.sum()

# Build a simple white→blue colormap
white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", ["#FFFFFF", "#1f77b4"])

fig, ax = plt.subplots(figsize=(6.24, 6.24), dpi=160)
im = ax.imshow(cm_counts, cmap=white_to_blue)

# Add annotations (counts + %)
for i in range(cm_counts.shape[0]):
    for j in range(cm_counts.shape[1]):
        count = int(cm_counts[i, j])
        formatted_count = f"{count:,}".replace(",", ".")
        perc = count / total * 100
        ax.text(j, i, f"{formatted_count}\n({perc:.2f}%)", ha="center", va="center", fontsize=ANNOT_SIZE, color="black")

# Axis ticks and labels
ax.set_xticks(np.arange(len(LABELS)), labels=LABELS, fontsize=FONT_BASE)
ax.set_yticks(np.arange(len(LABELS)), labels=LABELS, fontsize=FONT_BASE, rotation=90, va="center")
ax.set_xlabel("Valor Predito", fontsize=FONT_BASE)
ax.set_ylabel("Valor Real", fontsize=FONT_BASE)

# Title/caption
ax.set_title(CAPTION_TEXT, fontsize=TITLE_SIZE, pad=6)

# Colorbar with larger tick labels
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=FONT_BASE)

# Tight layout and save
plt.tight_layout()
plt.savefig(SAVE_PATH, bbox_inches="tight")
SAVE_PATH
