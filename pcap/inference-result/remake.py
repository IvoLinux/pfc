# Improved Confusion Matrix Plot (Matplotlib-only, white→blue gradient)
# You can edit CAPTION_TEXT below to change the figure title/caption.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ==== EDITABLE SETTINGS ====
CAPTION_TEXT = "Matriz de confusão para janelas de PCAP"  # <-- change this text to edit the caption/title
LABELS = ["Benigno", "Força Bruta"]              # axis labels
FONT_BASE = 14                               # base font size for ticks and labels
TITLE_SIZE = 18                              # title font size
ANNOT_SIZE = 14                              # numbers inside the squares
SAVE_PATH = "./confusion_matrix_white_blue.png"  # output path
# ===========================

# Provided confusion matrix (counts)
cm_counts = np.array([[41584, 25130],
                      [12376, 25767]], dtype=float)

# Normalize by row for optional display (not used in the heatmap, but left here for reference/extension)
row_sums = cm_counts.sum(axis=1, keepdims=True)
cm_row_norm = cm_counts / np.maximum(row_sums, 1)

# Build a simple white→blue colormap
white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", ["#FFFFFF", "#1f77b4"])

# Plot
fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
im = ax.imshow(cm_counts, cmap=white_to_blue)

# Add annotations (counts)
for i in range(cm_counts.shape[0]):
    for j in range(cm_counts.shape[1]):
        ax.text(j, i, f"{int(cm_counts[i, j])}", ha="center", va="center", fontsize=ANNOT_SIZE, color="black")

# Axis ticks and labels
ax.set_xticks(np.arange(len(LABELS)), labels=LABELS, fontsize=FONT_BASE)
ax.set_yticks(np.arange(len(LABELS)), labels=LABELS, fontsize=FONT_BASE)
ax.set_xlabel("Valor Predito", fontsize=FONT_BASE)
ax.set_ylabel("Valor Real", fontsize=FONT_BASE)

# Title/caption
ax.set_title(CAPTION_TEXT, fontsize=TITLE_SIZE, pad=12)

# Colorbar with larger tick labels
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=FONT_BASE)

# Tight layout and save
plt.tight_layout()
plt.savefig(SAVE_PATH, bbox_inches="tight")
SAVE_PATH
