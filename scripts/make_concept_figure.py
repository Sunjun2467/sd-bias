"""Usage (from C:\sd-bias): python scripts/make_concept_figure.py  -> figures/concept_gender_vector_3d.png

Concept figure: subtracting a gendered word pair leaves only the gender direction (toy 3D vectors)."""
import sys
import numpy as np
import matplotlib.pyplot as plt


out = sys.argv[1] if len(sys.argv) > 1 else "figures/concept_gender_vector_3d.png"

# toy 3D vectors: axis 0 = gender (+ male), axis 1 = royalty, axis 2 = family role
pairs = {
    "man / woman":       (np.array([2.0, 0.6, 0.4]), np.array([0.1, 0.6, 0.4])),
    "king / queen":      (np.array([2.3, 3.0, 1.0]), np.array([0.4, 3.1, 1.0])),
    "father / mother":   (np.array([1.9, 0.7, 2.8]), np.array([0.0, 0.6, 2.9])),
}
names = {"man / woman": ("man", "woman"), "king / queen": ("king", "queen"),
         "father / mother": ("father", "mother")}
colors = ["#2E6FBA", "#E8A33D", "#3A9A5B"]
MALE, FEMALE, DIR = "#2E6FBA", "#D1495B", "#1C1F2A"

fig = plt.figure(figsize=(13, 5.6))

# ---- left: words as points, arrows female -> male ----
ax = fig.add_subplot(1, 2, 1, projection="3d")
for (key, (m, f)), c in zip(pairs.items(), colors):
    mn, fn = names[key]
    ax.scatter(*m, s=60, color=MALE, depthshade=False)
    ax.scatter(*f, s=60, color=FEMALE, depthshade=False)
    ax.quiver(*f, *(m - f), color=c, linewidth=2.2, arrow_length_ratio=0.12)
    ax.text(*(m + [0.1, 0, 0.18]), mn, fontsize=11, color=MALE)
    ax.text(*(f + [-0.1, 0, 0.18]), fn, fontsize=11, color=FEMALE, ha="right")
ax.set_xlabel("axis 1 (gender)", labelpad=6)
ax.set_ylabel("axis 2 (royalty)", labelpad=6)
ax.set_zlabel("axis 3 (family)", labelpad=6)
ax.set_xlim(-0.5, 2.8); ax.set_ylim(0, 3.5); ax.set_zlim(0, 3.3)
ax.view_init(elev=22, azim=-60)
ax.set_title("(1) Words are points: each pair gives a female → male arrow", fontsize=13, pad=10)

# ---- right: difference vectors from the origin ----
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
diffs = []
for (key, (m, f)), c in zip(pairs.items(), colors):
    d = np.round(m - f, 1) + 0.0
    diffs.append(d)
    ax2.quiver(0, 0, 0, *d, color=c, linewidth=2.2, arrow_length_ratio=0.1, alpha=0.9)
    mn, fn = names[key]
    ax2.text2D(0.02, 0.97 - 0.07 * len(diffs), f"{mn} - {fn} = ({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f})",
               transform=ax2.transAxes, fontsize=11, color=c, weight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
mean = np.round(np.mean(diffs, axis=0), 1) + 0.0   # +0.0 turns -0.0 into 0.0
ax2.quiver(0, 0, 0, *mean, color=DIR, linewidth=4, arrow_length_ratio=0.1)
ax2.text2D(0.02, 0.97 - 0.07 * 4.3, f"mean = ({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}) → gender direction",
           transform=ax2.transAxes, fontsize=12, color=DIR, weight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax2.set_xlabel("axis 1 (gender)", labelpad=6)
ax2.set_ylabel("axis 2 (royalty)", labelpad=6)
ax2.set_zlabel("axis 3 (family)", labelpad=6)
ax2.set_xlim(0, 2.4); ax2.set_ylim(-1, 1); ax2.set_zlim(-1, 1)
ax2.view_init(elev=22, azim=-60)
ax2.set_title("(2) Subtracting cancels royalty and family,\nleaving only the gender direction", fontsize=13, pad=10)

fig.text(0.5, 0.015, "Conceptual illustration: toy 3D vectors, not real coordinates (actual vectors are 768-dimensional)",
         ha="center", fontsize=9, color="#6B707C")
plt.tight_layout(rect=(0, 0.04, 1, 1))
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
