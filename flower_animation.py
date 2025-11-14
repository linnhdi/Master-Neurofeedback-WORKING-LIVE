import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Ellipse, Rectangle
from PyQt5.QtWidgets import QApplication, QLabel


# --- Setup figure ---
fig, ax = plt.subplots(figsize=(6, 8), facecolor='#E0F7FA')
plt.subplots_adjust(bottom=0.2)  # leave space for slider
ax.set_xlim(-3, 3)
ax.set_ylim(-1, 8)
ax.set_aspect("equal")
ax.axis("off")



# --- Soil patch (elliptical mound) ---
soil = Ellipse(
    (0, -0.5),          # center (x=0, just below y=0)
    width=8,            # wide mound
    height=2,           # tall enough to cover bottom
    facecolor="#A0522D",
    edgecolor="#A0522D",
    zorder=0
)
ax.add_patch(soil)


# --- Flower parts ---
stem_line, = ax.plot([], [], color="#4CAF50", lw=6)
petals = []
leaf = None
leaf_2 = None
center = plt.Circle((0, 6), 0, color="#FFF176", zorder=5)
ax.add_patch(center)

def draw_flower(g):
    global leaf
    global leaf_2

    # Remove old leaf
    if leaf is not None:
        leaf.remove()
        leaf = None
    if leaf_2 is not None:
        leaf_2.remove()
        leaf_2 = None   

    # --- Stem (appears immediately) ---
    stem_height =  6 * g
    stem_line.set_data([0, 0], [0, stem_height])
    
    # --- Leaf (appears after 30%) ---
    if g > 0.3:
        leaf_size = g  # scales with growth
        leaf = Ellipse(
            (leaf_size/2, stem_height/2),  # attach to stem
            width=leaf_size*1.2,
            height=leaf_size*0.6,
            angle=30,
            facecolor="#4CAF50",
            edgecolor="#4CAF50",
            alpha=0.8,
            zorder=1
        )
        ax.add_patch(leaf)
    
    # --- Leaf 2 (appears after 30%) ---
    if g > 0.3:
        leaf_size = g  # scales with growth
        leaf_2 = Ellipse(
            (-leaf_size/2, stem_height/2-0.5),  # attach to stem
            width=leaf_size*1.2,
            height=leaf_size*0.6,
            angle=-30,
            facecolor="#4CAF50",
            edgecolor="#4CAF50",
            alpha=0.8,
            zorder=1
        )
        ax.add_patch(leaf_2)
    
    # --- Remove old petals ---
    for petal in petals:
        petal.remove()
    petals.clear()
    
# --- Petals (appear after 50%) ---
    if g > 0.5:
        n_petals = 30
        petal_length = 1.5 * (g - 0.5) * 2 + 0.5  # grows outward
        petal_width = 0.3  # thin petals

        # Arrange petals in a circle around center
        angles = np.linspace(0, 360, n_petals, endpoint=False)
        for angle in angles:
            # Petal center: offset from flower top
            x_center = petal_length/2 * np.cos(np.deg2rad(angle))
            y_center = stem_height + petal_length/2 * np.sin(np.deg2rad(angle))
            petal = Ellipse(
                (x_center, y_center),
                width=petal_width,
                height=petal_length,
                angle=angle+90,
                facecolor="#9B59B6",
                edgecolor="#7D3C98",
                alpha=0.5,
                zorder=2
            )
            ax.add_patch(petal)
            petals.append(petal)
    
    # --- Flower center ---
    if g > 0.5:
        center.radius = g * 0.3
        center.center = (0, stem_height)
    else:
        center.radius = 0  # hidden when not blooming
    
    fig.canvas.draw_idle()

# --- Slider setup ---
slider_ax = plt.axes([0.2, 0.05, 0.6, 0.05])  # [left, bottom, width, height]
growth_slider = Slider(slider_ax, "Concentration", 0, 100, valinit=0)

def update(val):
    g = growth_slider.val / 100.0
    draw_flower(g)

growth_slider.on_changed(update)

# Initial draw
draw_flower(0)
plt.show()
