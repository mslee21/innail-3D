
import numpy as np
import cv2
from scipy.ndimage import shift
import matplotlib.pyplot as plt

# Simulate synthetic light field
num_views = 7
sensor_res = 64
light_field = np.zeros((num_views, num_views, sensor_res, sensor_res))

# Gaussian dot in center for all views (simulate real lens variation)
for u in range(num_views):
    for v in range(num_views):
        x0 = sensor_res // 2 + (u - num_views // 2) * 2
        y0 = sensor_res // 2 + (v - num_views // 2) * 2
        xx, yy = np.meshgrid(np.arange(sensor_res), np.arange(sensor_res))
        light_field[u, v] = np.exp(-((xx - x0)**2 + (yy - y0)**2) / 50)

# Refocus function
def refocus_lf(lf, alpha):
    u_max, v_max, _, _ = lf.shape
    composite = np.zeros_like(lf[0, 0])
    for u in range(u_max):
        for v in range(v_max):
            dx = alpha * (u - u_max // 2)
            dy = alpha * (v - v_max // 2)
            view = lf[u, v]
            shifted = shift(view, shift=(-dy, -dx), order=1, mode='nearest')
            composite += shifted
    return composite / (u_max * v_max)

# Depth estimation from disparity (simple block match)
def estimate_disparity(ref, tgt, max_disp=5):
    h, w = ref.shape
    disparity = np.zeros((h, w))
    win = 5
    for y in range(win, h - win):
        for x in range(win + max_disp, w - win):
            best_offset = 0
            best_score = float('inf')
            ref_patch = ref[y - win:y + win + 1, x - win:x + win + 1]
            for d in range(-max_disp, max_disp + 1):
                tgt_patch = tgt[y - win:y + win + 1, x - win + d:x + win + 1 + d]
                score = np.sum((ref_patch - tgt_patch) ** 2)
                if score < best_score:
                    best_score = score
                    best_offset = d
            disparity[y, x] = best_offset
    return disparity

# Refocus and depth map
refocused = refocus_lf(light_field, 0)
side_view = light_field[0, 0]
disparity = estimate_disparity(refocused, side_view)
depth_map = 1.0 / (np.abs(disparity) + 1e-3)

# Plot result
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
axs[0].imshow(refocused, cmap='hot'); axs[0].set_title("Refocused Image")
axs[1].imshow(disparity, cmap='coolwarm'); axs[1].set_title("Disparity Map")
axs[2].imshow(depth_map, cmap='plasma'); axs[2].set_title("Estimated Depth Map")
for ax in axs: ax.axis('off')
plt.tight_layout(); plt.show()
