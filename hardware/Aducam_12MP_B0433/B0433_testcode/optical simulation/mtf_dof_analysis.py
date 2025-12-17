
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft

# Simulate slanted edge image
def generate_edge_image(size=256, angle_deg=5):
    angle = np.radians(angle_deg)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    rotated = (xx * np.cos(angle) + yy * np.sin(angle)) - size // 2
    edge = np.where(rotated > 0, 1.0, 0.0)
    return edge

# Extract Edge Spread Function (ESF)
def extract_esf(edge_img):
    center_line = edge_img[edge_img.shape[0]//2-5 : edge_img.shape[0]//2+5, :]
    esf = np.mean(center_line, axis=0)
    return esf

# Compute Line Spread Function (LSF)
def compute_lsf(esf):
    lsf = np.gradient(esf)
    return lsf / np.max(lsf)

# Compute MTF (Fourier of LSF)
def compute_mtf(lsf):
    mtf = np.abs(fft(lsf))
    mtf = mtf[:len(mtf)//2]
    mtf /= np.max(mtf)
    return mtf

# Main
edge_img = generate_edge_image()
esf = extract_esf(edge_img)
lsf = compute_lsf(esf)
mtf = compute_mtf(lsf)

# Depth simulation: blur and compare MTF
blurred = gaussian_filter(edge_img, sigma=3)
esf_blur = extract_esf(blurred)
lsf_blur = compute_lsf(esf_blur)
mtf_blur = compute_mtf(lsf_blur)

# Plot
fig, axs = plt.subplots(2, 3, figsize=(15, 6))
axs[0, 0].imshow(edge_img, cmap='gray'); axs[0, 0].set_title("Original Edge")
axs[0, 1].plot(esf); axs[0, 1].set_title("ESF"); axs[0, 1].grid()
axs[0, 2].plot(mtf, label='Original'); axs[0, 2].set_title("MTF"); axs[0, 2].grid()

axs[1, 0].imshow(blurred, cmap='gray'); axs[1, 0].set_title("Blurred (Defocus)")
axs[1, 1].plot(esf_blur); axs[1, 1].set_title("ESF (Blurred)"); axs[1, 1].grid()
axs[1, 2].plot(mtf_blur, label='Blurred', color='red'); axs[1, 2].set_title("MTF (Blurred)")
axs[1, 2].plot(mtf, label='Original', linestyle='--'); axs[1, 2].legend(); axs[1, 2].grid()

for ax in axs.flat:
    if hasattr(ax, 'axis'): ax.axis('off') if ax == axs[0,0] or ax == axs[1,0] else None

plt.tight_layout()
plt.show()
