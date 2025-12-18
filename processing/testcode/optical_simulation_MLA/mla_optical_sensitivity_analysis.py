import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches

# --- CONFIG PARAMETERS ---
f_nominal = 2.382  # mm (MLA focal length)
f_tolerance = 0.03
f_error = f_nominal * f_tolerance

mla_pitch_um = 1500
mla_pitch_mm = mla_pitch_um / 1000

sensor_width_mm = 6.4
sensor_height_mm = 4.8
pixel_pitch_um = 1.55

# === ① MLA ↔ Sensor 거리 민감도 ===
sensor_distances = [f_nominal - f_error, f_nominal, f_nominal + f_error]
labels1 = ['-3% (Under)', 'Nominal', '+3% (Over)']

size = 100
x = np.linspace(-1, 1, size)
xx, yy = np.meshgrid(x, x)
spot = np.exp(-(xx**2 + yy**2) * 20)

defocus_images = []
for d in sensor_distances:
    defocus_amount = abs(d - f_nominal) / 0.01
    blurred = gaussian_filter(spot, sigma=defocus_amount)
    defocus_images.append(blurred)

fig1, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, label in zip(axes, defocus_images, labels1):
    ax.imshow(img, cmap='hot')
    ax.set_title(f"Sensor ΔZ: {label}")
    ax.axis('off')
plt.suptitle("① MLA-to-Sensor Distance Sensitivity")
plt.tight_layout()

# === ② Sensor ↔ MLA 매핑 ===
resolution_x = int(sensor_width_mm * 1000 / pixel_pitch_um)
resolution_y = int(sensor_height_mm * 1000 / pixel_pitch_um)
num_mla_x = int(sensor_width_mm / mla_pitch_mm)
num_mla_y = int(sensor_height_mm / mla_pitch_mm)
pixels_per_mla_x = resolution_x / num_mla_x
pixels_per_mla_y = resolution_y / num_mla_y

fig2, ax = plt.subplots(figsize=(8, 6))
ax.add_patch(patches.Rectangle((0, 0), sensor_width_mm, sensor_height_mm,
                               linewidth=1.5, edgecolor='black', facecolor='none', label='Sensor'))
for i in range(num_mla_x):
    for j in range(num_mla_y):
        ax.add_patch(patches.Rectangle((i * mla_pitch_mm, j * mla_pitch_mm),
                                       mla_pitch_mm, mla_pitch_mm,
                                       linewidth=0.5, edgecolor='blue', facecolor='none'))
ax.set_xlim(0, sensor_width_mm)
ax.set_ylim(0, sensor_height_mm)
ax.set_aspect('equal')
ax.set_xlabel("Width (mm)")
ax.set_ylabel("Height (mm)")
ax.set_title("② Sensor & MLA Mapping (Top View)")
ax.legend()
plt.tight_layout()

# === ③ 초점거리 편차 민감도 분석 ===
focal_lengths = [f_nominal * (1 - f_tolerance), f_nominal, f_nominal * (1 + f_tolerance)]
labels3 = ["-3% (Shorter f)", "Nominal", "+3% (Longer f)"]
sensor_distance = f_nominal

blurred_images = []
for f in focal_lengths:
    defocus_amount = abs(f - sensor_distance) / 0.01
    blurred = gaussian_filter(spot, sigma=defocus_amount)
    blurred_images.append(blurred)

fig3, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, label in zip(axes, blurred_images, labels3):
    ax.imshow(img, cmap='hot')
    ax.set_title(f"Focal Length: {label}")
    ax.axis('off')
plt.suptitle("③ Focal Length Variation Sensitivity (±3%)")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 시스템 파라미터
f_main = 8.0          # mm, main lens focal length
f_mla = 2.382         # mm, MLA focal length
mla_pos = f_main      # MLA 위치 = main lens image plane
sensor_pos = mla_pos + f_mla  # 센서 위치

# 광선 수 및 시야각
num_rays = 7
max_angle_deg = 5
angles_deg = np.linspace(-max_angle_deg, max_angle_deg, num_rays)
angles_rad = np.radians(angles_deg)

# Scene에서 ray 발사 시작점
start_x = -10  # mm, 객체 거리
start_y = np.linspace(-2, 2, num_rays)

ray_paths_x = []
ray_paths_y = []

for y0, theta in zip(start_y, angles_rad):
    # ① Main lens 앞까지 직진
    x1 = 0
    slope1 = np.tan(theta)
    y1 = y0 + (x1 - start_x) * slope1

    # ② Main lens에서 굴절 → 초점(f_main)으로 향함
    x2 = mla_pos
    y2 = 0  # Main lens가 f_main에 상을 맺음
    slope2 = (y2 - y1) / (x2 - x1)

    # ③ MLA에서 다시 굴절 → MLA 초점(f_mla)으로 향함
    x3 = sensor_pos
    y3 = y2 + slope2 * (x3 - x2)

    ray_paths_x.append([start_x, x1, x2, x3])
    ray_paths_y.append([y0, y1, y2, y3])

# 시각화
plt.figure(figsize=(10, 4))
for x_pts, y_pts in zip(ray_paths_x, ray_paths_y):
    plt.plot(x_pts, y_pts, 'b')

# 주요 요소 위치 표시
plt.axvline(0, color='k', linestyle='-', label="Main Lens")
plt.axvline(mla_pos, color='g', linestyle='-', label="MLA Plane")
plt.axvline(sensor_pos, color='r', linestyle='--', label="Sensor Plane")

plt.xlabel("Optical Axis (mm)")
plt.ylabel("Height (mm)")
plt.title("2D Ray Tracing of Plenoptic 2.0 System")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

from pyrayoptics import Lens, OpticalSystem
import numpy as np
import matplotlib.pyplot as plt

# 시스템 생성
sys = OpticalSystem()
sys.add_surface(Lens(radius=np.inf, thickness=8.0, refr_index=1.0))   # Main lens
sys.add_surface(Lens(radius=np.inf, thickness=2.382, refr_index=1.0)) # MLA
sys.build()

# 광선 추적
rays = []
angles_deg = np.linspace(-5, 5, 7)
for angle in angles_deg:
    rays.append(sys.trace_ray(y0=angle / 5.0, theta_deg=angle))

# 시각화
fig, ax = plt.subplots(figsize=(10, 4))
for ray in rays:
    ax.plot(ray.x, ray.y, 'b')

ax.axvline(0, color='k', label='Main Lens')
ax.axvline(8.0, color='g', label='MLA')
ax.axvline(10.382, color='r', linestyle='--', label='Sensor')
ax.set_title("Ray Trace with pyrayoptics")
ax.set_xlabel("Optical Axis (mm)")
ax.set_ylabel("Height (mm)")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()