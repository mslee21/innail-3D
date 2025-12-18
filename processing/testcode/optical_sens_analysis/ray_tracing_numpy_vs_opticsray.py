
import numpy as np
import matplotlib.pyplot as plt

try:
    from opticsray import Ray, ThinLens
    opticsray_available = True
except ImportError:
    opticsray_available = False
    print("⚠️ opticsray is not installed. Install with: pip install opticsray")

# ---------- ① Pure Numpy Ray Tracing ----------
def pure_numpy_trace():
    f_main = 8.0      # mm
    f_mla = 2.382     # mm
    mla_pos = f_main
    sensor_pos = mla_pos + f_mla

    num_rays = 7
    max_angle_deg = 5
    angles_deg = np.linspace(-max_angle_deg, max_angle_deg, num_rays)
    angles_rad = np.radians(angles_deg)
    start_x = -10
    start_y = np.linspace(-2, 2, num_rays)

    ray_paths_x = []
    ray_paths_y = []

    for y0, theta in zip(start_y, angles_rad):
        x1 = 0
        slope1 = np.tan(theta)
        y1 = y0 + (x1 - start_x) * slope1

        x2 = mla_pos
        y2 = 0
        slope2 = (y2 - y1) / (x2 - x1)

        x3 = sensor_pos
        y3 = y2 + slope2 * (x3 - x2)

        ray_paths_x.append([start_x, x1, x2, x3])
        ray_paths_y.append([y0, y1, y2, y3])

    plt.figure(figsize=(10, 4))
    for x_pts, y_pts in zip(ray_paths_x, ray_paths_y):
        plt.plot(x_pts, y_pts, 'b')
    plt.axvline(0, color='k', label='Main Lens')
    plt.axvline(mla_pos, color='g', label='MLA')
    plt.axvline(sensor_pos, color='r', linestyle='--', label='Sensor')
    plt.title("① Pure Numpy Ray Tracing")
    plt.xlabel("Optical Axis (mm)")
    plt.ylabel("Height (mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- ② opticsray Ray Tracing ----------
def opticsray_trace():
    if not opticsray_available:
        return
    from opticsray import Ray, ThinLens

    f_main = 8.0
    f_mla = 2.382
    mla_pos = f_main
    sensor_pos = mla_pos + f_mla

    start_x = -10
    angles_deg = np.linspace(-5, 5, 7)
    angles_rad = np.radians(angles_deg)

    plt.figure(figsize=(10, 4))
    for theta in angles_rad:
        y0 = np.tan(theta) * (0 - start_x)
        ray = Ray(x=start_x, y=y0, angle=theta)
        ray = ThinLens(position=0, focal_length=f_main).propagate(ray)
        ray = ThinLens(position=mla_pos, focal_length=f_mla).propagate(ray)
        ray.propagate_to(sensor_pos)
        xs, ys = zip(*ray.history)
        plt.plot(xs, ys, 'b')

    plt.axvline(0, color='k', label='Main Lens')
    plt.axvline(mla_pos, color='g', label='MLA')
    plt.axvline(sensor_pos, color='r', linestyle='--', label='Sensor')
    plt.title("② Ray Tracing with opticsray")
    plt.xlabel("Optical Axis (mm)")
    plt.ylabel("Height (mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 실행 ---
pure_numpy_trace()
opticsray_trace()
