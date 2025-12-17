import rayoptics
print(rayoptics.__file__)

# 이후 로직
import numpy as np
import matplotlib.pyplot as plt
from rayoptics.opmodel.model import OpticalModel
from rayoptics.seq import medium

opt_model = OpticalModel()
opt_model.seq_model.list_surface.clear()

# 정의: S-TiH53 렌즈
glass = medium.Glass('S-TiH53', nd=1.83956)

# 렌즈 앞면 (convex)
opt_model.seq_model.insert_surface(0)
opt_model.seq_model.set_radius(0, 1.0)         # mm
opt_model.seq_model.set_thickness(0, 0.14595)  # mm
opt_model.seq_model.set_medium(0, glass)

# 렌즈 뒷면 (flat)
opt_model.seq_model.insert_surface(1)
opt_model.seq_model.set_radius(1, np.inf)
opt_model.seq_model.set_thickness(1, 2.382)    # EFL
opt_model.seq_model.set_medium(1, medium.air)

# 이미지 센서
opt_model.seq_model.insert_surface(2)
opt_model.seq_model.set_radius(2, np.inf)
opt_model.seq_model.set_thickness(2, 0.0)
opt_model.seq_model.set_medium(2, medium.air)

# 업데이트 및 시각화
opt_model.update_model()
fig, ax = plt.subplots(figsize=(10, 4))
opt_model.plot('ray', num_rays=7, pupil_sampling='full', fig=fig, ax=ax)
plt.title("Ray Trace of MLA Lens (S-TiH53)")
plt.tight_layout()
plt.show()