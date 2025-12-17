import numpy as np

from plenopticam_etri_09.cfg import PlenopticamConfig
cfg = PlenopticamConfig()
cfg.params[cfg.cal_meta] = ""
cfg.load_cal_data()
mic_list = np.asarray(cfg.calibs[cfg.mic_list])

from plenopticam_etri_09.misc import PlenopticamStatus
sta = PlenopticamStatus()

# do grid fitting
from plenopticam_etri_09.lfp_calibrator import GridFitter
obj = GridFitter(cfg=cfg, sta=sta)
obj.main()
new_list = np.asarray(obj.grid_fit)

# plot
import matplotlib.pyplot as plt
plt.figure()
plt.plot(mic_list[:, 1], mic_list[:, 0], 'rx')
plt.plot(new_list[:, 1], new_list[:, 0], 'b+')
plt.show()