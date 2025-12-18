import sys
HOME = "./"
# DATAHOME = "/home/mslee21/CaveHalley/[CodeCellar]/[2020-0-00168]/03-Research/23.12.12. (Cressem)/Cressem_plenoptic_camera/"
#DATAHOME = '/Volumes/mslee21@macbook14/Dropbox@macbook14/Dropbox/CaveDropbox/[CodeCellar]/[2020-0-00168]/03-Research/24.02.08_color_125um_pitch/'
#DATAHOME = '/Volumes/mslee21@macbook14/Dropbox@macbook14/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/reconstruction/centroid_extraction/test_data/'
DATAHOME = '/Volumes/mslee21@macbook14/Dropbox@macbook14/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/reconstruction/centroid_extraction/test_data/'

my_library = "/Volumes/mslee21@macbook14/Dropbox@macbook14/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/reconstruction/my_library/"
sys.path.append(HOME)
sys.path.append(DATAHOME)
sys.path.append(my_library)


print('HOME : ', HOME +'\n')
print('DATAHOME : ', DATAHOME +'\n')

import numpy as np
print('Python v'+sys.version+'\n')
sys.path.append(HOME + '../my_library/plenopticam_etri_09/')

import plenopticam_etri_09 as pcam
print('PlenoptiCam v'+pcam.__version__+'\n')
print(pcam)

import matplotlib.pyplot as plt
import cv2

# instantiate config object and set image file paths and options
cfg = pcam.cfg.PlenopticamConfig()
print("mslee21/cfg file path :",cfg._dir_path)
cfg.default_values()

## Calibration File path
# cal_img = 'reference2.tif'
cal_img = 'a-1.bmp'
#cfg.params[cfg.cal_path] = HOME + '(sample)/ETRI/23.04.26. (PM-V4-001) 레퍼런스 이미지/' + cal_img
cfg.params[cfg.cal_path] = DATAHOME + cal_img
cfg.params[cfg.opt_cali] = True

cfg.params[cfg.ptc_leng] = 9 # B0433
#cfg.params[cfg.ptc_leng] = 51

## calibration method, CALI_METH = ('area', 'peak', 'grid-fit', 'vign-fit', 'corn-fit')
cfg.params[cfg.cal_meth] = pcam.cfg.constants.CALI_METH[0]
print("mslee21/CALI_METH : ", cfg.params[cfg.cal_meth])
# instantiate status object to display processing progress

# Plenopticam에서 사용하는 status를 표시,progress, interrupt 등
sta = pcam.misc.PlenopticamStatus()

wht_img = pcam.misc.load_img_file(cfg.params[cfg.cal_path])
print("mslee21/original wht_img", wht_img.shape)

### Invalid number of channels in input image:
###     'VScn::contains(scn)'
### where
###     'scn' is 3
### 위 처럼 나오면 wht_img = wht_img[:,:,0]
### BGR to GRAY expects 3 channels . But your image has 2 channels and it will not work., When you do
### grayscale = open_cv_image[:, :, 0]
### you are considering first channel as gray
if wht_img.shape[2] == 3:
    wht_img = wht_img[:,:,0]
else:
    wht_img = wht_img

def plot_img(wht_img, title = None):
    s = 3
    h, w, c = wht_img.shape if len(wht_img.shape) == 3 else wht_img.shape + (1,)
    # hp, wp = 1000, 1000
    hp, wp = 200, 200
    fig, axs = plt.subplots(s, s, facecolor='w', edgecolor='k')
    fig.suptitle(title)

    for i in range(s):
        for j in range(s):
            # plot cropped image part
            # k = i * (h // s) + (h // s) // 2 - hp // 2
            k = i * ((h - hp) // 2)
            print("j, i, k, h//s, hp//2", j, i, k, (h // s) // 2, hp // 2)
            l = j * ((w - wp) // 2)
            # l = j * (w // s) + (w // s) // 2 - wp // 2
            print("j, i, l, h//s, hp//2", j, i, l, (w // s) // 2, wp // 2)

            axs[i, j].imshow(wht_img[k:k + hp, l:l + wp, ...], cmap='gray')
            axs[i, j].grid(False)
            axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                  labelleft=True, labelbottom=True)
            axs[i, j].set_yticks(range(0, hp + 1, hp // 2))
            axs[i, j].set_xticks(range(0, wp + 1, wp // 2))
            axs[i, j].set_yticklabels([str(k), str(k + hp // 2), str(k + hp)])
            axs[i, j].set_xticklabels([str(l), str(l + wp // 2), str(l + wp)])
    # set common labels
    fig.text(0.5, -0.05, 'Horizontal dimension [px]', ha='center', va='center', fontsize=14)
    fig.text(-0.01, 0.5, 'Vertical dimension [px]', ha='center', va='center', rotation='vertical', fontsize=14)

    fig.tight_layout()
    plt.legend(loc='upper right', bbox_to_anchor=(3, 3.85), fancybox=True, shadow=True)
    # plt.savefig(directory_path + "/" + filename + ".png")
    plt.show()


# wht_img = wht_img[:,:,0]
wht_img = cv2.cvtColor(wht_img, cv2.COLOR_GRAY2RGB)
# wht_img = cv2.rotate(wht_img, cv2.ROTATE_90_CLOCKWISE)
print("mslee21/wht_img GRAY2RGB.shape", wht_img.shape)
# %%
# center_x, center_y = wht_img.shape[0] // 2, wht_img.shape[1] // 2
# # 이미지를 중앙에서 256x256 크기로 자르기
# size = 512
# cropped_img = wht_img[center_x - size:center_x + size, center_y - size:center_y + size]
# wht_img = cropped_img
# %%
plt.figure()
plt.imshow(wht_img, cmap='gray', interpolation='none')
plt.grid(False)
plt.title('White calibration image original')
plt.savefig(DATAHOME + cal_img.split('.')[0] + '.jpg')
plt.show()
print(DATAHOME + cal_img.split('.')[0] + '.png')
# %%

plot_img(wht_img, title = "original image")

# # ##### 2023.12.27 ####
# # # reference image 의 회전을 시켜서 mla_list가 정상적으로 출력되도록 조정
# # # 이미지의 크기를 잡고 이미지의 중심을 계산합니다.
# # 이미지의 중심을 중심으로 이미지를 0.7도 회전합니다.
(h, w) = wht_img.shape[:2]
(cX, cY) = (w // 2, h // 2)
M = cv2.getRotationMatrix2D((cX, cY), -0.1, 1.0)
wht_img = cv2.warpAffine(wht_img, M, (w, h))
# wht_img = wht_img(10:h-10,10::w)
# print(wht_img.shape)

plt.figure()
plt.imshow(wht_img, cmap='gray', interpolation='none')
plt.grid(False)
plt.title('White calibration image rotate')
plt.savefig(DATAHOME + cal_img.split('.')[0] + '_rotate.png')
plt.show()
print(DATAHOME + cal_img.split('.')[0] + '_rotate.png')
# %%

plot_img(wht_img, title = "rotated image")

s = 3
h, w, c = wht_img.shape if len(wht_img.shape) == 3 else wht_img.shape + (1,)
# hp, wp = 1000, 1000
hp, wp = 200, 200
fig, axs = plt.subplots(s, s, facecolor='w', edgecolor='k')

for i in range(s):
    for j in range(s):
        # plot cropped image part
        # k = i * (h // s) + (h // s) // 2 - hp // 2
        k=i*((h-hp)//2)
        print("j, i, k, h//s, hp//2", j, i, k, (h//s) //2, hp//2)
        l=j*((w-wp)//2)
        # l = j * (w // s) + (w // s) // 2 - wp // 2
        print("j, i, l, h//s, hp//2", j, i, l, (w // s) // 2, wp // 2)

        axs[i, j].imshow(wht_img[k:k + hp, l:l + wp, ...], cmap='gray')
        axs[i, j].grid(False)
        axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                              labelleft=True, labelbottom=True)
        axs[i, j].set_yticks(range(0, hp + 1, hp // 2))
        axs[i, j].set_xticks(range(0, wp + 1, wp // 2))
        axs[i, j].set_yticklabels([str(k), str(k + hp // 2), str(k + hp)])
        axs[i, j].set_xticklabels([str(l), str(l + wp // 2), str(l + wp)])
# set common labels
fig.text(0.5, -0.05, 'Horizontal dimension [px]', ha='center', va='center', fontsize=14)
fig.text(-0.01, 0.5, 'Vertical dimension [px]', ha='center', va='center', rotation='vertical', fontsize=14)

fig.tight_layout()
plt.legend(loc='upper right', bbox_to_anchor=(3, 3.85), fancybox=True, shadow=True)
# plt.savefig(directory_path + "/" + filename + ".png")
plt.show()
# %%
# from PIL import Image
# save_wht_img = Image.fromarray(wht_img)
# wht_img_tif = HOME + '/(sample)/ETRI/23.04.14. PM-V4-001, V7-001, DOF/PM-V4-001_DOF target/reference.tif'

# save_wht_img.save(wht_img_tif)
# %% md
# %% md
## Micro image calibration

# %%
print(cfg.params[cfg.ptc_leng])
# %%
import os


def find_tif_files(folder_path):
    tif_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".tif"):
                tif_files.append(os.path.join(root, file))

    return tif_files


folder_path = HOME + '/'
tif_files = find_tif_files(folder_path)
print(tif_files)
# %%
# wht_img = pcam.misc.load_img_file(cfg.params[cfg.cal_path])

# grayscale 변환
if wht_img.ndim == 3:
    gray = cv2.cvtColor(wht_img, cv2.COLOR_BGR2GRAY)
else:
    gray = wht_img

# 8bit 강제 변환
gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# CLAHE 대비 강화
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
enhanced = clahe.apply(gray)

enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
# (선택) Blur로 LoG 안정화
blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

cv2.imwrite("wht_img_preprocessed.png", blurred)
cal_obj = pcam.lfp_calibrator.LfpCalibrator(blurred, cfg, sta)

# cal_obj.cfg.params[cfg.ptc_leng] = 14
## lfp calibration.LfpCalibrator에서 M=80으로 설정
# cal_obj._M = 80
cal_obj._M = 9 # B0433
# cal_obj._t = 0.001
print("cal_obj._M", cal_obj._M)


print("test1")
ret = cal_obj.main()
cfg = cal_obj.cfg
# del cal_obj
# %%
ret = cfg.load_cal_data()


print(f"Detected centroids: {len(cfg.calibs[cfg.mic_list])}")



y_coords = [row[0] for row in cfg.calibs[cfg.mic_list]]
x_coords = [row[1] for row in cfg.calibs[cfg.mic_list]]

s = 3
h, w, c = wht_img.shape if len(wht_img.shape) == 3 else wht_img.shape + (1,)
# hp, wp = 1000, 1000
hp, wp = 200, 200
fig, axs = plt.subplots(s, s, facecolor='w', edgecolor='k')

for i in range(s):
    for j in range(s):
        # plot cropped image part
        # k = i * (h // s) + (h // s) // 2 - hp // 2
        k=i*((h-hp)//2)
        print("j, i, k, h//s, hp//2", j, i, k, (h//s) //2, hp//2)
        l=j*((w-wp)//2)
        # l = j * (w // s) + (w // s) // 2 - wp // 2
        print("j, i, l, h//s, hp//2", j, i, l, (w // s) // 2, wp // 2)
        axs[i, j].imshow(wht_img[k:k + hp, l:l + wp, ...], cmap='gray')

        # plot centroids in cropped area
        coords_crop = [(y, x) for y, x in zip(y_coords, x_coords)
                       if k <= y <= k + hp - .5 and l <= x <= l + wp - .5]
        y_centroids = [row[0] - k for row in coords_crop]
        x_centroids = [row[1] - l for row in coords_crop]
        axs[i, j].plot(x_centroids, y_centroids, 'bx',
                       markersize=4, label=r'Centroids $\mathbf{c}_{j,h}$')
        axs[i, j].grid(False)
        axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                              labelleft=True, labelbottom=True)
        axs[i, j].set_yticks(range(0, hp + 1, hp // 2))
        axs[i, j].set_xticks(range(0, wp + 1, wp // 2))
        axs[i, j].set_yticklabels([str(k), str(k + hp // 2), str(k + hp)])
        axs[i, j].set_xticklabels([str(l), str(l + wp // 2), str(l + wp)])

# set common labels
fig.text(0.5, -0.05, 'Horizontal dimension [px]', ha='center', va='center', fontsize=14)
fig.text(-0.01, 0.5, 'Vertical dimension [px]', ha='center', va='center', rotation='vertical', fontsize=14)

fig.tight_layout()
plt.legend(loc='upper right', bbox_to_anchor=(3, 3.85), fancybox=True, shadow=True)
# plt.savefig(directory_path + "/" + filename + ".png")
plt.show()
# %%
