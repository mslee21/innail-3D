
### base code: PlenoMatrix_master_0.1_Cert_240214.py


import sys
sys.path.append(
    "./Net/")
import time
import os
import pickle
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from IPython.display import HTML
import argparse
from utils import *
from model import Net
import argparse
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

HOME = "../"
my_library = "../my_library/"

sys.path.append(my_library)
sys.path.append(my_library + 'plenopticam_etri_09/')
# sys.path.append(my_library)
import plenopticam_etri_09 as pcam

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

sys.path.append("./Depth-Anything/")
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

print('Python v'+sys.version+'\n')
print('PlenoptiCam v'+ pcam.__version__+'\n')
print(pcam)

import inspect
print(inspect.getfile(DepthAnything))

def saveimg(data, path):
    if data.dtype == np.float32:
        data = cv2.convertScaleAbs(data * 255)
    else:
        data = np.uint8(data)

    if data.ndim == 2:
        color_space = cv2.COLOR_GRAY2BGR
    else:
        color_space = cv2.COLOR_RGB2BGR
    cv2.imwrite(path, cv2.cvtColor(data, color_space))

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def showfigure(title, image, path=None):
    plt.figure()
    plt.imshow(image, interpolation='none')
    plt.grid(False)
    plt.title(title)
    if path:
        plt.savefig(path)
    plt.show()

def gray2color(raw_img):
    print("raw_img.shape", raw_img.shape)
    if len(raw_img.shape) == 2:
        return cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
    elif len(raw_img.shape) == 3:
        return raw_img
    else:
        return raw_img
def sparse_color_depthmap(depthmap, max_range):
    cmap = cm.get_cmap("jet", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # sparse depthmap인 경우 depth가 있는 곳만 추출합니다.
    depth_pixel_v_s, depth_pixel_u_s = np.where(depthmap > 0)

    H, W = depthmap.shape
    color_depthmap = np.zeros((H, W, 3)).astype(np.uint8)
    for depth_pixel_v, depth_pixel_u in zip(depth_pixel_v_s, depth_pixel_u_s):
        depth = depthmap[depth_pixel_v, depth_pixel_u]
        color_index = int(255 * min(depth, max_range) / max_range)
        color = cmap[color_index, :]
        cv2.circle(color_depthmap, (depth_pixel_u, depth_pixel_v), 1, color=tuple(color), thickness=-1)
    return color_depthmap

def full_gray_depthmap(depthmap, max_range):

    # sparse depthmap인 경우 depth가 있는 곳만 추출합니다.
    depth_pixel_v_s, depth_pixel_u_s = np.where(depthmap > 0)

    H, W = depthmap.shape
    color_depthmap = np.zeros((H, W, 3)).astype(np.uint8)
    for depth_pixel_v, depth_pixel_u in zip(depth_pixel_v_s, depth_pixel_u_s):
        depth = depthmap[depth_pixel_v, depth_pixel_u]
        color_index = int(255 * min(depth, max_range) / max_range)
        color = cmap[color_index, :]
        cv2.circle(color_depthmap, (depth_pixel_u, depth_pixel_v), 1, color=tuple(color), thickness=-1)
    return color_depthmap


def lfp_img_align(lfp_img, wht_img, cfg):

    if cfg.lfpimg:
        # hot pixel correction
        obj = pcam.lfp_aligner.CfaOutliers(bay_img=lfp_img, cfg=cfg, sta=sta)
        obj.rectify_candidates_bayer(n=9, sig_lev=2.5)
        _lfp_img = obj.bay_img
        del obj

    if cfg.lfpimg and len(lfp_img.shape) == 2:
        # perform color filter array management and obtain rgb image
        cfa_obj = pcam.lfp_aligner.CfaProcessor(bay_img=_lfp_img, wht_img=wht_img, cfg=cfg, sta=sta)
        cfa_obj.main()
        _lfp_img = cfa_obj.rgb_img
        CfaProcessor_lfp_img = cfa_obj.rgb_img
        del cfa_obj

    if lfp_img is not None:
        if cfg.params[cfg.opt_rota] == True:
            # de-rotate centroids
            obj = pcam.lfp_aligner.LfpRotator(lfp_img, cfg.calibs[cfg.mic_list], rad=None, cfg=cfg, sta=sta)
            obj.main()
            lfp_img, cfg.calibs[cfg.mic_list] = obj.lfp_img, obj.centroids

        else:
            obj = pcam.lfp_aligner.LfpRotator(lfp_img, cfg.calibs[cfg.mic_list], rad=float(cfg.params[cfg.val_rota]), cfg=cfg, sta=sta)
            obj.main()
            lfp_img, cfg.calibs[cfg.mic_list] = obj.lfp_img, obj.centroids
        del obj

    # interpolate each micro image with its MIC as the center with consistent micro image size
    obj = pcam.lfp_aligner.LfpResampler(lfp_img=lfp_img, cfg=cfg, sta=sta, method='linear')
    obj.main()
    lfp_img_align = obj.lfp_out()
    del obj

    # micro image crop
    lfp_obj = pcam.lfp_extractor.LfpCropper(lfp_img_align=lfp_img_align, cfg=cfg, sta=sta)
    lfp_obj.main()
    lfp_img_align = lfp_obj.lfp_img_align

    # if cfg.params[cfg.opt_view]:
    #     obj = pcam.lfp_extractor.LfpRearranger(lfp_img_align, cfg=cfg, sta=sta)
    #     obj.main()
    #     vp_img_linear = obj.vp_img_arr
    #     del obj

    return lfp_img_align

def depth_from_net(lf_sa_img, cfg):

    # OACCNET configure
    cfg.angRes = 9
    cfg.device = 'cuda:0'
    cfg.model_name = 'OACC-Net'
    cfg.crop = False
    cfg.patchsize = int(128)
    cfg.minibatch_test = int(4)
    cfg.model_path = './log/OACC-Net.pth.tar'
    cfg.save_path = RESULT

    cnt_img_stack = int(lf_sa_img.shape[0] / 2)
    lf_angCrop = lf_sa_img[cnt_img_stack-5:cnt_img_stack+4:1,cnt_img_stack-5:cnt_img_stack+4:1,:,:,0]

    lf_angCrop = lf_angCrop.astype('float16')
    net = Net(cfg.angRes)
    net.to(cfg.device)
    model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
    net.load_state_dict(model['state_dict'])

    if cfg.crop == False:
        data = rearrange(lf_angCrop, 'u v h w -> (u h) (v w) ')
        data = ToTensor()(data.copy())
        data = data.float()
        # print('data.shape',data.shape)
        with torch.no_grad():
            disp = net(data.unsqueeze(0).to(cfg.device))
        disp = np.float32(disp[0, 0, :, :].data.cpu())

    else:
        patchsize = cfg.patchsize
        stride = patchsize // 2
        data = torch.from_numpy(lf_angCrop)
        data = data.float()
        sub_lfs = LFdivide(data.unsqueeze(2), patchsize, stride)
        n1, n2, u, v, c, h, w = sub_lfs.shape
        sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')
        mini_batch = cfg.minibatch_test
        num_inference = (n1 * n2) // mini_batch

        with torch.no_grad():
            out_disp = []
            for idx_inference in range(num_inference):
                current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :]
                input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                out_disp.append(net(input_data.to(cfg.device)))

            if (n1 * n2) % mini_batch:
                current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                input_data = rearrange(current_lfs, 'b u v c h w -> b c (u h) (v w)')
                out_disp.append(net(input_data.to(cfg.device)))

        out_disps = torch.cat(out_disp, dim=0)
        out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
        disp = LFintegrate(out_disps, patchsize, patchsize // 2)
        disp = disp[0: data.shape[2], 0: data.shape[3]]
        disp = np.float32(disp.data.cpu())

    return data, disp

def depth_from_epi(vp_img_linear, cfg=None):

    epi_mai = rearrange(vp_img_linear, 'u v h w c-> (u h) (v w) c')

    vp_img_arr = vp_img_linear.copy() if vp_img_linear is not None else None
    if cfg.params[cfg.opt_view]:
        obj = pcam.lfp_extractor.LfpExporter(vp_img_arr=vp_img_arr, cfg=cfg, sta=sta)
        obj.write_viewpoint_data()

    if cfg.params[cfg.opt_dpth]:
        x, y, _, _, _ = vp_img_arr.shape
        if x % 2 == 0 and y % 2 == 0:
            print("x and y are both even.")
            vp_img_arr = vp_img_arr[:(x - 1), :(x - 1), :, :, :]
        else:
            pass
        obj = pcam.lfp_extractor.LfpDepth(vp_img_arr=vp_img_arr, cfg=cfg, sta=sta)
        obj.main()
        depth_map = obj.depth_map

    return epi_mai, depth_map, cfg.calibs

def depth_from_depthAnything(DEVICE, encoding = 'vits', cfg=None, ):

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoding)).to(
        DEVICE).eval()
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    return depth_anything, transform


def plot_PoI_img(wht_img, save_name=None, title=None, mic_list=None, crop=200 ):

    s = 3
    h, w, c = wht_img.shape if len(wht_img.shape) == 3 else wht_img.shape + (1,)
    # hp, wp = 1000, 1000
    hp, wp = crop, crop
    fig, axs = plt.subplots(s, s, facecolor='w', edgecolor='k')
    fig.suptitle(title)

    if not mic_list:
        for i in range(s):
            for j in range(s):
                # plot cropped image part
                k = i * ((h - hp) // 2)
                l = j * ((w - wp) // 2)

                axs[i, j].imshow(wht_img[k:k + hp, l:l + wp, ...], cmap='gray')
                axs[i, j].grid(False)
                axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                      labelleft=True, labelbottom=True)
                axs[i, j].set_yticks(range(0, hp + 1, hp // 2))
                axs[i, j].set_xticks(range(0, wp + 1, wp // 2))
                axs[i, j].set_yticklabels([str(k), str(k + hp // 2), str(k + hp)])
                axs[i, j].set_xticklabels([str(l), str(l + wp // 2), str(l + wp)])
    else:
        y_coords = [row[0] for row in mic_list]
        x_coords = [row[1] for row in mic_list]
        for i in range(s):
            for j in range(s):
                # plot cropped image part
                k = i * ((h - hp) // 2)
                l = j * ((w - wp) // 2)

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
    if not save_name == None:
        plt.savefig(save_name)
    plt.show()


def initialize():
    if torch.cuda.is_available():
        print("# of GPU:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("no GPUs")
    return True

def load_raw_img(image_path=None, cfg=None):

    dir, file = os.path.split(image_path)
    fname, ext = os.path.splitext(file)

    if ext == '.lfr':
        reader = pcam.lfp_reader.LfpReader(cfg, sta)
        reader.main()
        lfp_img = reader.lfp_img

        cal_finder = pcam.lfp_calibrator.CaliFinder(cfg, sta)
        cal_finder.main()
        wht_img = cal_finder.wht_bay

        spath = reader.dp
        reader.fn = os.path.basename(spath) +'_whtimg.tiff'
        wpath = os.path.join(spath, reader.fn)
        pcam.misc.save_img_file(pcam.misc.Normalizer(wht_img).uint16_norm(), file_path=wpath, file_type='tiff')

    else :
        dir, file = os.path.split(image_path)
        fname, ext = os.path.splitext(file)
        spath = os.path.join(dir, fname)
        lfp_img = pcam.misc.load_img_file(image_path)
        wht_img = pcam.misc.load_img_file(cfg.params[cfg.cal_path])

    return spath, lfp_img, wht_img

def cal_mla(cfg) :
    if not cfg.params[cfg.cal_path] == None:
        wht_img = pcam.misc.load_img_file(cfg.params[cfg.cal_path])
        # wht_img = cv2.cvtColor(wht_img, cv2.COLOR_GRAY2RGB)
        cfg.params[cfg.opt_cali] = True
        cfg.params[cfg.cal_meth] = pcam.cfg.constants.CALI_METH[2]
        print("mslee21/CALI_METH : ", pcam.cfg.constants.CALI_METH[2])

        plot_PoI_img(wht_img=wht_img, title = "REF original image", crop=200)

        (h, w) = wht_img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rot_angle = 0.2
        M = cv2.getRotationMatrix2D((cX, cY), float(cfg.params[cfg.img_rota]), 1.0)
        wht_img = cv2.warpAffine(wht_img, M, (w, h))
        plot_PoI_img(wht_img=wht_img, title="REF rotated image", crop=200)

        save_wht_img = (wht_img * 255).astype('uint8')
        cv2.imwrite(RESULT + "%s_wht_img_rot_%s.jpg" % (fname, str(cfg.params[cfg.img_rota])), save_wht_img)

        cal_obj = pcam.lfp_calibrator.LfpCalibrator(wht_img, cfg, sta)
        ## lfp calibration.LfpCalibrator에서 M=80으로 설정
        # cal_obj._M = 82 # PM-V1-001/ETRI
        # cal_obj._M = 32 # PM-V4-001
        cal_obj._M = 82 # PM-V1-001/Cressem
        # cal_obj._M = 13  # Lytro
        ret = cal_obj.main()
        cfg = cal_obj.cfg

        del cal_obj

        plot_PoI_img(wht_img=wht_img, mic_list=cfg.calibs[cfg.mic_list], crop=200)
    else:
        pass

    return ret

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--TP', type=str, default="01")
    parser.add_argument('--algorithm', type=str,
                        default='depthAny')
    # parser.add_argument('--raw_img_path', type=str, default='/home/mslee21/CaveHalley/[CodeCellar]/[2020-0-00168]/03-Research/'
    parser.add_argument('--raw_img_path', type=str,
                        default='./Cert_image/mslee21/'
                                                            # '24.02.02. (lytro)/B5151502880/Lytro_IMG_0084/IMG_0084.lfr')
                                                            # '23.11.24. (PM-V4-001) P1.0 depthTest/23.11.24_PM-V4-001_P1.0_h25um/ring/dog_bone_x10.tif')
                                                            # '[Plenomatrix] master/test_img/DepthAnything/demo16.png')
                                                            # '24.01.18. (Depth_test)/img_test.tif')
                                                            # '23.12.11-15. (JNU)/1211-15. JNU/1215/1215-1 (1).tif')
                                                            # '24.01.18. (Depth_test)/2d_grating_5um_x10.png')
                                                            # '24.02.08_color_125um_pitch/24.02.08_dogbone_h25um/dogbone02.tif')
                                                            # '24.02.06.Color_125um_pitch/Sample_H25.36um_none_coating/x10_grating_dogbone_ring lamp/x10_pitch_10um.tif')
                                                            # '24.02.08_color_125um_pitch/Resistance/resistor-1.tif')
                                                            # '24.02.08_color_125um_pitch/24.02.08_dogbone_h1um/dogbone4.tif')
                                                            # '24.02.16_Color_125um_pitch_h25um_20umpitch_x10_x20_x50/H27.54um_Cr_coating/h25_W_Cr_20um_pitch_coaxial_x50-3.tif')
                                                            'mslee21_52_20250529_102637_1.jpg')
    parser.add_argument('--raw_dir_path', type=str,
                        default='./Cert_image/mslee21/')

    # parser.add_argument('--raw_vdo_path', type=str,default='/home/mslee21/CaveHalley/[CodeCellar]/[2020-0-00168]/03-Research/'
    parser.add_argument('--raw_vdo_path', type=str,default='/Volumes/mslee21@studio/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/depth_map/test_img/'
                                                           '24.02.08_color_125um_pitch/Video/40fps.mp4')
    # parser.add_argument("--cal_img_path", type=str, default='/home/mslee21/CaveHalley/[CodeCellar]/[2020-0-00168]/03-Research/'
    parser.add_argument("--cal_img_path", type=str,
                        default='./Cert_image/ref_image/'
                                                            # '24.02.02. (lytro)/B5151502880/B5151502880.tar')
                                                            # '23.11.24. (PM-V4-001) P1.0 depthTest/23.11.24_PM-V4-001_P1.0_h25um/ring/reference-1.jpg')
                                                            # '24.01.18. (Depth_test)/mla_calibration_img.tif')
                                                            # '23.12.11-15. (JNU)/1211-15. JNU/1215/reference2.bmp')
                                                            # '24.02.06.Color_125um_pitch/Sample_H25.36um_none_coating/x10_grating_dogbone_ring lamp/reference2.bmp')
                                                            'reference2.tif')
    # parser.add_argument('--cal_dat_path', type=str, default= '/home/mslee21/CaveHalley/[CodeCellar]/[2020-0-00168]/03-Research/'
    parser.add_argument('--cal_dat_path', type=str,
                        default='./Cert_image/ref_image/'
                                                             # '24.02.02. (lytro)/B5151502880/Research/24/mod_0019.json')
                                                             # '23.11.24. (PM-V4-001) P1.0 depthTest/23.11.24_PM-V4-001_P1.0_h25um/ring/reference-1.json')
                                                             # '24.01.18. (Depth_test)/mla_calibration_img.json')
                                                             # '23.12.11-15. (JNU)/1211-15. JNU/1215/reference2.json')
                                                             'reference2.json')
                                                             # '24.02.06.Color_125um_pitch/Sample_H25.36um_none_coating/x10_grating_dogbone_ring lamp/reference2.json')
    parser.add_argument('--exec_cal', action='store_true', help = 'execute calibration')

    # parser.add_argument('--sel_dtp_meth', type=str, default='epi')
    # parser.add_argument('--save_dir', type=str, default='./Test/')

    # parser.add_argument('--crop', type=bool, default=False)
    # parser.add_argument('--patchsize', type=int, default=128)
    # parser.add_argument('--minibatch_test', type=int, default=4)
    # parser.add_argument('--model_path', type=str, default='./log/OACC-Net.pth.tar')
    # parser.add_argument('--save_path', type=str, default='./Results/')
    return parser.parse_args()

def plot_img(img, title, save_name):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        plt.figure()
        plt.imshow(img, interpolation='none')
        plt.grid(False)
        plt.title(title)
        plt.axis('off')  # 축을 숨깁니다 (선택 사항)
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        plt.show()

def img_info(img, name=None):
    print('type(%s): ' % name, type(img), "\n",
          '%s.dtype: ' % name, img.dtype, '\n',
          '%s.shape: ' % name, img.shape, '\n',
          '%s.max(): ' % name, img.max(), '\n',
          '%s.mean(): ' % name, np.mean(img[:,:,0] if img.ndim == 3 else img), '\n',
          '%s.min(): ' % name, img.min(), '\n',
          '%s.data : ' % name, img[:2, :2, :] if img.ndim == 3 else img[:2, :2])

def depth_profile(img, set_h=None, set_w=None, title = None, max_range=None):
    # hd, wd, cd= img.shape
    if not set_h == None:
        img_line = img[set_h, :, :]
    elif not set_w == None:
        img_line = img[:, set_w, :]
    img_avg = img_line.mean(axis=1)
    # print(img_avg[90])
    # print(img_avg[1000:1200])
    print("img_avg.max()",img_avg.max())
    print("img_avg.min()", img_avg.min())
    print("img.max()", img.max())
    print("img.min()", img.min())
    return img_avg

def plen_any_img(lfp_img, cfg):

    fpath, file = os.path.split(cfg.params[cfg.lfp_path])
    fname, ext = os.path.splitext(file)

    start_time = time.time()
    max_range = 255
    # filepath = '/home/mslee21/CaveHalley/[CodeCellar]/[2020-0-00168]/04-Experiment/[Plenomatrix] master/test_img/DepthAnything/coax_dog_bone_x10.tif'
    # filepath = '/Volumes/mslee21@studio/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/depth_map/test_img/DepthAnything/coax_dog_bone_x10.tif'
    # filepath = '/Volumes/mslee21@studio/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/depth_map/test_img/test_pl_nail_1.jpg'
    # filepath = '/Volumes/mslee21@studio/Dropbox/CaveDropbox/[GitCellar]/[24BK1300]/innail-3d/software/depth_map/test_img/test_pl_nail_1.jpg'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_anything, transform = depth_from_depthAnything(DEVICE, encoding='vitl', cfg=None)

    print("depth_anyting loading finish time", time.time() - start_time)

    h, w = lfp_img.shape[:2]
    raw_img = transform({'image': lfp_img})['image']
    image = torch.from_numpy(raw_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        depth_f_dA = depth_anything(image)

    depth_f_dA = F.interpolate(depth_f_dA[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth_f_dA = (depth_f_dA - depth_f_dA.min()) / (depth_f_dA.max() - depth_f_dA.min()) * 255.0

    depth_f_dA = depth_f_dA.cpu().numpy().astype(np.uint8)


    print("depth_anyting depth time", time.time() - start_time)

    return depth_f_dA





def plen_any_combine_video(filepath):
    print("plen any video loading", time.time() - start_time)
    # depth map from depthAnything
    # parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    margin_width = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_anything, transform = depth_from_depthAnything(DEVICE, encoding='vitl', cfg=None)

    raw_video = cv2.VideoCapture(filepath)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    print("input video frame_width: ", frame_width)
    print("input video frame_height: ", frame_height)
    print("input video frame rate: ", frame_rate)
    output_width = frame_width * 2 + margin_width

    filename = os.path.basename(filepath)
    outpath, _ = os.path.split(filepath)
    output_combine_path = os.path.join(outpath, filename[:filename.rfind('.')] + '_video_combine.mp4')
    output_depth_path = os.path.join(outpath, filename[:filename.rfind('.')] + '_video_depth.mp4')
    print('output_path', output_combine_path)
    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width//4, frame_height//4))
    out_combine = cv2.VideoWriter(output_combine_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                                  (output_width, frame_height))
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0

        frame = transform({'image': frame})['image']
        frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(frame)

        depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
        combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
        out_combine.write(combined_frame)
    raw_video.release()
    out_combine.release()

def plen_any_depth_video(filepath):
    print("plen any video loading", time.time() - start_time)
    # depth map from depthAnything
    # parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    margin_width = 50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_anything, transform = depth_from_depthAnything(DEVICE, encoding='vitl', cfg=None)

    raw_video = cv2.VideoCapture(filepath)
    frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
    print("input video frame_width: ", frame_width)
    loc_txt_w = frame_width - 300
    loc_txt_h = frame_height - 300
    print("input video frame_height: ", frame_height)
    print("input video frame rate: ", frame_rate)
    # output_width = frame_width * 2 + margin_width

    filename = os.path.basename(filepath)
    outpath, _ = os.path.split(filepath)
    output_depth_path = os.path.join(outpath, filename[:filename.rfind('.')] + '_video_depth.mp4')
    print('output_path', output_depth_path)
    # out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width//4, frame_height//4))
    out_depth = cv2.VideoWriter(output_depth_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate,
                                (frame_width, frame_height))
    while raw_video.isOpened():
        ret, raw_frame = raw_video.read()
        if not ret:
            break

        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB) / 255.0

        frame = transform({'image': frame})['image']
        frame = torch.from_numpy(frame).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(frame)

        depth = F.interpolate(depth[None], (frame_height, frame_width), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)


        depth_color = cv2.putText(depth_color, "%s fps" %str(frame_rate), (loc_txt_w, loc_txt_h),
                                     cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 10, cv2.LINE_AA, False)
        # split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
        # combined_frame = cv2.hconcat([split_region, depth_color,split_region])
        out_depth.write(depth_color)

    raw_video.release()
    out_depth.release()

def replace_color_with_depth(color_img, depth_map):
    """
    color 이미지 (H, W, 3)를 depthmap (H, W 또는 H, W, 1)으로 치환
    :param color_img: np.ndarray, shape (H, W, 3)
    :param depth_map: np.ndarray, shape (H, W) 또는 (H, W, 1)
    :return: np.ndarray, shape (H, W, 1) 또는 (H, W, D)
    """
    if depth_map.ndim == 2:
        # (H, W) -> (H, W, 1)
        depth_map = np.expand_dims(depth_map, axis=-1)

    assert depth_map.shape[:2] == color_img.shape[:2], "해상도가 일치해야 합니다."

    return depth_map

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_depth_surface_colormap_texture(depth_map, scale=0.05, colormap=cv2.COLORMAP_JET):
    """
    컬러맵 기반 텍스처로 깊이맵을 3D surface plot으로 시각화합니다.

    Parameters:
        depth_map (np.ndarray): 단일 채널 깊이맵 (H, W) 또는 컬러맵 이미지 (H, W, 3)
        scale (float): 시각화 속도 개선을 위한 다운샘플 비율
        colormap: OpenCV colormap (예: COLORMAP_JET, COLORMAP_INFERNO)

    Returns:
        None
    """

    # --- Step 1: Depth 정규화 (단일 채널이어야 함) ---
    if depth_map.ndim == 3:
        depth_gray = cv2.cvtColor(depth_map, cv2.COLOR_RGB2GRAY)
    else:
        depth_gray = depth_map.copy()

    depth_gray = depth_gray.astype(np.float32)
    depth_gray -= depth_gray.min()
    if depth_gray.max() != 0:
        depth_gray /= depth_gray.max()

    # --- Step 2: 컬러맵 텍스처 생성 ---
    depth_u8 = (depth_gray * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, colormap)
    texture = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # --- Step 3: 다운샘플링 ---
    depth_small = cv2.resize(depth_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    texture_small = cv2.resize(texture, (depth_small.shape[1], depth_small.shape[0]), interpolation=cv2.INTER_AREA)

    H, W = depth_small.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    # --- Step 4: 3D Surface Plot ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X, Y, depth_small,
        facecolors=texture_small,
        linewidth=0,
        antialiased=False,
        shade=False
    )

    ax.set_title("3D Depth Surface with Colormap Texture")
    ax.set_xlabel("X (width)")
    ax.set_ylabel("Y (height)")
    ax.set_zlabel("Depth")
    plt.tight_layout()
    plt.show()

def prepare_for_psnr(img1, img2):
    # dtype 통일
    if img1.dtype != np.float64:
        img1 = img1.astype(np.float64)
    if img2.dtype != np.float64:
        img2 = img2.astype(np.float64)

    # 정규화 (0~255 → 0~1)
    if img1.max() > 1.0:
        img1 /= 255.0
    if img2.max() > 1.0:
        img2 /= 255.0

    return img1, img2


if __name__ == "__main__":

    import time
    start_time = time.time()
## confirm GPU
    if torch.cuda.is_available():
        print("# of GPU:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("no GPUs")

## initialize plenopticam
    cfg = pcam.cfg.PlenopticamConfig()
    cfg.default_values()

## make sure the config file is loaded
    userdef = parse_args()

    cfg.params[cfg.lfp_path] = userdef.raw_img_path
    cfg.params[cfg.cal_path] = userdef.cal_img_path
    cfg.params[cfg.cal_meta] = userdef.cal_dat_path

    exec_cal = userdef.exec_cal
    fpath, file = os.path.split(cfg.params[cfg.lfp_path])
    fname, ext = os.path.splitext(file)
    sys.path.append(fpath+'/')
    print('DATAHOME : ', fpath+'/')
    RESULT = os.path.join(fpath, fname)
    print("RESULT", RESULT)

    if userdef.TP == "01":
        if not os.path.exists(RESULT):
            os.makedirs(RESULT)

        if not userdef.raw_img_path == None:
            if userdef.raw_img_path.endswith(('tif', 'tiff', 'jpg', 'jpeg','bmp', 'png')):
                spath, lfp_img, wht_img = load_raw_img(image_path=userdef.raw_img_path, cfg=cfg)
                # outdir = os.path.join(os.path.join(fpath, '/'), fname)

                print("spath: ",spath)
                depth_f_dA = plen_any_img(lfp_img, cfg)
            else:
                pass
        else:
            pass

        depth_color = cv2.applyColorMap(depth_f_dA, cv2.COLORMAP_INFERNO)
        img_info(depth_f_dA, fname)
        max_range = depth_f_dA.max()
        d4plen_depthmap = sparse_color_depthmap(depth_f_dA, max_range)

        plt.figure()
        plt.imshow(d4plen_depthmap)
        plt.grid(False)
        plt.title('depth4plen based depthmap, %s' % fname)
        # plt.axis('off')
        save_dir = fpath + '/' + fname
        plt.savefig(save_dir + "/depth4plen_%s.jpg" % fname, bbox_inches='tight', pad_inches=0)
        plt.show()
        cv2.imwrite(save_dir + "/depth4plen_%s.png" % fname, d4plen_depthmap)

        ## 3D surface plot with texture
        lfp_img_f = lfp_img.astype(np.float32) / 255.0
        depth_f = d4plen_depthmap.astype(np.float32) / 255.0

        plot_depth_surface_colormap_texture(lfp_img_f, depth_f, scale=1.0)

        ## depth profile
        if d4plen_depthmap.ndim == 2:
            hd, wd = depth_color.shape
        else:
            hd, wd, _ = depth_color.shape

        # img_info(depth_color, 'depth_color')
        img_avg_width = depth_profile(depth_color, set_w=wd // 2)
        img_avg_height = depth_profile(depth_color, set_h=hd // 2)

        print('img_avg_width.dtype', img_avg_width.dtype)
        print('type(img_avg_width)', type(img_avg_width))
        print('img_avg_width.max()', img_avg_width.max())

        print('img_avg_height.dtype', img_avg_height.dtype)
        print('type(img_avg_height)', type(img_avg_height))
        print('img_avg_height.max()', img_avg_height.max())

        # 그래프 그리기
        plt.figure(figsize=(10, 10))
        plt.plot(img_avg_width, label='depth profile')
        plt.title('depth profile', size=20)
        plt.xlabel('Pixel Position')
        plt.ylabel('relative depth')
        plt.savefig(save_dir + "/depth_profile_width_%s.jpg" % fname, bbox_inches='tight', pad_inches=0)
        plt.show()

        # 그래프 그리기
        plt.figure(figsize=(10, 10))
        plt.plot(img_avg_height, label='depth profile')
        plt.title('depth profile', size=20)
        plt.xlabel('Pixel Position')
        plt.ylabel('relative depth')
        plt.savefig(save_dir + "/depth_profile_height_%s.jpg" % fname, bbox_inches='tight', pad_inches=0)
        plt.show()


    elif userdef.TP == "02":  # multiple plen4depthAny Image & PSNR

        from skimage.io import imread
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr

        psnr_values = []

        ref_img_path = ("./Cert_image/depth4plen_mslee21_52_20250529_102637_1.png")
        ref_img = imread(ref_img_path, cv2.IMREAD_UNCHANGED)

        i = 0
        for root, dirs, files in os.walk(userdef.raw_dir_path):
            mulitiple_result_dir = os.path.join(root,"result")
            if not os.path.exists(mulitiple_result_dir):
                os.makedirs(mulitiple_result_dir)

            for file in files:
                fname, ext = os.path.splitext(file)
                depth_img_path = os.path.join(root, fname)+'.jpg'
                lfp_img = pcam.misc.load_img_file(depth_img_path)
                depth_f_dA = plen_any_img(lfp_img, cfg)

                depth_color = cv2.applyColorMap(depth_f_dA, cv2.COLORMAP_INFERNO)
                img_info(depth_f_dA, fname)
                max_range = depth_f_dA.max()
                d4plen_depthmap = sparse_color_depthmap(depth_f_dA, max_range)

                plt.figure()
                plt.imshow(d4plen_depthmap)
                plt.grid(False)
                plt.title('depth4plen based depthmap, %s' % fname)
                # plt.axis('off')
                plt.savefig(os.path.join(mulitiple_result_dir, "depth4plen_%s.jpg" % fname), bbox_inches='tight', pad_inches=0)
                plt.show()
                cv2.imwrite(os.path.join(mulitiple_result_dir, "depth4plen_%s.png" % fname), d4plen_depthmap)
                cv2.imwrite(os.path.join(mulitiple_result_dir, "depth4plen_%s.tiff" % fname), d4plen_depthmap)
                print("d4plen_depthmap.ndim", d4plen_depthmap.ndim)
                print("d4plen_depthmap.shape", d4plen_depthmap.shape)

            for file in files:
                fname, ext = os.path.splitext(file)
                depth_img_path = os.path.join(mulitiple_result_dir, "depth4plen_%s.png" % fname)
                if not os.path.exists(depth_img_path):
                    continue
                    print("depth_img_path", depth_img_path)
                org_img = imread(depth_img_path, cv2.IMREAD_UNCHANGED)

                ref_img, compare_img = prepare_for_psnr(ref_img, org_img)


                psnr_value = psnr(ref_img, compare_img)
                psnr_values.append(psnr_value)
                i += 1
                print( "%sth image: " % str(i) + "depth4plen_%s.png" % fname + ", psnr :" + str(psnr_value))
            break

        average_psnr = np.mean(psnr_values)
        print(f'Average PSNR: {average_psnr} dB')

        #         img_ref = imread(depth_img_path + "/depth4plen_%s.png" % fname )
        #         plt.figure()
        #         plt.imshow(img_ref)
        #         # plt.imshow(d4plen_depthmap)
        #         # plt.imshow(gray_depthmap / gray_depthmap.max(), interpolation='none',cmap='gray')
        #         plt.grid(False)
        #         plt.title(fname + '_depth4plen based depthmap Image')
        #         plt.show()
        #
        #         psnr = cv2.PSNR(img_org, img_ref)
        #         psnr_values.append(psnr)
        #         i += 1
        #         print( "%sth image: " % str(i) + "depth4plen_%s.png" % fname + ", psnr :" + str(psnr))
        #
        #     break

    # #
    # ########################################
    # ## plen4depthAny Video
    # ########################################
    #
    # vid_fpath, vid_file = os.path.split(userdef.raw_vdo_path)
    # vid_fname, vid_ext = os.path.splitext(vid_file)
    # VID_RESULT = os.path.join(vid_fpath, vid_fname + "_result")
    # if not os.path.exists(VID_RESULT):
    #     os.makedirs(VID_RESULT)
    #
    # output_video_path = os.path.join(VID_RESULT, f"{vid_fname}_resize.mp4")
    # print('userdef.raw_vdo_path', userdef.raw_vdo_path)
    # vid_fpath, vid_file = os.path.split(userdef.raw_vdo_path)
    # vid_fname, vid_ext = os.path.splitext(file)
    # # RESULT = fpath+"/"+fname+"/"
    # # createFolder(RESULT)
    # sys.path.append(vid_fpath+'/')
    # print('Video DATAHOME : ', vid_fpath+'/')
    #
    # VID_RESULT = os.path.join(vid_fpath, vid_fpath)
    # print("Video RESULT", VID_RESULT)
    # if not os.path.exists(VID_RESULT):
    #     os.makedirs(VID_RESULT)
    # #
    # #
    # #
    # #
    # cap = cv2.VideoCapture(userdef.raw_vdo_path)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # if fps == 0 or fps != fps:
    #     fps = 30.0
    #
    # target_width, target_height = 640, 480
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))
    #
    # print(f"[INFO] Writing resized video to {output_video_path} at {fps} FPS")
    #
    # # 프레임 리사이즈 및 저장
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     resized_frame = cv2.resize(frame, (target_width, target_height))
    #     out.write(resized_frame)
    #
    # cap.release()
    # out.release()
    #
    # plen_any_depth_video(userdef.raw_vdo_path)
    # plen_any_combine_video(userdef.raw_vdo_path)