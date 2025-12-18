#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "info@christopherhahne.de"
__license__ = """
    Copyright (c) 2019 Christopher Hahne <info@christopherhahne.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from plenopticam_etri_09.lfp_extractor import LfpViewpoints
from plenopticam_etri_09.misc import PlenopticamError

from multiprocessing import Pool, cpu_count



# class LfpRearranger_1(LfpViewpoints):
#     # 클래스 정의 및 기타 메서드들...
#     def parallel_process(self, lfp_img_align, size_pitch, k):
#         cpu_cores = cpu_count()
#         pool = Pool(cpu_cores)
#         step = lfp_img_align.shape[0] // cpu_cores
#         ranges = [(lfp_img_align, size_pitch, k, i, min(i + step, lfp_img_align.shape[0])) for i in
#                   range(0, lfp_img_align.shape[0], step)]
#
#         results = pool.map(self.process_image_slice, ranges)
#         pool.close()
#         pool.join()
#
#         vp_img_arr = np.vstack(results)
#         return vp_img_arr
#
#     def process_image_slice(args):
#         lfp_img_align, size_pitch, k, start, end = args
#         vp_img_arr_slice = np.zeros_like(lfp_img_align)
#
#         for j in range(start, end):
#             for i in range(size_pitch):
#                 vp_img_arr_slice[j, i, ...] = lfp_img_align[j:j + size_pitch * k[0]:size_pitch,
#                                               i:i + size_pitch * k[1]:size_pitch, :]
#         return vp_img_arr_slice

import torch
import torch.nn as nn


class Conv2DOperation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2DOperation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class LfpRearranger(LfpViewpoints):

    def __init__(self, lfp_img_align=None, *args, **kwargs):
        super(LfpRearranger, self).__init__(*args, **kwargs)

        self._lfp_img_align = lfp_img_align if lfp_img_align is not None else None
        self._dtype = self._lfp_img_align.dtype if self._lfp_img_align is not None else self._vp_img_arr.dtype


        
    def _init_vp_img_arr(self):
        """ initialize viewpoint output image array """

        if len(self._lfp_img_align.shape) == 3:
            m, n, p = self._lfp_img_align.shape
        elif len(self._lfp_img_align.shape) == 2:
            m, n, p = self._lfp_img_align.shape[:2] + (1,)
        else:
            raise PlenopticamError('Dimensions %s of provided light-field not supported', self._lfp_img_align.shape,
                                   cfg=self.cfg, sta=self.sta)
        # 23.10.21. (RPC, Error)    
        self._vp_img_arr = np.zeros([int(self._size_pitch), int(self._size_pitch),
                                     int(m//self._size_pitch), int(n//self._size_pitch), p], dtype=self._dtype)
        
#        self._vp_img_arr = np.zeros([int(self._size_pitch), int(self._size_pitch),
#                                      int(m//self._size_pitch), int(n//self._size_pitch), p], dtype=self._dtype)
# could not broadcast input array from shape (90,90,3) into shape (91,91,3), int(n//self._size_pitch)-8

    def _init_lfp_img_align(self):
        """ initialize micro image output image array """

        if len(self._vp_img_arr.shape) == 5:
            m, n, p = self._vp_img_arr.shape[2:]
        elif len(self._vp_img_arr.shape) == 4:
            m, n, p = self._vp_img_arr.shape[2:] + (1,)
        else:
            raise PlenopticamError('Dimensions %s of provided light-field not supported', self._vp_img_arr.shape,
                                   cfg=self.cfg, sta=self.sta)

        # create empty array
        m *= self._vp_img_arr.shape[0]
        n *= self._vp_img_arr.shape[1]
        self._lfp_img_align = np.zeros([m, n, p], dtype=self._dtype)
        
        # update angular resolution parameter
        self._size_pitch = self._vp_img_arr.shape[0] if self._vp_img_arr.shape[0] == self._vp_img_arr.shape[1] else float('inf')
        

    def main(self):

        # check interrupt status
        if self.sta.interrupt:
            return False

        # rearrange light-field to viewpoint representation
        self.compose_viewpoints()

    def compose_viewpoints(self):
        """
        Conversion from aligned micro image array to viewpoint array representation. The fundamentals behind the
        4-D light-field transfer were derived by Levoy and Hanrahans in their paper 'Light Field Rendering' in Fig. 6.
        """

        # print status
        self.sta.status_msg('Viewpoint composition', self.cfg.params[self.cfg.opt_prnt])
        self.sta.progress(None, self.cfg.params[self.cfg.opt_prnt])

        # initialize basic light-field parameters
        self._init_vp_img_arr()
        
        # test
        
        m, n, p = self._lfp_img_align.shape

        k = [int(m//self._size_pitch), int(n//self._size_pitch)]

#         k = int(m//self._size_pitch-1) * int(self._size_pitch)

#         self._lfp_img_align = self._lfp_img_align[0:k, 0:k, :]

        ####### 23.11.30 수정
        import time
        stime_rearranger = time.time()
        for j in range(self._size_pitch):
            for i in range(self._size_pitch):
                # check interrupt status
                if self.sta.interrupt:
                    return False

                # extract viewpoint by pixel rearrangement
                self._vp_img_arr[j, i, ...] = self._lfp_img_align[j:j+self._size_pitch*k[0]:self._size_pitch, i:i+self._size_pitch*k[1]:self._size_pitch, :]

        # import matplotlib.pyplot as plt
        # print("self._vp_img_arr.shape", self._vp_img_arr.shape)
        # plt.figure()
        # plt.imshow(self._vp_img_arr[0,0,:,:,:] / self._vp_img_arr[0,0,:,:,:].max(), interpolation='none')
        # plt.grid(False)
        # plt.title('Conv2DOperation Image')
        # plt.axis('off')  # 축을 숨깁니다 (선택 사항)
        # # plt.savefig(RESULT + "LfpCropper.%s.png" % fname, bbox_inches='tight', pad_inches=0)
        # plt.show()

        # ##### 병렬처리


        # tensor = torch.from_numpy(self._lfp_img_align).float()
        # tensor = tensor.permute(2,0,1)
        # tensor_3channels = tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        # conv_layer = Conv2DOperation(3, 51, 1, 51, 0)
        # self._vp_img_arr = conv_layer(tensor_3channels).detach().numpy()
        # self._vp_img_arr = self._vp_img_arr.transpose(1, 2, 3, 0)
        #
        # import matplotlib.pyplot as plt
        # print("self._vp_img_arr.shape", self._vp_img_arr.shape)
        # plt.figure()
        # plt.imshow(self._vp_img_arr[0,:,:,:] / self._vp_img_arr[0,:,:,:].max(), interpolation='none')
        # plt.grid(False)
        # plt.title('Conv2DOperation Image')
        # plt.axis('off')  # 축을 숨깁니다 (선택 사항)
        # # plt.savefig(RESULT + "LfpCropper.%s.png" % fname, bbox_inches='tight', pad_inches=0)
        # plt.show()



        # import numpy as np
        # from multiprocessing import Pool, cpu_count
        #
        # # 전역 함수로 변경
        #
        #
        # rearranger = LfpRearranger_1()
        # self.vp_img_arr = rearranger.parallel_process(self._lfp_img_align, self._size_pitch, k)

        ##############

        # print status
        # percentage = (j * self._size_pitch + i + 1) / self._size_pitch ** 2
        # self.sta.progress(percentage*100, self.cfg.params[self.cfg.opt_prnt])


        return True

    def decompose_viewpoints(self):
        """
        Conversion from viewpoint image array to aligned micro image array representation. The fundamentals behind the
        4-D light-field transfer were derived by Levoy and Hanrahans in their paper 'Light Field Rendering' in Fig. 6.
        """

        # print status
        self.sta.status_msg('Viewpoint decomposition', self.cfg.params[self.cfg.opt_prnt])
        self.sta.progress(None, self.cfg.params[self.cfg.opt_prnt])

        # initialize basic light-field parameters
        self._init_lfp_img_align()

        # rearrange light field to multi-view image representation
        for j in range(self._size_pitch):
            for i in range(self._size_pitch):

                # check interrupt status
                if self.sta.interrupt:
                    return False

                # extract viewpoint by pixel rearrangement
                self._lfp_img_align[j::self._size_pitch, i::self._size_pitch, :] = self._vp_img_arr[j, i, :, :, :]

                # print status
                percentage = (j * self._size_pitch + i ) / self._size_pitch ** 2
                self.sta.progress(percentage*100, self.cfg.params[self.cfg.opt_prnt])

        return True
