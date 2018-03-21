from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from PIL import Image
import time
import argparse
import pyflow

parser = argparse.ArgumentParser(
    description='Generate optical flows from video frames using python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '--data_dir', default='../data/NewVideos/jpegs_256/', type=str, 
    metavar='PATH', help='path to input video frames')
parser.add_argument(
    '--u_dir', default='../data/NewVideos/tvl1_flow/u/', type=str, 
    metavar='PATH', help='path to generated u flows')
parser.add_argument(
    '--v_dir', default='../data/NewVideos/tvl1_flow/v/', type=str,
    metavar='PATH', help='path to generated v flows'
)

arg = parser.parse_args()

if not os.path.exists(arg.u_dir) and not os.path.exists(arg.v_dir):
    os.makedirs(arg.u_dir)
    os.makedirs(arg.v_dir)
# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

for subdir in sorted(os.listdir(arg.data_dir)):
    path_to_subdir = os.path.join(arg.data_dir, subdir)
    list_of_images = sorted(os.listdir(path_to_subdir))
    if not os.path.exists(os.path.join(arg.u_dir, subdir)) and not os.path.exists(os.path.join(arg.v_dir, subdir)):
        os.makedirs(os.path.join(arg.u_dir, subdir))
        os.makedirs(os.path.join(arg.v_dir, subdir))
    start = time.time()
    for i in range(len(list_of_images)):
        try:
            im1 = np.array(Image.open(os.path.join(path_to_subdir, list_of_images[i])))
            im2 = np.array(Image.open(os.path.join(path_to_subdir, list_of_images[i+1])))
            im1 = im1.astype(float) / 255.
            im2 = im2.astype(float) / 255.
            s = time.time()
            u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, 
                                                nOuterFPIterations, nInnerFPIterations,
                                                nSORIterations, colType)
            e = time.time()
            print('Time taken: %.2f seconds for image of size (%d, %d, %d)' % (
                e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
            
            # Rescale and save u and v
            rescaled_u = (255.0 / u.max() * (u - u.min())).astype(np.uint8)
            rescaled_v = (255.0 / v.max() * (v - v.min())).astype(np.uint8)
            u_im = Image.fromarray(rescaled_u)
            path_to_u_im = os.path.join(os.path.join(arg.u_dir, subdir), list_of_images[i])
            u_im.save(path_to_u_im)
            v_im = Image.fromarray(rescaled_v)
            path_to_v_im = os.path.join(os.path.join(arg.v_dir, subdir), list_of_images[i])
            v_im.save(path_to_v_im)
        except IndexError:
            pass
    end = time.time()
    print('Time taken to generate optical flows from video {}'.format(start - end))