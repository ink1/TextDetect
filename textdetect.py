#!/usr/bin/env python
import time
import os
import sys
import numpy as np
from keras.models import load_model
import cv2
import argparse

# Pillow image size:
#     2-tuple (width, height)     X, Y
# Pillow image converted to np array has shape:
#     2-tuple (height, width)     Y, X
#     3-tuple (height, width, 3)
#
# OpenCV image is np array with shape:
#     2-tuple (height, width)     Y, X
#     3-tuple (height, width, 3)  Y, X

parser = argparse.ArgumentParser(description='Text detection in medical images.')
parser.add_argument('image_path', metavar='image_path', type=str,
                    help='path to the image.')

parser.add_argument('-m', '--model', dest='model', type=str, default='model.h5',
                    help='model to use for text detection.')

parser.add_argument('-t', '--threshold', dest='thr_level', type=int, default=248, # 127
                    help='detection threshold from 0 to 255. Default 240.')

parser.add_argument('--shift', type=int, default=2,
                    help='sliding window shift can takes values 2, 4, 8, 16, 32. Default 2.')

parser.add_argument('--max_scale', type=int, default=5,
                    help='number of passes with different scales. Default 5.')

parser.add_argument('-k', '--kernel_size', dest='kernel_size', type=int, default=5,
                    help='kernel size for noise filtering: 0, 3, 5, 7. Default 5.')

parser.add_argument('-f', '--fill', action='store_true',
                    help='fill with paint detected areas.')

parser.add_argument('-i', '--intermediate_scales', action='store_true',
                    help='save results of intermediate scales.')

args = parser.parse_args()

if not os.path.isfile(args.image_path):
    print(args.image_path, 'does not exist')
    exit()
basename = os.path.basename(args.image_path)

if not os.path.isfile(args.model):
    print(args.model, 'could not be found')
    exit()

shift = args.shift

if not (shift == 2 or shift == 4 or
        shift == 8 or shift == 16 or shift == 32):
    print('invalid value of shift')
    exit()

kernel_size = args.kernel_size

if not (kernel_size == 0 or kernel_size == 3 or
        kernel_size == 5 or kernel_size == 7):
    print('invalid value of shift')
    exit()

thr_level = args.thr_level

if thr_level < 0 or thr_level > 255:
    print('invalid value of kernel size')
    exit()

max_scale = args.max_scale

if max_scale < 0 or max_scale > 10:
    print('invalid value of max scale')
    exit()

if args.fill:
    fill = -1
else:
    fill = 1

#=============================================================================

# all images are uint8 unless explicitly converted; Blue, Green, Red order
#
# optional 2nd arg for cv2.imread()
#  The default flag is cv2.IMREAD_COLOR
# cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected.
# cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
# cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel

original_image = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
print('original image shape', original_image.shape)
original_image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

aggregate_mask = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)

#img = np.uint8(np.around(img, decimals=0))
#=============================================================================
# built-in settings

sz = 32
batch_size = 256

#=============================================================================
# loop over different scales

for scale in range(max_scale):
    # scales: 1=2^0, 2^(i*(0.5))
    print('scale', scale)
    if scale != 0:
        sfactor = 2**(scale*0.5)
        rescaled_size = (int(gray_image.shape[0]/sfactor), int(gray_image.shape[1]/sfactor))
        img = cv2.resize(gray_image, (rescaled_size[1], rescaled_size[0]), interpolation = cv2.INTER_NEAREST)
    else:
        sfactor = 1.0
        rescaled_size = (gray_image.shape[0], gray_image.shape[1])
        img = np.copy(gray_image)

    if img.shape[0] < 2*sz or img.shape[1] < 2*sz:
        print('scale too small', scale)
        break

    # crop to size which is multiple of shift; assume sz is also a multiple
    newsize = ((img.shape[0]//shift)*shift, (img.shape[1]//shift)*shift)
    img = img[0:newsize[0], 0:newsize[1]]
    print('new img shape', img.shape)

    # pad with black
    border = (sz - shift)//2  # 12 for shift=8
    img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT, value=0)

    ny = newsize[0]//shift
    nx = newsize[1]//shift
    print('target prob sz:', ny, nx)

    # padded size
    ysz = img.shape[0]
    xsz = img.shape[1]
    print('padded size:', ysz, xsz)

    img = np.asarray(img, dtype=np.float32)
    # normalise image
    img = img*(2./255.) - 1.0

    # find the size of input array padded to be multiple of batch_size
    if nx*ny == (nx*ny//batch_size)*batch_size:
        data_size = nx*ny
    else:
        data_size = (nx*ny//batch_size + 1)*batch_size
    print('nx*ny', nx*ny, 'data_size', data_size)

    xdata = np.zeros((data_size, sz, sz, 1), dtype=np.float32)
    ydata = np.zeros((data_size), dtype=np.float32)

    counter = 0
    for i in range(ny):        # over columns Y
        for j in range(nx):    # over rows X
            xdata[counter, :, :, 0] = img[i*shift:i*shift+sz, j*shift:j*shift+sz]
            counter += 1

    model = load_model(args.model)

    ydata = model.predict(xdata, batch_size=batch_size)

    # method 1 - prob of background
    img = ydata[:nx*ny, 0]
    img = (1.-img)
    # method 2 - prob (confidence) of chars
    #img = np.amax(ydata[:, 1:], axis=1)

    img = img.reshape(ny, nx)
    img = img*255.
    img = np.uint8(np.clip(img, 0, 255))

    #img = cv2.resize(img, (org.shape[1], org.shape[0]), interpolation = cv2.INTER_NEAREST)
    #cv2.imwrite('img.' + f + '.png', img)

    ret, thresh = cv2.threshold(img, args.thr_level, 255, 0)

    ## manipulate shape
    ## erosion: pixel in the original image (either 1 or 0) will be considered 1 only
    ## if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
    ## kernel = np.ones((3,3),np.uint8)
    ## erosion = cv2.erode(img,kernel,iterations = 1)
    ## opening: erosion followed by dilation
    ## opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    if kernel_size > 0:
        mask = np.uint8(np.clip(thresh, 0, 1))
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        #kernel = np.ones((3,3),np.uint8)
        #kernel[0,0] = 0; kernel[0,2] = 0; kernel[2,0] = 0; kernel[2,2] = 0
        thresh = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        thresh = thresh*255

    # cv.INTER_NEAREST
    # cv.INTER_LINEAR  zoom
    # cv.INTER_CUBIC
    # cv.INTER_AREA for shrinking
    # cv.INTER_LANCZOS4 Lanczos interpolation over 8x8 neighborhood

    # resize back including scale change
    #                               width, height
    thresh = cv2.resize(thresh, (int(newsize[1]*sfactor), int(newsize[0]*sfactor)), interpolation = cv2.INTER_NEAREST)

    # find contours
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # save intermediate masks
    if args.intermediate_scales:
        # draw contours (green) on (section of) original image, must be uint
        # args: image to draw on, contours list (-1 means all contours), colour, thickness in pixels (-1 to fill)
        image = np.copy(original_image)
        if len(original_image.shape) == 2:
            image[:newsize[0], :newsize[1]] = cv2.drawContours(image[:newsize[0], :newsize[1]], contours, -1, 255, fill)
        else:
            image[:newsize[0], :newsize[1], :] = cv2.drawContours(image[:newsize[0], :newsize[1], :], contours, -1, (0, 255, 0), fill)

        fname = '%s.t%03d.sc%d.sh%02d.k%d.png' % (basename[:-4], thr_level, scale, shift, kernel_size)
        #basename[:-4] + '.t' + str(thr_level) + '.sc' + str(scale) +'.sh' + str(shift) + '.png'
        cv2.imwrite(fname, image)

    # aggregate masks
    mask = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
    mask[:newsize[0], :newsize[1]] = cv2.drawContours(mask[:newsize[0], :newsize[1]], contours, -1, 255, fill)
    aggregate_mask = np.where(mask > 0, mask, aggregate_mask)


image = np.copy(original_image)

if len(original_image.shape) == 2:
    image = np.where(aggregate_mask > 0, 255, image)
else:
    image[:,:,0] = np.where(aggregate_mask > 0,   0, image[:,:,0])
    image[:,:,1] = np.where(aggregate_mask > 0, 255, image[:,:,1])
    image[:,:,2] = np.where(aggregate_mask > 0,   0, image[:,:,2])

fname = '%s.t%03d.sh%02d.k%d.png' % (basename[:-4], thr_level, shift, kernel_size)
#basename[:-4] + '.t' + str(thr_level) + '.sc' + str(scale) +'.sh' + str(shift) + '.png'
cv2.imwrite(fname, image)