import numpy as np
import random
from tqdm import tqdm

"""
Code converted from Matlab/C to Python
https://github.com/donggong1/motion-flow-syn
"""

PARAMS_HEAVY_BLUR = {
    "tx_max": 27, "tx_acc_max": 0.3, 
    "ty_max": 27, "ty_acc_max": 0.3, 
    "tz_max":5e-3, "cen_z_shift_max":15, "rot_z_max":np.pi/70,
}

PARAMS_MILD_BLUR = {
    "tx_max": 19, "tx_acc_max": 0.25, 
    "ty_max": 19, "ty_acc_max": 0.25, 
    "tz_max":2e-3, "cen_z_shift_max":5, "rot_z_max":np.pi/120,
}

def sample_x(img, params=PARAMS_MILD_BLUR):
    """
    Sample a motion flow in the X direction
    """
    r = np.random.uniform(low=-params['tx_acc_max'], high=params['tx_acc_max'])
    t = np.random.uniform(low=-params['tx_max'], high=params['tx_max'])

    left = t * (1 - r / 2)

    coordinates = np.arange(1, img.shape[1] + 1).reshape((1, -1))
    coordinates = np.repeat(coordinates, img.shape[0], axis=0)
    
    u, v = coordinates * r * t / img.shape[1] + left, np.zeros(img.shape[:2])
    return np.stack((u, v), axis=0)

def sample_y(img, params=PARAMS_MILD_BLUR):
    """
    Sample a motion flow in the Y direction
    """
    r = np.random.uniform(low=-params['ty_acc_max'], high=params['ty_acc_max'])
    t = np.random.uniform(low=-params['ty_max'], high=params['ty_max'])

    bottom = t * (1 - r / 2)

    coordinates = np.arange(1, img.shape[0] + 1).reshape((-1, 1))
    coordinates = np.repeat(coordinates, img.shape[1], axis=1)

    u, v = np.zeros(img.shape[:2]), coordinates * r * t / img.shape[0] + bottom
    return np.stack((u,v), axis=0)

def sample_z(img, params=PARAMS_MILD_BLUR):
    """
    Sample a motion flow in the Z direction (both translation and rotational)
    """
    x = np.arange(1, img.shape[1] + 1)
    y = np.arange(1, img.shape[0] + 1)
    xx, yy = np.meshgrid(x, y)

    t = np.random.uniform(low=-params["tz_max"], high=params["tz_max"])
    p = np.array(img.shape[:2]) / 2 + np.random.uniform(low=-params["cen_z_shift_max"], high=params["cen_z_shift_max"], size=2)
    omega = np.random.uniform(low=-params['rot_z_max'], high=params['rot_z_max'])

    d = np.sqrt(np.square(xx - p[1]) + np.square(yy - p[0]))
    rot = np.tan(omega) * d

    tmph = 1 + img.shape[0] - yy - p[0]
    tmpw = xx - p[1]

    alpha = np.arctan2(tmph, tmpw)

    u = np.sin(alpha - np.pi/2) * rot + t * np.power(d, 1.5) * np.sin(alpha)
    v = np.cos(alpha - np.pi/2) * rot + t * np.power(d, 1.5) * np.cos(alpha)
    
    return np.stack((u, v), axis=0)

def sample_motion_flow(img, params=PARAMS_MILD_BLUR):
    """
    Sample a motion flow with comprised of MFs with all three directions
    """
    return sample_x(img, params=params) + sample_y(img, params=params) + sample_z(img, params=params)

def blur_image(img, motion_flow):
    """
    Blur an image according to a motion flow
    """
    output_image = np.zeros_like(img).astype(float)

    h_img, w_img = img.shape[:2]

    mag = np.linalg.norm(motion_flow, axis=0)
    phi = np.arctan2(motion_flow[1], np.where(motion_flow[0] == 0, 1e-16, motion_flow[0]))
    half = (mag - 1) / 2
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    xsign = np.sign(cosphi)

    linewdt, sumKernel = 1, 0

    tmp = half * cosphi + linewdt * xsign - mag * 2.2204e-16
    sx = np.floor(np.abs(tmp)).astype(int)

    tmp = half * sinphi + linewdt - mag * 2.2204e-16
    sy = np.floor(np.abs(tmp)).astype(int)

    for j in range(h_img):
        for i in range(w_img):
            sum_kernel = 0
            for l in range(-sy[j,i], sy[j,i] + 1):
                for k in range(-sx[j,i], sx[j,i] + 1):
                    dist2line = l * cosphi[j,i] + k * sinphi[j,i]
                    dist2cent = np.sqrt(l * l + k * k)

                    if abs(dist2line) <= linewdt and dist2cent >= half[j,i]:
                        x2lastpix = half[j,i] - abs((k + dist2line * sinphi[j,i])/cosphi[j,i])
                        dist2line = np.sqrt(dist2line * dist2line + x2lastpix * x2lastpix)

                    dist2line = linewdt + 2.2204e-16 - abs(dist2line)
                    dist2line = 0 if dist2line < 0 else dist2line

                    off_x = k if (i + k < w_img and i + k >= 0) else (-i if i + k < 0 else w_img - 1 - i)
                    off_y = l if (j + l < h_img and j + l >= 0) else (-j if j + l < 0 else h_img - 1 - j)

                    output_image[j, i] += dist2line * img[j + off_y, i + off_x]
                    sum_kernel += dist2line
            if sum_kernel > 0:
                output_image[j, i] /= sum_kernel

    return output_image