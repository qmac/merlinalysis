import sys
import os
import numpy as np

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg


def average_images(files):
    total = np.zeros((65, 77, 49))
    all_data = []
    n = float(len(files))
    if 'semantic' in files[0]:
        p = 300
    elif 'object' in files[0]:
        p = 422
    else:
        p = 1
    for f in files:
        data = check_niimg(f).get_data()
        # data = np.sqrt(np.abs(data)) * np.sign(data)  # convert back to R
        # data = np.nan_to_num(data)
        # data = np.arctanh(data) # equivalent to Fisher R to Z transformation
        all_data.append(data)

    std = np.std(all_data, axis=0)
    avg = np.mean(all_data, axis=0)
    z_scores = avg / (1e-10 + std)
    # avg = np.tanh(avg)  # convert from Z back to R
    # avg = np.sign(avg) * np.power(avg, 2)  # convert back to R^2
    # avg = 1 - (1 - avg) * ((n - 1) / (n - p - 1))  # adjuted R squared
    return avg, z_scores

if __name__ == '__main__':
    output_file = sys.argv[1]
    image_files = sys.argv[2:]
    avg_data, z_scores = average_images(image_files)
    affine = check_niimg(image_files[0]).affine
    avg_nifti = Nifti1Image(avg_data, affine=affine)
    avg_nifti.to_filename(output_file)
    z_nifti = Nifti1Image(z_scores, affine=affine)
    z_path = os.path.dirname(output_file) + '/z_' + os.path.basename(output_file)
    #z_nifti.to_filename(z_path)
