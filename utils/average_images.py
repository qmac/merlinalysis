import sys
import numpy as np

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg


def average_images(files):
    total = np.zeros((65, 77, 49))
    for f in files:
        data = check_niimg(f).get_data()
        data = np.sqrt(np.abs(data)) * np.sign(data)  # convert back to R
        data = np.arctanh(data) # equivalent to Fisher R to Z transformation
        total += data
    avg = total / float(len(files))
    avg = np.tanh(avg)  # convert from Z back to R
    avg = np.sign(avg) * np.power(avg, 2)  # convert back to R^2
    return avg

if __name__ == '__main__':
    output_file = sys.argv[1]
    image_files = sys.argv[2:]
    avg_data = average_images(image_files)
    affine = check_niimg(image_files[0]).affine
    avg_nifti = Nifti1Image(avg_data, affine=affine)
    avg_nifti.to_filename(output_file)
