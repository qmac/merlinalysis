''' Averages Nifti images. '''
import sys
import numpy as np

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg


def average_images(files):
    all_data = []
    for f in files:
        data = check_niimg(f).get_data()
        all_data.append(data)

    std = np.std(all_data, axis=0)
    avg = np.mean(all_data, axis=0)

    # I don't think this is right
    t_scores = avg / ((1e-10 + std) / np.sqrt(float(len(files))))

    return avg, t_scores


if __name__ == '__main__':
    output_file = sys.argv[1]
    image_files = sys.argv[2:]
    avg_data, t_scores = average_images(image_files)
    affine = check_niimg(image_files[0]).affine
    avg_nifti = Nifti1Image(avg_data, affine=affine)
    avg_nifti.to_filename(output_file)
