import sys
import numpy as np

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg

from ridge.utils import zscore
from detrend_sgolay import sgolay_filter_volume

TR = 1.5


def detrend_data(Y):
    return sgolay_filter_volume(Y, filtlen=181, degree=3)


def run_preprocessing(image_file, output_file):
    # Load and crop
    img = check_niimg(image_file, ensure_ndim=4)
    img_data = img.get_data()
    img_data = img_data[:, :, :, 17:]  # Initial scanner setup
    img_data = img_data[:, :, :, 27:]  # Starting cartoon
    img_data = img_data[:, :, :, :975]
    print('Data matrix shape: ' + str(img_data.shape))

    img_data = np.reshape(img_data, (245245, img_data.shape[3]))

    # Normalize data
    Y = zscore(img_data).T

    # Detrend data
    Y = detrend_data(Y)
    print('Detrended data matrix: ' + str(Y.shape))

    Y = np.reshape(Y.T, (65, 77, 49, 975))

    r_squared_img = Nifti1Image(Y, affine=img.affine)
    r_squared_img.to_filename(output_file)


if __name__ == '__main__':
    image_file = sys.argv[1]
    output_file = sys.argv[2]
    run_preprocessing(image_file, output_file)
