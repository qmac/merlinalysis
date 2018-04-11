import sys

from nibabel import Nifti1Image
from nilearn._utils.niimg_conversions import check_niimg


def variance_partition(d1, d2, comb):
    d1 = np.maximum(d1, 0)
    d2 = np.maximum(d2, 0)
    comb = np.maxmimum(comb, 0)
    only_d1 = comb - d2
    only_d2 = comb - d1
    intersection = d1 - only_d1
    return only_d1, only_d2, intersection


if __name__ == '__main__':
    img = check_niimg(sys.argv[1])
    data1 = img.get_data()
    data2 = check_niimg(sys.argv[2]).get_data()
    data_combined = check_niimg(sys.argv[3]).get_data()
    ex_data1, ex_data2, intersection = variance_partition(data1, data2, data_combined)
    ex_data1_img = Nifti1Image(ex_data1, affine=img.affine)
    ex_data1_img.to_filename(sys.argv[4])
    ex_data2_img = Nifti1Image(ex_data2, affine=img.affine)
    ex_data2_img.to_filename(sys.argv[5])
    intersection_img = Nifti1Image(intersection, affine=img.affine)
    intersection_img.to_filename(sys.argv[6])
