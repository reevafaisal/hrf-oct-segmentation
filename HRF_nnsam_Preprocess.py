import multiprocessing
import shutil
from multiprocessing import Pool

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes
import numpy as np
import cv2 as cv 


def load_and_covnert_case(input_image: str, input_seg: str, output_image: str, output_seg: str,
                          min_component_size: int = 50):
    seg = io.imread(input_seg)
    image = io.imread(input_image)
    seg = seg.astype(np.uint8)
    print("Unique labels in mask:", np.unique(seg))
    seg = cv.cvtColor(seg, cv.COLOR_GRAY2RGB)
    io.imsave(output_seg, seg, check_contrast=False)
    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    io.imsave(output_image, image, check_contrast=False)
    #shutil.copy(input_image, output_image)


if __name__ == "__main__":
    source = "/home/shared/projects/HRF_Reeva/final_data"

    dataset_name = 'Dataset005_HRF'

    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    imagests = join(nnUNet_raw, dataset_name, 'imagesTs')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    labelsts = join(nnUNet_raw, dataset_name, 'labelsTs')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    train_source = join(source, 'training')
    test_source = join(source, 'testing')

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(train_source, 'output'), join=False, suffix='png')
        num_train = len(valid_ids)
        r = []
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(train_source, 'input', v),
                         join(train_source, 'output', v),
                         join(imagestr, v[:-4] + '_0000.png'),
                         join(labelstr, v),
                         50
                     ),)
                )
            )

        # test set
        valid_ids = subfiles(join(test_source, 'output'), join=False, suffix='.png')
        for v in valid_ids:
            r.append(
                p.starmap_async(
                    load_and_covnert_case,
                    ((
                         join(test_source, 'input', v),
                         join(test_source, 'output', v),
                         join(imagests, v[:-4] + '_0000.png'),
                         join(labelsts, v),
                         50
                     ),)
                )
            )
        _ = [i.get() for i in r]

    generate_dataset_json(join(nnUNet_raw, dataset_name), {0: 'R', 1: 'G', 2: 'B'}, 
                          {'background': 0, 'HRF': 1},
                          num_train, '.png', dataset_name)
