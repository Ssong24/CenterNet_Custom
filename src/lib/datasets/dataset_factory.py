from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset

from .dataset.etri_undistort import EtriUndistort
from .dataset.etri_distort import EtriDistort

dataset_factory = {
  'etri_undistort': EtriUndistort,
  'etri_distort': EtriDistort
}

_sample_factory = {
  'ctdet': CTDetDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset

