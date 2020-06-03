from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset

from .dataset.etri import Etri
from .dataset.etri_distortion import EtriDistortion

dataset_factory = {
  'etri': Etri,
  'etri_distort': EtriDistortion
}

_sample_factory = {
  'ctdet': CTDetDataset,
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset

