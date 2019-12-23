import numpy as np
from random import shuffle
from PIL import Image
from torch.utils.data import Dataset
from utils.Resize import resize, ResizeMode

NUM_CLASSES = 2


class MapillaryBaseDataset(Dataset):
  def __init__(self, root, is_train, id_divisor, cat_ids_to_use, crop_size, resize_mode, min_size =0, name='mapillary'):
    # note: in case of mapillary the min sizes are always based on the sizes in quarter resolution!
    self.data_dir = root
    self.validation_set_size = -1
    self.is_train = is_train
    self.min_size = min_size
    self._cat_ids_to_use = cat_ids_to_use
    self._id_divisor = id_divisor
    self.name = name
    self.crop_size = crop_size
    self.resize_mode = ResizeMode(resize_mode)
    self.imgs, self.anns = self.read_inputfile_lists()

  def read_inputfile_lists(self):
    data_list = "datasets/mapillary/training.txt" if self.is_train else "datasets/mapillary/validation.txt"
    print("{} ({}): using data_dir:".format(self.name, self.is_train), self.data_dir)
    imgs_ans = []
    with open(data_list) as f:
      for l in f:
        im, an, *im_ids_and_sizes = l.strip().split()
        im = self.data_dir + im
        an = self.data_dir + an
        ids_to_use = []
        for id_and_size in im_ids_and_sizes:
          id_ = id_and_size.split(":")[0]
          size_ = int(id_and_size.split(":")[1])
          if self.is_train and size_ < self.min_size:
            continue
          cat_id = int(id_) // self._id_divisor
          if self._cat_ids_to_use is not None and cat_id not in self._cat_ids_to_use:
            continue
          ids_to_use += [id_]
        if len(ids_to_use) > 0:
          imgs_ans.append((im, an + ":" + str(ids_to_use)))
    if self.is_train:
      shuffle(imgs_ans)
    elif self.validation_set_size != -1:
      imgs_ans = imgs_ans[:self.validation_set_size]
    imgs = [x[0] for x in imgs_ans]
    ans = [x[1] for x in imgs_ans]
    return imgs, ans

  def load_image(self, img_filename):
    img = np.array(Image.open(img_filename).convert('RGB'))
    return img

  def load_annotation(self, annotation_filename):
    annotation_filename_without_id = annotation_filename.split(':')[0]
    ann = np.array(Image.open(annotation_filename_without_id), dtype=np.uint16)
    ann = self.postproc_annotation(annotation_filename, ann)
    return ann

  def postproc_annotation(self, ann_filename, ann):
    # get id list representing valid objects
    ids = eval(ann_filename.split(':')[1])
    ann_postproc = np.zeros_like(ann)
    for id_ in  ids:
      # cat ids as mask ids
      ann_postproc[ann == int(id_)] = int(id_)

    return ann_postproc

  def __len__(self):
    return len(self.anns)

  def __getitem__(self, item):
    pass


