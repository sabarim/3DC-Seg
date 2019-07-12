import glob
import os
import numpy as np
import time
import multiprocessing as mp
from PIL import Image
from scipy.misc import imresize

from datasets.DAVIS import DAVIS
from utils.Resize import ResizeMode, resize


class YoutubeVOSDataset(DAVIS):
  def __init__(self, root, imset='train', is_train=False,
               random_instance=False, crop_size=None,temporal_window=3, min_temporal_gap=2,
               max_temporal_gap=8, resize_mode=ResizeMode.FIXED_SIZE):
    # maintain a list of files for each video
    self.video_frames = {}
    super(YoutubeVOSDataset, self).__init__(root, imset=imset, is_train=is_train, random_instance=random_instance,
                                            crop_size=crop_size, temporal_window=temporal_window,
                                            min_temporal_gap=min_temporal_gap, max_temporal_gap=max_temporal_gap,
                                            resize_mode=resize_mode)

  def set_paths(self, imset, resolution, root):
    if imset == 'train':
      self.mask_dir = os.path.join(root, imset, 'CleanedAnnotations')
    else:
      self.mask_dir = os.path.join(root, imset, 'Annotations')
    self.image_dir = os.path.join(root, imset, 'JPEGImages')
    _imset_dir = os.path.join(root, imset)
    _imset_f = glob.glob(_imset_dir + "/JPEGImages/*")
    return _imset_f

  def create_img_list(self, _imset_f):
    start = time.time()
    pool = mp.Pool(10)
    results = [pool.apply(self.video_list, args=(line,)) for line in _imset_f]
    # for line in _imset_f:
    #   _video = line.split("/")[-1]
    #   self.videos.append(_video)
    #   self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
    #   mask_files = glob.glob(os.path.join(self.mask_dir, _video, '*.png'))
    #   if len(mask_files) == 0:
    #     mask_files = glob.glob(os.path.join(self.mask_dir.replace("CleanedAnnotations", "Annotations"), _video, '*.png'))
    #   n_objects = [np.max(np.array(Image.open(f).convert("P")))
    #                for f in mask_files]
    #   self.num_objects[_video] = np.max(n_objects)
    #   self.shape[_video] = np.shape(np.array(Image.open(mask_files[0]).convert("P")))
    #   self.img_list += list(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
    results = np.array(results)
    self.videos = results[:, 0]
    self.img_list = np.concatenate(results[:, 1])

    video_frames = np.array([results[:, 0], results[:, 1]]).transpose()
    self.video_frames.update({video_frames[i][0]:video_frames[i][1] for i in range(len(video_frames))})
    shapes = np.array([results[:, 0], results[:, -1]]).transpose()
    self.shape.update({shapes[i][0]:shapes[i][1] for i in range(len(shapes))})
    num_objects = np.array([results[:, 0], results[:, 2]]).transpose()
    self.num_objects.update({num_objects[i][0]: num_objects[i][1] for i in range(len(num_objects))})
    num_frames = np.array([results[:, 0], results[:, -2]]).transpose()
    self.num_frames.update({num_frames[i][0]: num_frames[i][1] for i in range(len(num_frames))})

    print("Time to create image list".format(time.time() - start))

  def video_list(self, video_path):
    start = time.time()
    _video = video_path.split("/")[-1]
    mask_files = glob.glob(os.path.join(self.mask_dir, _video, '*.png'))
    if len(mask_files) == 0:
      mask_files = glob.glob(os.path.join(self.mask_dir.replace("CleanedAnnotations", "Annotations"), _video, '*.png'))
    n_objects = [np.max(np.array(Image.open(f).convert("P")))
                 for f in mask_files]
    img_list = list(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
    num_frames = len(img_list)
    max_objects = np.max(n_objects)
    # max_objects = 1
    shape = np.shape(np.array(Image.open(mask_files[0]).convert("P")))

    print("Time to create image list for video {} is {}".format(_video, time.time() - start))
    return _video, img_list, max_objects, num_frames, shape

  def read_frame(self, shape, video, f, instance_id=None):
    # use a blend of both full random instance as well as the full object
    img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
    raw_frames = np.array(Image.open(img_file).convert('RGB')) / 255.
    try:
      mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  # allways return first frame mask
      # cleaned annotations do not exist for all sequences. Use normal annotations for these frames
      if os.path.exists(mask_file):
        raw_mask = imresize(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8), shape,
                            interp="nearest")
      else:
       raw_mask = imresize(np.array(Image.open(mask_file.replace("CleanedAnnotations", "Annotations")).convert('P'), dtype=np.uint8), shape,
                            interp="nearest")
    except:
      mask_file = os.path.join(self.mask_dir, video, '00000.png')
      raw_mask = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)

    raw_masks = (raw_mask == instance_id).astype(np.uint8) if instance_id is not None else raw_mask
    tensors_resized = resize({"image": raw_frames, "mask": raw_masks}, self.resize_mode, shape)

    return tensors_resized["image"] / 255.0, tensors_resized["mask"]

  def get_support_indices(self, index, sequence):
    # in youtube-vos index does not correspond to the file index
    # i = int(os.path.splitext(os.path.basename(self.img_list[file_index]))[0])
    sample_list = [int(os.path.splitext(os.path.basename(f))[0]) for f in self.video_frames[sequence]
                   if int(os.path.splitext(os.path.basename(f))[0]) < index]
    sample_list.sort()

    if len(sample_list) == 0:
      support_indices = np.repeat([index], self.temporal_window + 2)
    elif (len(sample_list) - (self.temporal_window + 2)) < 0:
      support_indices = np.repeat([index], abs(len(sample_list) - (self.temporal_window + 2)))
      support_indices = np.append(support_indices, sample_list[-1])
      support_indices = np.append(support_indices, sample_list[:-1])
    else:
      support_indices = np.random.choice(np.array(sample_list)[:-1], self.temporal_window, replace=False)
      support_indices = np.append(support_indices, np.array([index, sample_list[-1]]))

    support_indices.sort()
    return support_indices.astype(np.int)