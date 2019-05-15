import os
import glob
import pickle
import numpy as np
import multiprocessing as mp

from util import select_top_predictions, save_mask

CONF_THRESH=0.5
IOU_THRESH = 0.35
maskrcnn_data_dir = "/globalwork/mahadevan/mywork/data/training/pytorch/forwarded/maskrcnn/"
warped_data_dir = "results/maskrcnn_warped"
davis_data_dir = '/globalwork/data/DAVIS-Unsupervised/DAVIS/'
flow_dir = "/globalwork/data/DAVIS-Unsupervised/DAVIS/flo/"
out_dir = "results/eval_maskrcnn_warp/"


def get_iou(gt, pred):
  i = np.logical_and(pred > 0, gt > 0).sum()
  u = np.logical_or(pred > 0, gt > 0).sum()
  if u == 0:
    iou = 1.0
  else:
    iou = i / u
  return iou


def create_object_id_mapping(ref_obj, warped_proposals):
  best_iou = 0
  target_id = -1
  for obj_id in warped_proposals.keys():
    iou = get_iou(ref_obj.astype(np.uint8), warped_proposals[obj_id]['mask'][0, 0].data.cpu().numpy().astype(np.uint8))
    if iou > best_iou and iou > IOU_THRESH:
      best_iou = iou
      target_id = obj_id

  return target_id, best_iou


def save_tracklets(proposals, warped_proposals, out_folder, f):
  """
  
  :param proposals: BoxList 
  :param warped_proposals: dict - contains 'n' top predictions orgnanised as dict
                            dict[i] = {'mask':<nd array with binary mask>, 
                                       'score':<score of the prediction before warp>}
  :param out_folder: 
  :param f: 
  :return: 
  """
  # top_predictions = select_top_predictions(proposals, CONF_THRESH)
  if hasattr(proposals, "get_field"):
    shape = proposals.get_field("mask").shape[2:]
  else:
    print("proposal length", len(list(proposals.values())))
    shape = list(proposals.values())[0]['mask'].shape[2:]
  output_mask = np.zeros(shape)
  result_dict = {}
  ids_chosen = []

  proposal_ids = range(len(proposals.get_field('scores'))) if hasattr(proposals, 'get_field') else \
    proposals.keys()
  for i in proposal_ids:
    if hasattr(proposals, "get_field"):
      mask = proposals.get_field('mask')[i][0].data.numpy()
    else:
      mask = proposals[i]['mask']
    target_id, iou = create_object_id_mapping(mask, warped_proposals)
    ids_chosen += [target_id]
    if target_id != -1:
      output_mask[warped_proposals[target_id]['mask'].cpu().numpy()[0, 0] == 1] = i + 1
      result_dict[i] = {}
      result_dict[i]['mask'] = warped_proposals[target_id]['mask'].data.cpu().numpy()
      result_dict[i]['scores'] = warped_proposals[target_id]['score']
      result_dict[i]['iou_score'] = iou

  for key, prop in warped_proposals.items():
    if key not in ids_chosen:
      result_dict[i] = prop
      result_dict[i]['mask'] = result_dict[i]['mask'].data.cpu().numpy()
      i+=1

  out_file = os.path.join(out_folder, '{:05d}'.format(f + 1) + ".pickle")
  print("pickling {}".format(out_file))
  pickle.dump(result_dict, open(out_file, 'wb'))

  out_file = os.path.join(out_folder, '{:05d}'.format(f + 1) + ".png")
  if np.max(output_mask) > 255:
      print("max value is greater than 255 for {}".format(out_file))
      #output_mask /= np.max(output_mask).astype(np.uint8)
  else:
    save_mask(output_mask, out_file)

  return result_dict


def run_eval(line):
  print(line)
  line = line.rstrip()
  out_folder = out_dir + line
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)
  all_proposals = glob.glob(os.path.join(maskrcnn_data_dir, line, "*.pickle"))
  initial_proposals = os.path.join(maskrcnn_data_dir, line, '{:05d}'.format(0) + ".pickle")
  proposals = pickle.load(open(initial_proposals, 'rb'))
  # TODO: select topn n instead of using conf thresh
  proposals = select_top_predictions(proposals, CONF_THRESH)
  for i in range(len(all_proposals) - 1):
    warped_proposals_path = os.path.join(warped_data_dir, line, '{:05d}'.format(i + 1) + ".pickle")
    warped_proposals = pickle.load(open(warped_proposals_path, 'rb'))
    proposals = save_tracklets(proposals, warped_proposals, out_folder, i)


def main():
  seqs = davis_data_dir + "ImageSets/2017/val.txt"
  lines = ['soapbox']
  pool = mp.Pool(5)
  #with open(os.path.join(seqs), "r") as lines:
  #  pool.map(run_eval, [line for line in lines])
  for line in lines:
    run_eval(line)


if __name__ == '__main__':
  main()
