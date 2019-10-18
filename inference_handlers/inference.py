from inference_handlers.DAVIS import infer_DAVIS
from inference_handlers.DAVIS3d import infer_DAVIS3d
from inference_handlers.SpatialEmbInference import infer_spatial_emb


def infer(args, testloader, model, criterion, writer):
  task = args.task
  if 'DAVIS3d' in task:
    infer_DAVIS3d(testloader, model, criterion, writer, args)
  elif 'DAVIS' in task:
    infer_DAVIS(testloader, model, criterion, writer, args)
  elif 'spatialemb' in task.lower():
    infer_spatial_emb(testloader, model, criterion, writer, args)
  else:
    raise ValueError("Unknown inference task {}".format(task))