def infer(args, testset, model, criterion, writer, distributed = False):
  task = args.task
  if 'DAVIS3d' in task:
    from inference_handlers.DAVIS3d import infer_DAVIS3d
    infer_DAVIS3d(testset, model, criterion, writer, args)
  elif 'DAVIS' in task:
    from inference_handlers.DAVIS import infer_DAVIS
    infer_DAVIS(testset, model, criterion, writer, args)
  elif 'spatialemb' in task.lower():
    from inference_handlers.SpatialEmbInference import infer_spatial_emb
    infer_spatial_emb(testset, model, criterion, writer, args, distributed=distributed)
  else:
    raise ValueError("Unknown inference task {}".format(task))