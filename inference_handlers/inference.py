def infer(args, testloader, model, criterion, writer):
  task = args.task
  if 'DAVIS3d' in task:
    from inference_handlers.DAVIS3d import infer_DAVIS3d
    infer_DAVIS3d(testloader, model, criterion, writer, args)
  elif 'DAVIS' in task:
    from inference_handlers.DAVIS import infer_DAVIS
    infer_DAVIS(testloader, model, criterion, writer, args)
  elif 'spatialemb' in task.lower():
    from inference_handlers.SpatialEmbInference import infer_spatial_emb
    infer_spatial_emb(testloader, model, criterion, writer, args)
  else:
    raise ValueError("Unknown inference task {}".format(task))