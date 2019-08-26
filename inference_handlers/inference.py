from inference_handlers.DAVIS import infer_DAVIS


def infer(args, testloader, model, criterion, writer):
  task = args.task
  if 'DAVIS' in task:
    infer_DAVIS(testloader, model, criterion, writer, args)
  else:
    raise ValueError("Unknown inference task {}".format(task))