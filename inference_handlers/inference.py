from inference_handlers.DAVIS import infer_DAVIS


def infer(task, testloader, model, criterion, writer, *args):
  if 'DAVIS' in task:
    infer_DAVIS(testloader, model, criterion, writer)
  else:
    raise ValueError("Unknown inference task {}".format(task))