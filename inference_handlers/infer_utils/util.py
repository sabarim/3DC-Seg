from inference_handlers import BaseInferenceEngine
from utils.util import all_subclasses


def get_inference_engine(cfg):
  engines = all_subclasses(BaseInferenceEngine)
  try:
    class_index = [cls.__name__ for cls in engines].index(cfg.INFERENCE.ENGINE)
  except:
    raise ValueError("Inference engine {} not found.".format(cfg.INFERENCE.ENGINE))

  engine = list(engines)[class_index]
  return engine(cfg)