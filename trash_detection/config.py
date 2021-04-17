from abc import ABC,ABCMeta, abstractmethod
from yacs.config import CfgNode as CN


_C = CN()
_C.general=CN()
_C.general.seed = 42
_C.general.tool = "detectron2"
_C.general.category = "super_category"
_C.general.TTA = False
_C.general.LOGS_PATH = "logs"
_C.general.MODELS_PATH = "models"
_C.general.DEBUG_PATH = "test"

_C.preprocess=CN()
_C.preprocess.height = None
_C.preprocess.width = None
_C.preprocess.max_size = 1500

_C.model=CN()

_C.model.num_classes = 29 #29 if super category 60 if normal category 
_C.model.model_name = "faster_rcnn_R_101_FPN_3x"
_C.model.batchsize_per_image = 1024
_C.model.images_per_batch = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    #return _C.clone()
    return _C

def dump_cfg(config = get_cfg_defaults() , path = "experiment.yaml"):
    """Save a yacs CfgNode object in a yaml file in path."""
    stream = open(path, 'w')
    stream.write(config.dump())
    stream.close()

def inject_config(funct):
    """Inject a yacs CfgNode object in a function as first arg."""
    def function_wrapper(*args,**kwargs):
        return funct(_C,*args,**kwargs)  
    return function_wrapper
c=get_cfg_defaults()

