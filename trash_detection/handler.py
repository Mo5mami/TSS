import io
import logging
import numpy as np
import os
import torch
from PIL import Image
import base64

import json
import pickle
import random
import cv2
logger = logging.getLogger(__name__)

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

import albumentations as A
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


@inject_config
def get_detectron2_config(config , weight_path):
    cfg = get_cfg()
    #cfg.MODEL.DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{config.model['model_name']}.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 1
    #cfg.MODEL.WEIGHTS = "./models/best.pth"
    #cfg.MODEL.WEIGHTS = properties.get("model_dir")
    cfg.MODEL.WEIGHTS = weight_path
    cfg.SOLVER.IMS_PER_BATCH = config.model["images_per_batch"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config.model["batchsize_per_image"]   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.model["num_classes"]  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = config.general["LOGS_PATH"]
    cfg.OUTPUT_DIR_BEST=f'{config.general["LOGS_PATH"]}'
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SEED = config.general["seed"]
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR_BEST, exist_ok=True)
    return cfg

def get_valid_transforms():
    return A.Compose(
        [
            
            A.SmallestMaxSize(max_size=1000, p=1.0),
            A.CLAHE(p=1),
            
        ], 
        p=1.0, 
        
    )

class FixFormat:
    
    @classmethod
    def string_to_base64(cls , s):
        return base64.b64encode(s.encode('utf-8'))

    @classmethod
    def byte_to_base64(cls , b):
        return base64.b64encode(b)
    
    @classmethod
    def base64_to_byte(cls , b):
        return base64.b64decode(b)

    @classmethod
    def base64_to_string(cls , b):
        return base64.b64decode(b).decode('utf-8')

    @classmethod
    def byte_to_string(cls , b):
        return base64.b64encode(b).decode('utf-8')


class ModelGenerator(object):
    """
    This class takes as input an image apply preprocess to it 
    to generate the boxes of trash detected then postprocess if needed
    """
    

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.categories = {0: 'Bottle', 1: 'Carton', 2: 'Bottle cap', 3: 'Can',
         4: 'Pop tab', 5: 'Cup', 6: 'Plastic bag & wrapper', 7: 'Styrofoam piece',
         8: 'Other plastic', 9: 'Plastic container', 10: 'Paper', 11: 'Lid', 12: 'Straw',
         13: 'Paper bag', 14: 'Broken glass', 15: 'Plastic utensils', 16: 'Glass jar',
         17: 'Food waste', 18: 'Squeezable tube', 19: 'Shoe', 20: 'Aluminium foil',
         21: 'Unlabeled litter', 22: 'Blister pack', 23: 'Battery', 24: 'Rope & strings',
         25: 'Cigarette', 26: 'Scrap metal',27: 'Plastic glooves'}

    


    def initialize(self, ctx):
        """First try to load torchscript else load eager mode state_dict based model"""

        properties = ctx.system_properties        
        self.device = torch.device("cpu")
        
        model_dir = properties.get("model_dir")
        weight_path = os.path.join(model_dir, "best_model.pth")
        self.cfg=get_detectron2_config(weight_path=weight_path)
        self.predictor = DefaultPredictor(self.cfg)
        logger.debug('Model loaded correctly')
        self.initialized = True

    def preprocess(self, data):
        """
         Preprocess image
        """
        print("data[0] : ",data[0].keys())
        print("data[0][image] : ",type(data[0]["image"]))
        #print("data : ",data.keys())
        #image = data[0].get("data")
        image = bytes(data[0]["image"])
        print("image length : ",len(image))
        self.return_image=data[0].get("return_image").decode()=="True"
        print("return image : ",self.return_image)
        if image is None:
            image = data[0].get("body")
    
        #image=Image.open(io.BytesIO(image)).convert('RGB') 
        image=Image.open(io.BytesIO(image))
        print("mode : " , image.mode)
        #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.image = np.array(image)
        #np_image = np.fromstring(image.decode(), np.uint8)
        #image = cv2.imdecode(np_image, 1)
        #print("original_image : " , image)
        #image = cv2.imdecode(image,cv2.IMREAD_COLOR)
        #image = cv2.imdecode(image)
        self.original_image_shape = self.image.shape
        transformed_image = get_valid_transforms()(image = self.image)["image"]
        self.transformed_image_shape = transformed_image.shape
        return transformed_image

    def inference(self, image, topk=5):
        ''' inference
        '''
        
        

        outputs = self.predictor(image)
        self.outputs = outputs
        #print("outputs : ",outputs)
        print("instances : ",outputs["instances"])
        #print("Boxes : ",outputs["instances"].pred_boxes)
        #print("Boxes to numpy : ",outputs["instances"].pred_boxes.tensor.detach().cpu().numpy())
        #print("All to numpy : ",outputs["instances"].get_fields())
        if (len(outputs["instances"].pred_boxes)==0):
            return []
        
        else : 
            result = outputs["instances"].get_fields()
            for key in result:
                if key == "pred_boxes":
                    result[key] = result[key].tensor.detach().cpu().numpy().tolist()
                else :
                    result[key] = result[key].detach().cpu().numpy().tolist()
            print("Final res : ",result)
            return result
        
        return []

    def boxes_to_original_shape(self , inference_output):
        height_ratio = self.original_image_shape[0] / self.transformed_image_shape[0]
        width_ratio = self.original_image_shape[1] / self.transformed_image_shape[1]
        for box in inference_output["pred_boxes"]:
            box[0] = box[0] * height_ratio
            box[1] = box[1] * width_ratio
            box[2] = box[2] * height_ratio
            box[3] = box[3] * width_ratio
        return inference_output
    
    def add_pred_class_name(self , inference_output):
        inference_output["pred_classes_names"] = []
        for idx in inference_output["pred_classes"]:
            inference_output["pred_classes_names"].append(self.categories[idx])
        return inference_output

    def jsonify(self , inference_output):
        json_result = json.dumps(inference_output , indent = 4)
        return [json_result]

    def set_meta_data(self,):
        meta = detectron2.data.Metadata()
        #meta.name = 'my_dataset'
        meta.set(json_file='new_annotations.json',
          image_root='dataset',
          evaluator_type='coco',
          thing_classes=list(self.categories.values()))
        return meta
    
    def draw_prediction(self ,):
        meta = self.set_meta_data()
        v = Visualizer(self.image[:, :, ::-1], meta, scale=1.2)
        out = v.draw_instance_predictions(self.outputs["instances"].to("cpu"))
        #result_image = cv2.cvtColor(out.get_image() , cv2.COLOR_BGR2RGB)
        result_image = out.get_image()
        #byte_obj = cv2.imencode(".png",out.get_image()[:, :, ::-1])[1].tobytes()
        byte_obj = cv2.imencode(".png",result_image)[1].tobytes()
        #return io.BytesIO(byte_obj)
        #return byte_obj.tostring()
        #return FixFormat.base64_to_string(byte_obj)
        return FixFormat.byte_to_string(byte_obj)

    def postprocess(self, inference_output):
        inference_output = self.add_pred_class_name(inference_output)
        inference_output = self.boxes_to_original_shape(inference_output)
        result = {}
        result["boxes"] = inference_output
        if self.return_image : 
            result["image"] = self.draw_prediction()

        return self.jsonify(result)
        #return torch.Tensor([result["boxes"] , result["image"]])
        #return torch.Tensor([[self.jsonify(result["boxes"]) , result["image"]]])
        

        
        #return inference_output

_service = ModelGenerator()


def handle(data, context):
    #print("data : ",data)
    #print("context : ",context)
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data

if __name__ == "__main__" :
    data = []
    data["body"] = io.BytesIO()

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    print(data)