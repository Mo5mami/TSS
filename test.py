import requests 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import io
import base64
import matplotlib
from PIL import Image
matplotlib.use('Qt5Agg')

#inference_url = "http://127.0.0.1:8080/predictions/trash_detection"
inference_url = "http://127.0.0.1:8000/trash-detection/inference/"



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
    
    @classmethod
    def string_to_byte(cls , s):
        #return base64.b64decode(s.encode('utf-8'))
        return base64.b64decode(s)



if __name__ == "__main__":
    #image_url = "/home/mo5/Desktop/work/projects/trash_taco/taco/media/bottles.jpg"
    #image_url = "/home/mo5/Desktop/work/projects/trash_taco/taco/media/cig.jpg"
    image_url = "/home/mo5/Desktop/work/projects/trash_taco/taco/media/IMG_20210404_121831.jpg"
    image = open(image_url , "rb").read()

    #to_send = {"data" : image , "return_image" : False}
    return_image = True
    to_send = {"data" : image , "return_image" : return_image }
    #result = requests.post(inference_url , data = image)
    result = requests.post(inference_url , data = to_send)
    print("response code : " , result)
    prediction = json.loads(result.content.decode())
    print("response content : " , prediction.keys())
    print("response boxes content : " , prediction["boxes"].keys())
    image_to_show = cv2.imread(image_url)
    image_to_show = cv2.cvtColor(image_to_show , cv2.COLOR_RGB2BGR)
    color = (255, 0, 0)
    thickness = 2
    for box in prediction["boxes"]['pred_boxes']:
        start_point = (int(box[0]),int(box[1]))
        #end_point = (int(box[0] + box[2]), int(box[1]+ box[3]))
        end_point = (int(box[2]),int(box[3]))
        image_to_show = cv2.rectangle(image_to_show, start_point, end_point,color,thickness)
    #print(image_to_show)
    plt.imshow(image_to_show)
    if return_image:
        fl = io.BytesIO(FixFormat.string_to_byte(prediction["image"]))
        predicted_image = np.array(Image.open(fl))
        plt.imshow(predicted_image)
    plt.show()


