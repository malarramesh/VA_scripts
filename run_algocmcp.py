# last edit -MalarRamesh
import os
import ffmpeg
import numpy as np
import cv2
import sys
import time
from datetime import datetime
from elasticsearch import Elasticsearch
from use.tracking9 import Tracker
from tracking.tracker.byte_tracker_yolox import BYTETracker
from use.stream import createFileVideoStream
from use.mysql2 import Mysql
from requests import post

# from algo_helper.intrusion_d import Intrusion
#from algo_helper.helmet import Violence
from algo_helper.Intrusion import Intrusion
#from algo_helper.violence_d import Violence
from algo_helper.thk_violence_yasin2 import Violence
# from algo_helper.loiter1 import Loiter
# from algo_helper.loitering_1 import Loiter
from algo_helper.collapse_d import Collapse
#from algo_helper.vehicle_class_count_1 import Vehicle
from algo_helper.vc_ad_1 import Vehicle
#from algo_helper.anpr_megawide import Plate

#from algo_helper.ANPR_India_1 import Plate
#from algo_helper.crowd_d_12 import Crowd
#from algo_helper.fire_vs1 import Fire
#from algo_helper.fr_dan import FR
#from algo_helper.fr_dan_2 import FR
from algo_helper.Loitering import Loiter
# from algo_helper.aod_v import AOD

#from algo_helper.queue_24_4_27 import Queue

# from algo_helper.aod_d import AOD   ##danny new
#from algo_helper.aod_v import AOD
#from algo_helper.bkpaod import AOD
#from algo_helper.aod_clark import AOD
#from algo_helper.person_tracking import personTracking
#from algo_helper.Illegal_Parking import Parking
from algo_helper.parking_dd_3 import Parking
#from algo_helper.Speed_Triton_1 import Speed
#from algo_helper.Speed_ANPR_15 import Speed
#from algo_helper.jaywalk_dd_2 import Jaywalk
#from algo_helper.people_entry_dd_1 import EntryExit
#from algo_helper.pcount_clark import Crowd

from algo_helper.people_entry_exit_new import pcount

#from algo_helper.accident_dd_1 import Accident
#from algo_helper.ppe_dd import PPE
#from algo_helper.clothing_danny import clothing
# from algo_helper.drinkers_group_ui_pipeline import DrinkersGroup
# from algo_helper.purse_snatching_ui_pipeline import PurseSnatching
#from algo_helper.smokers_group_ui_pipeline import SmokersGroup
# from algo_helper.suicide_tendency_ui_pipeline import SuicideTendency

# from algo_helper.test_loiter import Loiter
#from algo_helper.collapse_thk_esc import Collapse

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from algo_helper.utils import CFEVideoConf, image_resize

import pytz
dubai_timezone = pytz.timezone('Asia/Dubai')
indian_timezone = pytz.timezone('Asia/Kolkata')

OUT_XY = (640,480)
MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')
ES_USER = "elastic"
ES_PASSWORD = "a73jhx59F0MC39OPtK9YrZOA"
ES_ENDPOINT = "https://e99459e530344a36b4236a899b32887a.westus2.azure.elastic-cloud.com:9243"

class Timer:
    def __init__(self):
        self.now = datetime.now(indian_timezone)
        self.now_t = time.time()
        self.count = 0
        self.dt = 0
        self.last_sent = {}

    def update(self):
        self.now = datetime.now(indian_timezone)
        self.count += 1
        self.dt = time.time() - self.now_t
        self.now_t = time.time()

    def hasExceed(self, name, duration):
        if name not in self.last_sent:
            self.last_sent[name] = self.now_t
            return False       
        elif self.now_t - self.last_sent[name] > duration:
            self.last_sent[name] = self.now_t
            return True
        else:
            return False

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
class Yolov7:
    def __init__(self):
        ## Triton YoloV7
        use_yolo_low_res = False
        self.input_height = 640 #288
        self.input_width = 640 #512
        # self.model_name = "collapseyolov7"
        self.model_name = "violenceyolov7" # Should be the same as the folder name. /home/ubuntu/danny/triton(run_triton.sh)
        # self.input_height = 704
        # self.input_width = 1280
        
        # self.model_name = "yolov7_w6" # Should be the same as the folder name. /home/ubuntu/malar/docker/triton
        # Please check this function to change the triton_yolov7 default parameters
        self.triton_client, self.inputs, self.outputs = self.prepare_triton_yolov7(self.input_height, self.input_width, self.model_name)
        #print(self.triton_cl12:'crowd'ient, self.inputs, self.outputs)
        ## Tracker
        self.algoSize = (1280, 720)
        self.yoloTracker = {}
        # track_args = ByteArgs()
        for cls in ['person','object','vehicle']:
        #for cls in ['person']:
        #for cls in ['person', 'vehicle','animal','object','backpack','suitcase','handbag']:
            # self.yoloTracker[cls] = Tracker((1280,720))
            self.yoloTracker[cls] = BYTETracker() #bytetracker

    def detect(self, frame):
        # 1. Prepare Input
        self.input_image_buffer = self.yolov7_preprocess(frame, [self.input_height, self.input_width])
        self.inputs[0].set_data_from_numpy(self.input_image_buffer)
        # 2. Detect
        detections = self.triton_client.infer(model_name=self.model_name,
                                            inputs=self.inputs,
                                            outputs=self.outputs,
                                            client_timeout=None)
        # 3. Post Processing Output
        yoloDets = self.yolov7_postprocess(detections)
        #print(f"raw dets {yoloDets}")
        yoloDets = self.format(yoloDets)
        #yoloMot = self.format(yoloDets, True)
        #return yoloMot
        #print(yoloDets)
        return yoloDets

    def format(self, dets_all, mot=False):
        dets = {cls:[] for cls in {'person', 'object', 'vehicle', 'animal'}}
        #dets = {cls:[] for cls in {'person', 'vehicle','animal''object','backpack', 'suitcase','handbag'}}
        for cls, conf, [x1, y1, x2, y2] in dets_all:
            conf *= 100
            #print(cls)
            # # CHANGE TO HEAD
            # h = int(y2 - y1)
            # y2 = int(y1 + h*0.15)
            if mot:
                det = [int(x1),int(y1),int(x2),int(y2),conf,cls]
            else:
                det = {}
                det['conf'] = conf
                det['cls'] = cls
                det['xyxy'] = [x1,y1,x2,y2]
                h = int(y2 - y1)
                w = int(x2 - x1)
                x = int(x1 + w/2)
                y = int(y1 + h/2)
                det['xywh'] = [x,y,w,h]
                #print(det['cls'])
            # only for person
            if names[int(cls)] == 'person' and float(conf) > 10:#CI
                dets['person'].append(det)
            elif names[int(cls)] in ['car', 'motorcycle', 'bus', 'train'] and float(conf) > 10: ## SOMETIMES FINE TUNING NEEDED
                dets['vehicle'].append(det)
            # elif names[int(cls)] in {'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'} and float(conf) > 10:
            #     dets['animal'].append(det)
            elif names[int(cls)] in {'backpack', 'suitcase', 'handbag'} and float(conf) > 10:
                dets['object'].append(det)
            """
            elif names[int(cls)] == 'backpack' and float(conf) > 10:
                dets['backpack'].append(det)
            elif names[int(cls)] == 'suitcase' and float(conf) > 10:
                dets['suitcase'].append(det)
            elif names[int(cls)] == 'handbag' and float(conf) > 0:
                dets['handbag'].append(det)
            """
        return dets

    def track(self, dets_all):
        typeofdets="violenceyolov7"
        # typeofdets= "yolov7_w6"
        tracks_all = {}
        for cls, tracker in self.yoloTracker.items():
            print(f"tracker===========> {tracker} and its class ====> {cls}")
            tracks = tracker.update(np.array(dets_all[cls]))
            
            tracks_all[cls] = tracks
            
        return tracks_all

    # def track_raw(self, dets):
    #     raw_targets = tracker.update(dets)
    #     return raw_targets

    def yolov7_preprocess(self, image, input_shape, letter_box=True):
        if letter_box:
            self.img_h, self.img_w, _ = image.shape
            new_h, new_w = input_shape[0], input_shape[1]
            offset_h, offset_w = 0, 0

            if (new_w / self.img_w) <= (new_h / self.img_h):
                new_h = int(self.img_h * new_w / self.img_w)
                offset_h = (input_shape[0] - new_h) // 2
            else:
                new_w = int(self.img_w * new_h / self.img_h)
                offset_w = (input_shape[1] - new_w) // 2

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
            img[offset_h:(offset_h+new_h), offset_w:(offset_w+new_w), :] = resized # Middle.
        else:
            img = cv2.resize(image, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0

        input_image_buffer = np.expand_dims(img, axis=0)
        return input_image_buffer

    def yolov7_postprocess(self, results, letter_box=True):
        num_dets = results.as_numpy("num_dets")
        det_boxes = results.as_numpy("det_boxes")
        det_scores = results.as_numpy("det_scores")
        det_classes = results.as_numpy("det_classes")

        boxes = det_boxes[0, :num_dets[0][0]] / np.array([self.input_width, self.input_height, self.input_width, self.input_height], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]] # Because the topk is included in the tensorrt engine.
        classes = det_classes[0, :num_dets[0][0]].astype(np.int)

        old_h, old_w = self.img_h, self.img_w
        offset_h, offset_w = 0, 0

        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
        if letter_box:
            boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
        boxes = boxes.astype(np.int) # [x1, y1, x2, y2]

        #boxes, scores, classes= self.conversion(det_boxes,det_scores,det_classes)

        detected_objects = []
        for box, score, label in zip(boxes, scores, classes):
            detected_objects.append([label, score, box])
        return detected_objects

    def predict_track(self, frame):
        self.input_image = frame
        self.input_image_buffer = self.preprocess(frame, [self.input_height, self.input_width])

        self.inputs[0].set_data_from_numpy(self.input_image_buffer)

        self.results = self.triton_client.infer(model_name=self.model_name,
                                                inputs=self.inputs,
                                                outputs=self.outputs,
                                                client_timeout=self.client_timeout)

        self.detected_objects = self.postprocess_track(self.results,
                [self.input_height, self.input_width])
        self.end = time.time()

        return self.detected_objects
    def postprocess_track(self, results, input_shape, letter_box=True):
        num_dets = results.as_numpy("num_dets")
        det_boxes = results.as_numpy("det_boxes")
        det_scores = results.as_numpy("det_scores")
        det_classes = results.as_numpy("det_classes")

        boxes = det_boxes[0, :num_dets[0][0]] / np.array([self.input_width,
                                                          self.input_height,
                                                          self.input_width,
                                                          self.input_height], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]] # Because the topk is included in the tensorrt engine.
        classes = det_classes[0, :num_dets[0][0]].astype(np.int64)

        old_h, old_w = self.img_h, self.img_w
        offset_h, offset_w = 0, 0
        # Remove, it doesn't work properly.
        if letter_box:
            if (self.img_w / input_shape[1]) >= (self.img_h / input_shape[0]):
                old_h = int(input_shape[0] * self.img_w / input_shape[1])
                offset_h = (old_h - self.img_h) // 2
            else:
                old_w = int(input_shape[1] * self.img_h / input_shape[0])
                offset_w = (old_w - self.img_w) // 2

        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
        if letter_box:
            boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)

        return np.hstack((boxes, scores[:, None], classes[:, None]))

    #@njit
    '''
    def conversion(self,det_boxes,det_scores,det_classes):
        boxes = det_boxes[0, :num_dets[0][0]] / np.array([self.input_width, self.input_height, self.input_width, self.input_height], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]] # Because the topk is included in the tensorrt engine.
        classes = det_classes[0, :num_dets[0][0]].astype(np.int)

        #old_h, old_w = self.img_h, self.img_w
        old_h, old_w = 1280,720
        offset_h, offset_w = 0, 0

        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
        if letter_box:
            boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
        boxes = boxes.astype(np.int) # [x1, y1, x2, y2]
        return boxes, scores, classes
        '''

    @staticmethod
    def prepare_triton_yolov7(input_height, input_width, model_name):
        INPUT_NAMES = ["images"]
        OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]
        # Create server context.
        try:
            triton_client = grpcclient.InferenceServerClient(
                url= "172.17.0.2:8001", # IP:8001
                verbose=False,
                ssl=False,                                     # Enable SSL encrypted channel to the server.
                root_certificates=False,                       # File holding PEM-encoded root certificates.
                private_key=None,                              # File holding PEM-encoded private key.
                certificate_chain=None)      # File holding PEM-encoded certicate chain.
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        # Health check.
        if not triton_client.is_server_live():
            print("FAILED: is_server_live")
            sys.exit(1)
        if not triton_client.is_server_ready():
            print("FAILED: is_server_ready")
            sys.exit(1)
        if not triton_client.is_model_ready(model_name):
            print("FAILED: is_model_ready")
            sys.exit(1)

        # Input Output Buffer
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, input_height, input_width], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))
        return triton_client, inputs, outputs

# class ANPR_Yolox:
#     def __init__(self):
#         self.anprTracker = {}
#         for cls in ['Registration_Plate']:
#             self.anprTracker[cls] = Tracker((1280, 720))

#     def detect(self, frame):
#         ret, im_np = cv2.imencode('.jpg', frame)
#         im_byte = im_np.tobytes()
#         dets = self.get_dets(im_byte)
#         dets_reformatted = self.format_dets(dets)
#         return dets_reformatted

#     def get_dets(self, im_byte):
#         #metadata = post('http://172.16.3.151:5004/predict', data=im_byte)
#         metadata = post('http://172.30.1.53:5004/predict', data=im_byte) ## IF THE NETWORK IS 'host'
#         return metadata.json()

#     def format_dets(self, raw_dets):
#         detections = {cls:[] for cls in {'Registration_Plate'}}
#         # boxes, scores, cls_ids, class_names = raw_dets
#         # boxes = boxes.astype('int16').tolist()
#         # cls_ids = cls_ids.astype('int16').tolist()
#         # scores = scores.astype('float32').tolist()
#         for det in raw_dets:
#             x,y,w,h = det['xywh']
#             x1,y1,x2,y2 = det['xyxy']
#             conf = det['conf']
#             _cls = det['cls']
#             det = {}
#             x, y = (x1+x2)/2, (y1+y2)/2
#             w, h = x2-x1, y2-y1
#             det['conf'] = conf
#             det['cls'] = _cls
#             det['xyxy'] = [x1,y1,x2,y2]
#             det['xywh'] = [x,y,w,h]

#             if _cls == 'Registration_Plate' and conf > .2:
#                 detections['Registration_Plate'].append(det)
#         return detections

#     def track(self, dets_all):
#         typeofdets = "anpr"
#         tracks_all = {}
#         for cls, tracker in self.anprTracker.items():
#             tracks = tracker.update(dets_all[cls], typeofdets)
#             tracks_all[cls] = tracks
#         return tracks_all
    
# class ByteArgs:
#     track_thresh = 0.1 #0.3
#     track_buffer = 30
#     match_thresh = 0.5 #0.8
#     aspect_ratio_thresh = 2.0 #1.6
#     min_box_area = 10.0
#     mot20 = False
#     fps = 30
        
# class YoloX:
#     def __init__(self):
#         self.yoloTracker = {}
#         for cls in ['person', 'vehicle', 'backpack','suitcase']:
#             self.yoloTracker[cls] = Tracker((1280,720))
#             #self.yoloTracker[cls] = BYTETracker()
        
#     def detect(self,frame):
#         ret, im_np = cv2.imencode('.jpg', frame)
#         im_byte = im_np.tobytes()
#         yoloDets_x = self.get_dets(im_byte)
#         yoloDets = self.format(yoloDets_x)
#         #yoloMot = self.format(yoloDets, True)
#         return yoloDets
    
#     def get_dets(self,im_byte):
#         metadata = post('http://172.30.1.53:5008/predict', data=im_byte)
#         return metadata.json()

#     def format(self, dets_all, mot=False):                                      
#         dets = {cls:[] for cls in {'person', 'vehicle', 'backpack', 'suitcase','handbag'}}
#         #for cls, conf, (x1,y1,x2,y2) in dets_all:   
#         dets_AOD = []
#         for det in dets_all:
#             x,y,w,h = det['xywh']                                               
#             x1,y1,x2,y2 = det['xyxy']
#             conf = det['conf']                                                  
#             cls = det['cls'] 
#             if mot:                                                             
#                 det = [int(x1),int(y1),int(x2),int(y2),conf,cls]      
#                 dets_AOD.append(det)
#             else:                                                               
#                 det = {}                                                        
#                 x, y = (x1+x2)/2, (y1+y2)/2                                     
#                 w, h = x2-x1, y2-y1                                             
#                 det['conf'] = conf                                              
#                 det['cls'] = cls                                                
#                 det['xyxy'] = [x1,y1,x2,y2]                                     
#                 det['xywh'] = [x,y,w,h]
#                 dets_AOD.append(det)
#             if cls == 'person' and conf > .2:                                  
#                 dets['person'].append(det)                                      
#             elif cls in {'car','truck','motorcycle','bus','aeroplane','train','boat'} and conf > .01:                                                             
#                 dets['vehicle'].append(det)                                     
#             elif cls == 'backpack' and conf > .01:                              
#                 dets['backpack'].append(det)                                    
#             elif cls == 'suitcase' and conf > .01:                              
#                 dets['suitcase'].append(det) 
#             elif cls == 'handbag' and conf > .01:                              
#                 dets['handbag'].append(det) 
#         return dets,dets_AOD                           
    
#     def track(self, dets_all):
#         tracks_all = {}
#         for cls, tracker in self.yoloTracker.items():
#             tracks = tracker.update(dets_all[cls])
#             tracks_all[cls] = tracks
#         return tracks_all
    

# class ANPR_Yolox:
#     def __init__(self):
#         self.anprTracker = {}
#         for cls in ['Registration_Plate']:
#             self.anprTracker[cls] = Tracker((1280, 720))

#     def detect(self, frame):
#         ret, im_np = cv2.imencode('.jpg', frame)
#         im_byte = im_np.tobytes()
#         dets = self.get_dets(im_byte)
#         dets_reformatted = self.format_dets(dets)
#         return dets_reformatted

#     def get_dets(self, im_byte):
#         #metadata = post('http://localhost:5008/predict', data=im_byte)
#         metadata = post('http://172.30.1.53:5004/predict', data=im_byte) ## IF THE NETWORK IS 'host'
#         return metadata.json()

#     def format_dets(self, raw_dets):
#         detections = {cls:[] for cls in {'Registration_Plate'}}
#         # boxes, scores, cls_ids, class_names = raw_dets
#         # boxes = boxes.astype('int16').tolist()
#         # cls_ids = cls_ids.astype('int16').tolist()
#         # scores = scores.astype('float32').tolist()
#         for det in raw_dets:
#             x,y,w,h = det['xywh']
#             x1,y1,x2,y2 = det['xyxy']
#             conf = det['conf']
#             _cls = det['cls']
#             det = {}
#             x, y = (x1+x2)/2, (y1+y2)/2
#             w, h = x2-x1, y2-y1
#             det['conf'] = conf
#             det['cls'] = _cls
#             det['xyxy'] = [x1,y1,x2,y2]
#             det['xywh'] = [x,y,w,h]

#             if _cls == 'Registration_Plate' and conf > .2:
#                 detections['Registration_Plate'].append(det)

#         return detections
    
#     def track(self, dets_all):
#         tracks_all = {}
#         for cls, tracker in self.anprTracker.items():
#             tracks = tracker.update(dets_all[cls])
#             tracks_all[cls] = tracks
#         return tracks_all
    
class OutStream:
    def __init__(self, outxy, output_path):
       # print('outstream')
        self.output_path = output_path
        self.outxy = outxy
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(outxy[0], outxy[1]))
            .output('{}'.format(output_path))
            .overwrite_output()
            .global_args('-loglevel', 'error')
            .run_async(pipe_stdin=True))

    def write(self, frame):
       # print('write frame')
        frame = cv2.resize(frame, self.outxy)
        self.process.stdin.write(frame)
       # print(self.output_path)
    
class AlgosMan:
    def __init__(self, timer, cam_id, algos_dict):
        self.algos_func = {}
        self.mysql_helper = Mysql({"ip":MYSQL_IP, "user":'graymatics', "pwd":'graymatics', "db":MYSQL_DB, "table":""})
      # self.mysql_helper = Mysql({"ip":MYSQL_IP, "user":'root', "pwd":'MultiTenant@123', "db":MYSQL_DB, "table":""})
        elastic = Elasticsearch([ES_ENDPOINT], http_auth=(ES_USER, ES_PASSWORD))
        self.timer = timer
        mysql_fields = [
                ['id','varchar(40)'],
                ['type','varchar(40)'],
                ["createdAt","datetime"],
                ["updatedAt","datetime"],
                ["assigned","varchar(40)"],
                ["id_account","varchar(40)"], 
                ["id_branch","varchar(40)"],
                ["level","int(11)"], 
                ['reviewed',"varchar(45)"]
                ]
        self.mysql_helper.add_table('tickets', mysql_fields)
        

        if 'pcount' in algos_dict:
            key = 'pcount'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            self.algos_func[key] = pcount(timer, outstream, self.mysql_helper,algos_dict[key], elastic)


        # queue for queue_24_4_27 script
        # if 'queue' in algos_dict:
        #     #print("key is correct !!!!")
        #     key = 'queue'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Queue(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
            #print("....Queue........")

        # if 'loiter' in algos_dict:
        #     key = 'loiter'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Loiter(timer, outstream, self.mysql_helper, algos_dict[key],elastic)


        if 'violence' in algos_dict:                                                                                                      
           key = 'violence'                                                                                                              
           outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
           #print(f"{'*' * 15} Running Violence Detection {'*' * 15}")                                                                   
           self.algos_func[key] = Violence(timer, outstream, self.mysql_helper, algos_dict[key], elastic) 


        # if 'collapse' in algos_dict:
        #     key = 'collapse'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     #print(f"{'*' * 15} Person Collapse / Fall Detection {'*' * 15}")
            
        #     self.algos_func[key] = Collapse(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        #if 'collapse' in algos_dict:
        #    key = 'collapse'
        #    outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #    #print(f"{'*' * 15} Person Collapse / Fall Detection {'*' * 15}")
        #    self.algos_func[key] = Fall(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # if 'fr' in algos_dict:
        #     #print("FR !!!!!!!!!!")
        #     key = 'fr'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     print(f"{'*' * 15} Running FR {'*' * 15}")
        #     #print(timer,outstream,self.mysql_helper,algos_dict[key],elastic)
        #     self.algos_func[key] = FR(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
            #print("After calling algos.func")
        
        # if 'fire' in algos_dict:
        #     key = 'fire'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     print(f"{'*' * 15} Fire Detection {'*' * 15}")
        #     self.algos_func[key] = Fire(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        
        # if 'intrusion' in algos_dict:
        #     key = 'intrusion'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     print(f"{'*' * 15} Running Intrusion {'*' * 15}")
        #     self.algos_func[key] = Intrusion(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        
        # if 'crowd' in algos_dict:
        #     print("Inisde crowd algos dict")
        #     key = 'crowd'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Crowd(timer, outstream, self.mysql_helper, algos_dict[key], elastic)


        # if 'crowd' in algos_dict:
        #     print("Inisde crowd algos dict")
        #     key = 'crowd'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Crowd(timer, outstream, self.mysql_helper, algos_dict[key], elastic)


        # if 'personTrack' in algos_dict:
        #     key = 'personTrack'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = personTracking(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # if 'vcount' in algos_dict:
        #     key = 'vcount'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Vehicle(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # if 'anpr' in algos_dict:
        #     key = 'anpr'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Plate(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

           # AOD
        
        if 'aod' in algos_dict:
            key = 'aod'
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            #self.algos_func[key] = Object_Removal([], timer, outstream, self.mysql_helper,algos_dict[key], elastic) 
            self.algos_func[key] = AOD(timer, outstream, self.mysql_helper,algos_dict[key], elastic)
        
        
        # if 'parking' in algos_dict:
        #     key = 'parking'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Parking(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        
        # if 'Jaywalk' in algos_dict:
        #     key = 'Jaywalk'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     #print('Jay outstream....!')
        #     self.algos_func[key] = Jaywalk(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        #if 'speed' in algos_dict:
            #key = 'speed'
            #outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            #self.algos_func[key] = Speeding(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        
        # if 'speed' in algos_dict: ##jemish
        #     key = 'speed'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Speed(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        
        
        # if 'entryExit' in algos_dict:
        #                 # print('key_enterExit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     key = 'entryExit'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     # print('enterexit outstream....!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     self.algos_func[key] = Crowd(timer, outstream, self.mysql_helper, algos_dict[key],elastic)

        
        # if 'crash' in algos_dict:
        #     key = 'crash'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     #outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Accident(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # if 'ppe' in algos_dict:
        #     key = 'ppe'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     #print(f"{'*' * 15} Running PPE Detection {'*' * 15}")
        #     self.algos_func[key] = PPE(timer, outstream,self.mysql_helper, algos_dict[key], elastic)



        #if 'clothing' in algos_dict:
            #key = 'clothing'
            #outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
            #print('============outstream_clothing')
            #self.algos_func[key] = clothing(timer, outstream, self.mysql_helper, algos_dict[key], elastic)


        # ####################### DRINKERS GROUP ##############
        # if 'drinkersGroup' in algos_dict:
        #     key = 'drinkersGroup'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = DrinkersGroup(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # ################ PURSE SNATCHING ############
        # if 'purseSnatching' in algos_dict:
        #    key = 'purseSnatching'
        #    outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #    self.algos_func[key] = PurseSnatching(timer, outstream, self.mysql_helper,algos_dict[key], elastic)

        # ################ SMOKERS GROUP ############
        # if 'smokersGroup' in algos_dict:
        #    print("INSIDE THE ALGOS FUNCTION !!!!!!!!!!!")
        #    key = 'smokersGroup'
        #    outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #    self.algos_func[key] = SmokersGroup(timer, outstream, self.mysql_helper,algos_dict[key], elastic)

        # ################ SUICIDE TENDENCY ############
        # if 'suicideTendency' in algos_dict:
        #    key = 'suicideTendency'
        #    outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #    self.algos_func[key] = SuicideTendency(timer, outstream, self.mysql_helper,algos_dict[key], elastic)

        if self.timer.hasExceed('all_algos', 5):
            self.mysql_helper.commit_all()

    def run(self, frame, yoloDets, yoloTracks):
        
        # queue for queue_24_4_27 script
        if 'queue' in self.algos_func:
            #self.algos_func['Queue Management'].run(frame.copy(), personDets['person'], algo_id=22, delay_time=5)
            self.algos_func['queue'].run(frame.copy(), yoloDets['person'], algo_id=22, delay_time=5)
            #print("run-function----Q.......") 


        if 'violence' in self.algos_func:                                                                                                 
           self.algos_func['violence'].run(frame.copy(), yoloTracks["person"]) 
           print("run-function------voilence----") 

        #for Helmet
        #if 'violence' in self.algos_func:
            #print('run ....Jaywalk')
            #self.algos_func['violence'].run(frame.copy(), yoloTracks['person'])
        
        
        if 'fr' in self.algos_func:
            #print("FR RUN FUNCTION IS CALLED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.algos_func['fr'].run(frame.copy())

             # PERSON_COLLAPSE_FR
        # if 'collapse' in self.algos_func:
        #     #print("running collapse run function")
        #     self.algos_func['collapse'].run(frame.copy(),yoloDets['person'],yoloTracks['person'])
        if 'collapse' in self.algos_func:
            print("Run frame")                                                                                        
            self.algos_func['collapse'].run(frame.copy())#,yoloTracks["person"])

        if 'intrusion' in self.algos_func:
            self.algos_func['intrusion'].run(frame.copy(),yoloTracks['person'])
        
        if 'loiter' in self.algos_func:
            self.algos_func['loiter'].run(frame.copy(), yoloTracks['person'])
            print("......loiter........")
        
        if 'crowd' in self.algos_func:
            self.algos_func['crowd'].run(frame.copy(),yoloDets['person'], yoloTracks['person'])
        
        if 'personTrack' in self.algos_func:
            self.algos_func['personTrack'].run(frame.copy(), yoloTracks['person'])
        
        if 'vcount' in self.algos_func:
            self.algos_func['vcount'].run(frame.copy(),yoloDets, yoloTracks['vehicle'])
        
        # if 'anpr' in self.algos_func:
        #     self.algos_func['anpr'].run(frame.copy(), anprDets['Registration_Plate'])
            #self.algos_func['anpr'].run(frame.copy(), anprTracks['Registration_Plate'])
        # AOD
        # if 'aod' in self.algos_func:  ##malar one
        #     self.algos_func['aod'].run(frame.copy(),yoloDetsAOD,yoloTracks['person'])

        # Malar using this now 
        if 'aod' in self.algos_func:
            self.algos_func['aod'].run(frame.copy(),yoloDets['person']+yoloDets['object'], yoloTracks['person'])
            # self.algos_func['aod'].run(frame.copy())
            print("......run-function.....aod......")
        # #using now
        # if 'aod' in self.algos_func:
        #     self.algos_func['aod'].run(frame.copy(),yoloDets['person']+yoloDets['backpack']+yoloDets['suitcase'], yoloTracks['person'])
        #     # self.algos_func['aod'].run(frame.copy())
        #     print("......run-function.....aod......")
        
        #danny aod
     
        if 'parking' in self.algos_func:
            self.algos_func['parking'].run(frame.copy(), yoloDets['vehicle']+yoloDets['person'], yoloTracks['vehicle'])
        
        if 'Jaywalk' in self.algos_func:
            #print('run ....Jaywalk')
            self.algos_func['Jaywalk'].run(frame.copy(), yoloTracks['person'],yoloTracks['vehicle'])
        
        #if 'speed' in self.algos_func:(danny)
            #self.algos_func['speed'].run(frame.copy(), yoloDets['vehicle']+yoloDets['person'], yoloTracks['vehicle'])
            #self.algos_func['speed'].run(frame.copy())
        if 'speed' in self.algos_func: ##jemish
            self.algos_func['speed'].run(frame.copy())
            ##self.algos_func['speed'].run(frame.copy(), yoloDets['vehicle']+yoloDets['person'], yoloTracks['vehicle'])
        if 'entryExit' in self.algos_func:
            # print('run_enterExit!!!!!!!!!!!!!!!')
            self.algos_func['entryExit'].run(frame.copy(),yoloDets ['person'], yoloTracks['person'])

        if 'crash' in self.algos_func:
            self.algos_func['crash'].run(frame.copy(), yoloTracks['vehicle'])
        
        if 'ppe' in self.algos_func:
            self.algos_func['ppe'].run(frame.copy())

        #if 'aod' in self.algos_func:
            #self.algos_func['aod'].run(frame.copy(),yoloDets['person']+yoloDets['backpack']+yoloDets['suitcase'], yoloTracks['person'])
        
        #if 'clothing' in self.algos_func:
            #self.algos_func['clothing'].run(frame.copy())

        if 'fire' in self.algos_func: ## id : 39
            #self.algos_func['fire'].run(frame.copy(), yoloTracks['person']+yoloTracks['vehicle'])
            self.algos_func['fire'].run(frame.copy())
        # #################### DRINKERS GROUP ########################
        # if 'drinkersGroup' in self.algos_func:
        #     self.algos_func['drinkersGroup'].run(frame.copy())

        # #################### PURSE SNATCHING ########################
        # if 'purseSnatching' in self.algos_func:
        #     self.algos_func['purseSnatching'].run(frame.copy())

        # #################### SMOKERS GROUP ########################
        # if 'smokersGroup' in self.algos_func:
        #     self.algos_func['smokersGroup'].run(frame.copy())

        # #################### SUICIDE TENDENCY ########################
        # if 'suicideTendency' in self.algos_func:
        #     self.algos_func['suicideTendency'].run(frame.copy())
        if 'pcount' in self.algos_func:
            self.algos_func['pcount'].run(frame.copy(),yoloDets['person'], yoloTracks['person'])

        if self.timer.hasExceed('all_algcos', 5):
            self.mysql_helper.commit_all()
        

def main(cam_dict):
    cam_id, input_path, algos_dict = cam_dict['camera_id'], cam_dict['rtsp_in'], cam_dict['algos']
    #print(f'starting {input_path}')
    print("Creating stream")
    fvs = createFileVideoStream('video', input_path, (1280,720), True, 0, skip=3)
    print("Creating YOLOv7")
    yolov7 = Yolov7()
    print("Creating ANPR")
    # anpr_yolox = ANPR_Yolox()
    timer = Timer()
    #byteMe_yolox = BYTETracker()
    print("Creating AlgosMan")
    algosMan = AlgosMan(timer, cam_id, algos_dict)

    print('run_algo.py')
    
    while True:
        timer.update()
        frame = fvs.read()
        yoloDets = yolov7.detect(frame)
        yoloTracks = yolov7.track(yoloDets)
        #print(yoloTrack
        # anprDets = anpr_yolox.detect(frame)
        # print(yoloTracks['object'])
        # anprTracks = anpr_yolox.track(anprDets)
        # print("Run main")                                  
        # algosMan.run(frame, yoloDets, yoloTracks, anprDets, anprTracks)
        algosMan.run(frame, yoloDets, yoloTracks)
    
    # while True:
    #     timer.update()
    #     frame = fvs.read()
    #     # yoloDets , yoloDetsAOD= yolox.detect(frame)
    #     #yoloDets = yolox.detect(frame)
    #     anprDets = anpr_yolox.detect(frame)
    #     #print("YOLODETS :::::",yoloDetsAOD)
    #     yoloTracks = yolox.track(yoloDets)
    #     #anprTracks = anpr_yolox.track(anprDets)
    #     #print("yoloTracks",yoloTracks)
    #     #algosMan.run(frame,yoloDets, yoloTracks,yoloDetsAOD,anprDets,anprTracks)
    #     algosMan.run(frame,yoloDets, yoloTracks,yoloDetsAOD, anprDets)
    #     #algosMan.run(frame)






