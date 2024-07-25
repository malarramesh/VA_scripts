# Hiro Project 13/02/24

import os
import ast
import ffmpeg
import numpy as np
import cv2
cv2.setNumThreads(0)
import time
import sys
from datetime import datetime,timedelta
from elasticsearch import Elasticsearch
import tritonclient.grpc as grpcclient

# LOCAL LIB
from use.tracking9 import Tracker
from use.stream import createFileVideoStream
from use.mysql2 import Mysql

# from algo_helper.venkyloiter import Loiter

from algo_helper.Aod_yolov7 import AOD
#from algo_helper.pcount import Crowd
# from algo_helper.pcount_v1 import Crowd
# from algo_helper.pcount_danny import Crowd

# from algo_helper.queue_24_4_27 import Queue
# from algo_helper.veh_dwell_count_d import Vehicle
# from algo_helper.veh_dwell_new import Vehicle

import pytz
timezone = pytz.timezone('UTC')

OUT_XY = (640,580)
MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_DB = 'multi_tenant'
MYSQL_IP = os.environ.get('MYSQL_IP')
ES_USER = "elastic"
ES_PASSWORD = "a73jhx59F0MC39OPtK9YrZOA"
ES_ENDPOINT = "https://e99459e530344a36b4236a899b32887a.westus2.azure.elastic-cloud.com:9243"

class Timer:
    def __init__(self):
        
       
        self.now = datetime.now(timezone)
        # self.now = datetime.utcnow() 
      
        self.now_t = time.time()
        self.count = 0
        self.dt = 0
        self.last_sent = {}

    def update(self):
        self.now = datetime.now(timezone)
        # self.now = datetime.utcnow()
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
        self.input_height = 288
        self.input_width = 512
        self.model_name = "yolov7" # Should be the same as the folder name.
        # Please check this function to change the triton_yolov7 default parameters
        self.triton_client, self.inputs, self.outputs = self.prepare_triton_yolov7(self.input_height, self.input_width, self.model_name)
        ## Tracker
        self.algoSize = (1280, 720)
        self.yoloTracker = {}
        # track_args = ByteArgs()
        for cls in ['person', 'object', 'vehicle', 'animal']:
            # self.yoloTracker[cls] = BYTETracker(track_args, frame_rate=track_args.fps)
            self.yoloTracker[cls] = Tracker((1280,720))
            #self.yoloTracker[cls] = BYTETracker() #bytetracker

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
            if names[int(cls)] == 'person' and float(conf) > 20:#CI
                dets['person'].append(det)
            elif names[int(cls)] in ['car', 'motorcycle', 'bus', 'truck'] and float(conf) > 30: ## SOMETIMES FINE TUNING NEEDED
                #print()
                #print(f"Confidence of {names[int(cls)]}: {conf}")
                dets['vehicle'].append(det)
            # elif names[int(cls)] in {'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'} and float(conf) > 10:
            #     dets['animal'].append(det)
            elif names[int(cls)] in {'backpack', 'suitcase', 'handbag'} and float(conf) > 10:
                dets['object'].append(det)
        return dets

    
    

    def track(self, dets_all):
        typeofdets="yolov7"
        tracks_all = {}
        for cls, tracker in self.yoloTracker.items():
            detections_for_class = dets_all.get(cls, [])
            tracks = tracker.update(np.array(detections_for_class))
            tracks_all[cls] = tracks
        return tracks_all

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

            resized = cv2.resize(image, (new_w, new_h))
            img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
            img[offset_h:(offset_h+new_h), offset_w:(offset_w+new_w), :] = resized # Middle.
        else:
            img = cv2.resize(image, (input_shape[1], input_shape[0]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        
        input_image_buffer = np.expand_dims(img, axis=0)
        #print("Input image shape:", input_image_buffer.shape)
        return input_image_buffer

    def yolov7_postprocess(self, results, letter_box=True):
        num_dets = results.as_numpy("num_dets")
        det_boxes = results.as_numpy("det_boxes")
        det_scores = results.as_numpy("det_scores")
        det_classes = results.as_numpy("det_classes")
        #print("Detected objects:")

        boxes = det_boxes[0, :num_dets[0][0]] / np.array([self.input_width, self.input_height, self.input_width, self.input_height], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]] # Because the topk is included in the tensorrt engine.
        classes = det_classes[0, :num_dets[0][0]].astype(int)
        

        old_h, old_w = self.img_h, self.img_w
        offset_h, offset_w = 0, 0
        i=0

        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
        if letter_box:
            boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
        boxes = boxes.astype(int) # [x1, y1, x2, y2]

        detected_objects = []
        for box, score, label in zip(boxes, scores, classes):
            detected_objects.append([label, score, box])
        #print("Detected objects:", detected_objects)
        #print(f"[{i+1}] Class index: {classes}, Confidence: {score}, Box: {box}")
        return detected_objects

    @staticmethod
    def prepare_triton_yolov7(input_height, input_width, model_name):
        INPUT_NAMES = ["images"]
        OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]
        # Create server context.
        try:
            triton_client = grpcclient.InferenceServerClient(
                url="172.17.0.2:8001",
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

    

class OutStream:
    def __init__(self, outxy, output_path):
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
        frame = cv2.resize(frame, self.outxy, interpolation=cv2.INTER_NEAREST)
        self.process.stdin.write(frame)

class AlgosMan:
    def __init__(self, timer, cam_id, algos_dict):
        self.algos_func = {}
        self.mysql_helper = Mysql({"ip":MYSQL_IP, "user":'graymatics', "pwd":'graymatics', "db":MYSQL_DB, "table":" "})
        elastic = Elasticsearch([ES_ENDPOINT], http_auth=(ES_USER, ES_PASSWORD))
        #elastic = Elasticsearch(["https://search-graymatics-01-jesphdlkxfbucvt6ovyuwoyd6a.us-east-1.es.amazonaws.com"], http_auth=("admin", "Graymatics@123"))
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
        
        # if 'fr' in algos_dict:
        #     #print("FR !!!!!!!!!!")
        #     key = 'fr'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     print(f"{'*' * 15} Running FR {'*' * 15}")
        #     #print(timer,outstream,self.mysql_helper,algos_dict[key],elastic)
        #     self.algos_func[key] = FR(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        #     #print("After calling algos.func")

        

        # Intrusion 
        # if 'intrusion' in algos_dict:
        #     #print("INTRUSION IN ALGOS_DICT!!!!!!!!!")
        #     key = 'intrusion'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Intrusion(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # LOITER
        # if 'loiter' in algos_dict:
        #     key = 'loiter'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Loiter(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # if 'queue' in algos_dict:
        #     #print("key is correct !!!!")
        #     key = 'queue'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Queue(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        #     #print("....Queue........")
            
         # AOD 
        if 'aod' in algos_dict:
            key = 'aod'
            #print("..inside AOD....")
            outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])        
            self.algos_func[key] = AOD(timer, outstream, self.mysql_helper,algos_dict[key], elastic)

         # Vehicle_Count
        # if 'vcount' in algos_dict:
        #     key = 'vcount'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Vehicle(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # if 'uniformtracking' in algos_dict:
        #     print("#################UNIFORM TRACKING #############")
        #     key = 'uniformtracking'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = ClarkUniform(timer, outstream, self.mysql_helper, algos_dict[key], elastic)
        

            # People_Count

        # if 'pcount' in algos_dict:
        #     key = 'pcount'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Crowd(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # if 'violence' in algos_dict:                                                                                                      
        #     key = 'violence'                                                                                                              
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])                                                                   
        #     self.algos_func[key] = Violence(timer, outstream, self.mysql_helper, algos_dict[key], elastic) 
        
        # if 'Weapon' in algos_dict:   
        #     #print("WEAPON IN ALGOS DICT FUNCTION!!!!S")                                                                                                   
        #     key = 'Weapon'                                                                                                              
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])                                                                   
        #     self.algos_func[key] = Weapon(timer, outstream, self.mysql_helper, algos_dict[key], elastic) #self.mysql_helper
        
        # # if 'Clothing' in algos_dict:
        # #     key = 'Clothing'
        # #     #print("...clothing.....")
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] =Clothing1(timer, outstream, self.mysql_helper, algos_dict[key], elastic)  
            #print("...clothing.....")

        # if 'EntryExit' in algos_dict:
        #     #print('key_enterExit!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     key = 'EntryExit'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     # print('enterexit outstream....!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     self.algos_func[key] = EntryExit(timer, outstream, self.mysql_helper, algos_dict[key],elastic)
        
        # #vandalism
        # if 'graffiti' in algos_dict:
        #     key = 'graffiti'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Graffiti(timer, outstream, self.mysql_helper, algos_dict[key], elastic)

        # if 'vehicle_classification' in algos_dict:
        #     key = 'vehicle_classification'
        #     outstream = OutStream(OUT_XY, algos_dict[key]['stream_in'])
        #     self.algos_func[key] = Vehicle(timer, outstream,self.mysql_helper, algos_dict[key], elastic)
    
    def run(self, frame, yoloDets, yoloTracks):
        
        # if 'fr' in self.algos_func:
        #     #print("fr run function initiate")
        #     #self.algos_func['fr'].run(frame.copy(), yoloTracks['person'])
        #     self.algos_func['fr'].run(frame.copy())
    
        # if 'Clothing' in self.algos_func:
        #     #print("...clothing.....")
        #     self.algos_func['Clothing'].run(frame.copy(),yoloTracks['person']) 
        #     # print(".......run-function....clothing........")

        # # # INTRUSION_FR 
        # if 'intrusion' in self.algos_func:
        #     #print("INTRUSION Algos_Function Enabled!!!!!!!!!!!!!")
        #     self.algos_func['intrusion'].run(frame.copy(), yoloTracks['person'])
  
        # # LOITERING_FR 
        # if 'loiter' in self.algos_func:
        #     #print("Loitering Algos_Function Enabled!!!!!!!!!!!!!")
        #     # print("Running Loitering Analytics.")
        #     self.algos_func['loiter'].run(frame.copy(), yoloTracks['person'])

        # if 'queue' in self.algos_func:
        #     #self.algos_func['Queue Management'].run(frame.copy(), personDets['person'], algo_id=22, delay_time=5)
        #     # print("Running Queue Management Analytics.")
        #     self.algos_func['queue'].run(frame.copy(), yoloDets['person'], algo_id=22, delay_time=5)
        #     #print("run-function----Q.......")    

        # AOD

        if 'aod' in self.algos_func:
            print("Running Abandoned Object Analytics.")
            self.algos_func['aod'].run(frame.copy(),yoloTracks['object'])
            #print("....AOD-run-function........")

        # if 'Weapon' in self.algos_func:
        #     #print("INSIDE THE RUN FUNCTION OF WEAPON ANAYTICS@@@@@@@@@S")
        #     #self.algos_func['Weapon'].run(frame.copy())
        #     self.algos_func['Weapon'].run(frame.copy(), yoloTracks['vehicle']+yoloTracks['animals'])
        #     #self.algos_func['Weapon'].run(frame.copy(), yoloTracks['person'])
        
        #if 'weapon' in self.algos_func:
            #self.algos_func['weapon'].run(frame.copy())
            #self.algos_func['weapon'].run(frame.copy(), yoloTracks['person']+yoloTracks['vehicle']+yoloTracks['animals'])
            #self.algos_func['weapon'].run(frame.copy(), yoloTracks['vehicle']+yoloTracks['animals'])
            
        # People Count   
        if 'pcount' in self.algos_func:
            # print("Running People Count Analytics.")
            self.algos_func['pcount'].run(frame.copy(),yoloDets['person'], yoloTracks['person'])

        # if 'uniformtracking' in self.algos_func:
        #     print("UNIFORM TRACKING RUN FUNCTION@@@@@@@@@@@@@@@@@@@@@@@@@")
        #     self.algos_func['uniformtracking'].run(frame.copy(),yoloDets['person'], yoloTracks['person'])

        # if 'vcount' in self.algos_func:
        #     # print("Running Vehicle Count Analytics.")
        #     self.algos_func['vcount'].run(frame.copy(), yoloDets,yoloTracks['vehicle'])  
        
        if 'vcount' in self.algos_func:
            # print("Running Vehicle Count Analytics.")
            self.algos_func['vcount'].run(frame.copy(), yoloTracks['vehicle'])      
        
        # if 'EntryExit' in self.algos_func:
              #self.algos_func['EntryExit'].run(frame.copy(),yoloDets, yoloTracks['vehicle'])

        # if 'violence' in self.algos_func:
        #     self.algos_func['violence'].run(frame.copy())

        # if 'graffiti' in self.algos_func:
        #     # print("Vandalism Algos_Function Enabled!!!!!!!!!!!!!")
        #     self.algos_func['graffiti'].run(frame.copy())

        
        if self.timer.hasExceed('all_algos',5):
            self.mysql_helper.commit_all()


def main(cam_dict):
    cam_id, input_path, algos_dict = cam_dict['camera_id'], cam_dict['rtsp_in'], cam_dict['algos']
    print("algos dict - ", algos_dict)

    #fvs = createFileVideoStream('live', input_path, (1280,720), True, 0, skip=3)
    fvs = createFileVideoStream('video', input_path, (1280,720), True, 0, skip=5)
    
    yolo = Yolov7()
    timer = Timer()
    algosMan = AlgosMan(timer, cam_id, algos_dict)

    print('run_algo_Hiro.py')


    while True:
        
        timer.update()
        frame = fvs.read()
        yoloDets = yolo.detect(frame)
        yoloTracks = yolo.track(yoloDets)
        algosMan.run(frame, yoloDets, yoloTracks)
    
