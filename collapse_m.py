#last edited - Danny (23rd May 2023)
#Manideep (22nd May 2023)

import os
import sys
import cv2
cv2.setNumThreads(1)
import numpy as np
import use.drawing as drawing
import math
import uuid
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
from skimage.io import imread, imshow
from skimage import transform
from use.point_in_poly import point_in_poly
from use.alert_video import AlertVideo ## FOR VIDEO ALERT
from use.stream import createFileVideoStream
import random


import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')
FRAME_SIZE = (1280,720) # HD


from datetime import datetime
d = datetime.now()
prev_time = d.strftime('%Y-%-m-%-d_%H:%M:%S')


INPUT_NAMES = ["images"]
OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]

# _classes = ['NonViolence', 'Violence']

class TritonClient:
    def __init__(self, model_name, input_width, input_height, url="172.17.0.2:8000", verbose=False,
                 client_timeout=None, ssl=False, root_certificates=None, private_key=None, certificate_chain=None) -> None:
        self.model_name = model_name
        self.input_width = input_width
        self.input_height = input_height
        self.url = url
        self.verbose = verbose
        self.client_timeout = client_timeout
        self.ssl = ssl
        self.root_certificates = root_certificates
        self.private_key = private_key
        self.certificate_chain = certificate_chain

        # Create server context.
        try:
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
                ssl=self.ssl,
                root_certificates=self.root_certificates,
                private_key=self.private_key,
                certificate_chain=self.certificate_chain)
        except Exception as e:
            print("context creation failed: " + str(e)+model_name)
            sys.exit()

        # Health check.
        if not self.triton_client.is_server_live():
            print("FAILED: is_server_live")
            sys.exit(1)

        if not self.triton_client.is_server_ready():
            print("FAILED: is_server_ready")
            sys.exit(1)

        if not self.triton_client.is_model_ready(self.model_name):
            print("FAILED: is_model_ready")
            sys.exit(1)

        self.inputs = []
        self.outputs = []
        self.inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, self.input_height, self.input_width], "FP32"))
        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        self.outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))

    def predict(self, frame):
        self.input_image = frame
        self.input_image_buffer = self.preprocess(frame, [self.input_height, self.input_width])

        self.inputs[0].set_data_from_numpy(self.input_image_buffer)

        self.results = self.triton_client.infer(model_name=self.model_name,
                                                inputs=self.inputs,
                                                outputs=self.outputs,
                                                client_timeout=self.client_timeout)
        self.detected_objects = self.postprocess(self.results, [self.input_height, self.input_width])
        # self.output_image = self.render(self.input_image, self.detected_objects)
        # cv2.imshow("demo", self.output_image)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     sys.exit()
        self.end = time.time()

        return self.detected_objects

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

    def timer(self):
        self.start = time.time()

    def fps(self):
        s = self.end - self.start
        fps = 1.0 / s
        print("FPS:", "{:.2f}".format(fps))

    def preprocess(self, image, input_shape, letter_box=True):
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
        return input_image_buffer

    def postprocess(self, results, input_shape, letter_box=True):
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
        boxes = boxes.astype(np.int64) # [n, x1, y1, x2, y2]

        detected_objects = []
        for box, score, label in zip(boxes, scores, classes):
            detected_objects.append([label, score, box[0], box[1], box[2], box[3], self.img_w, self.img_h])
        return detected_objects

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

    def info(self):
        # Model metadata.
        try:
            metadata = self.triton_client.get_model_metadata(self.model_name)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED: get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED: get_model_metadata")
                sys.exit(1)
        # Model configuration.
        try:
            config = self.triton_client.get_model_config(self.model_name)
            if not (config.config.name == self.model_name):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED: get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
class Collapse:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        self.outstream = outstream
        self.attr = algos_dict
        self.timer = timer
        self.es = elastic 
        self.castSize = (640, 480)

        mysql_fields = [
            ["id","varchar(45)"],
            ["time","datetime"],
            ["camera_name","varchar(45)"],
            ["cam_id","varchar(45)"],
            ["id_account","varchar(45)"],
            ["id_branch", "varchar(45)"]
            ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('collapse', mysql_fields)
        self.tcount = 0
        self.last_sent = time.time()
        self.alertVideo = AlertVideo()
        self.color = [random.randint(0, 255) for _ in range(3)]


         ###### TRITON SERVER PARAMS ########
        self.model_name = "collapseyolov7" # "yolov7_512x288" #### Needs to be changed.
        self.model_input_width = 640 #512 # 640
        self.model_input_height = 640 #288 # 640 
        self.server_url = "172.30.1.53:8001" # "40.84.143.162:8551"
        self.yolov7 = TritonClient(self.model_name, self.model_input_width,
                                   self.model_input_height, self.server_url)
        
        self.count = 0
        size = (1280, 720)
        self.out = cv2.VideoWriter("md_1.avi", cv2.VideoWriter_fourcc(*'XVID'), 7, size)
       

    def send_video(self, frame, id_, date):
        #date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        videoName= f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/collapse/{self.attr['camera_id']}/{date}_{id_}_video.mp4"
        videoPath = '/home' + videoName
        #print(f"video path ----> {videoPath}")
        resizedFrame = cv2.resize(frame, (640,360))
        self.alertVideo.trigger_alert(videoPath,videoName,30)  ## INCRESE / DECREASE THE DURATION OF THE VIDEO ALERT. (default = 10)
        drawing.saveVideo(videoPath, resizedFrame)
        return videoName

    def send_es(self, index, date, ppe_object, imgName):
        data = {}
        data['description'] = f'Person Collapse detected at {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        self.es.index(index=index, body=data)

    #Image alerts

    def send_img(self, frame, id_, date):
        #print('img alert')
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/collapse/{self.attr['camera_id']}/{date}_{id_}.jpg"
        imgPath = '/home' + imgName
       # resizedFrame = cv2.resize(frame, (640, 480))
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_alert(self, frame): # need to write
        global prev_time
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        if abs(int(date[-2:]) - int(prev_time[-2:])) >= 8:
            # date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
            # self.startTime = time.time()
            # self.tcount += 1
            uuid0 = str(uuid.uuid4())
            date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
            date2 = date.replace(" ","_",1)
            imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/collapse/{self.attr['camera_id']}/{date2}_{uuid0}.jpg"
            imgPath = '/home' + imgName

            mysql_values1 = (uuid0,date,self.attr['camera_name'], self.attr['camera_id'],
                    self.attr['id_account'], self.attr['id_branch'],'NULL','NULL','NULL','NULL')
            mysql_values2 = (uuid0, 'Person Collapse', date, date, 'NULL', self.attr['id_account'], self.attr['id_branch'], 1, self.attr['camera_name'],self.attr['camera_id'], imgPath, 'NULL','NULL')
            self.mysql_helper.insert_fast('collapse', mysql_values1)
            self.mysql_helper.insert_fast('tickets', mysql_values2)
            prev_time = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
            imgName = self.send_img(frame, uuid0, date2)
            videoName = self.send_video(frame, uuid0, date2)
            #self.send_es('gmtc_searcher', date, imgName) 
    

    def run(self, frame):
        start_time_det = time.time()
        
        if (self.attr['rois'] is not None):
            drawing.draw_rois(frame, self.attr['rois'], (0,0,0))
            triton_dets = self.yolov7.predict(frame)
            #text_scale = max(1, frame.shape[1] / 1600.)

        else:
            triton_dets = self.yolov7.predict(frame)
            #text_scale = max(1, frame.shape[1] / 1600.)


        for dets in triton_dets:
            cls_, conf, x1, y1, x2, y2, img_w, img_h = dets

            x,y = (x2+x1)/2, (y2+y1)/2
            if point_in_poly(x,y2,self.attr['rois']):
                if (self.attr['rois'] is not None):
                    drawing.draw_rois(frame, self.attr['rois'], (231,11,165))
                point1, point3 = (x1,y1), (x2, y2)
                h,w = y2-y1,x2-x1                                                                                           

                line_thickness = max(1, int(frame.shape[1] / 500.))
                text_font = cv2.FONT_HERSHEY_COMPLEX_SMALL                
                high_severity = 2

                if conf > 0.75 and cls_ == 0: ## NEED FINE TUNING
                    # cv2.rectangle(frame, (x1,y1), (x2,y2), (83,138,255), 2) 
                    # 
                    drawing.plot_one_box(frame, (x1,y1), (x2, y2), color = self.color, label='Person Collapsed'+str(conf), line_thickness=3, line_tracking=False, polypoints=None)                                                
                    # drawing.putTexts(frame, ['Violence Detected'], (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
                    texts = f"Fall Detected"
                    drawing.putTexts(frame, [texts], 50, 650, size=2, thick=2, color=(255,255,255))
                    self.send_alert(frame)

                else:
                    continue
        if self.count>497:
            self.out.release()
            
            print(f"{'*' * 10} Video is Ready! {'*' * 10}")
        else:
            self.out.write(frame)
            print(self.count)
            print(f"{'*' * 10} Video Processing {'*' * 10}") 
        #self.alertVideo.updateVideo(frame)
        self.outstream.write(frame)
        # self.count+=1
