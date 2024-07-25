#last edited by danny - dwell time, severity, alert rate minimized

import os
import cv2
import time
from datetime import datetime
import use.drawing as drawing
from use.point_in_poly import point_in_poly
from collections import deque
import math
import uuid


MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')


d = datetime.now()
prev_time = d.strftime('%Y-%-m-%-d_%H:%M:%S')


names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



class Parking:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        self.outstream = outstream
        self.timer = timer
        self.attr = algos_dict
        self.es = elastic

        # ADDITIONS
        mysql_fields = [                                               
                ["track_id", 'varchar(45)'], 
                ["time","datetime"],                                                                                                   
                ["dwell","int(10)"],                                                                                                   
                ['cam_id','varchar(45)'],                                                                                              
                ['cam_name','varchar(45)'],                                                                                            
                ['id_account','varchar(45)'],                                                                                          
                ['id_branch','varchar(45)'],
                ["level","int"]
                ]                                                                                           
        self.mysql_helper = mysql_helper                                                                                               
        self.mysql_helper.add_table('parking', mysql_fields)
        self.now_t = time.time()
        self.saved_tids = deque([], maxlen=15)
        self.saved_tids_severe = deque([], maxlen=15)
        self.saved_tids_critical = deque([], maxlen=15)


    def send_img(self, frame, id_,date):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/parking/{self.attr['camera_id']}/{date}_{id_}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, (640, 480))
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_alert(self, frame, track, date, duration, level, msg, tid):
        global prev_time      
        #date = self.timer.now.strftime('%Y-%m-%-d_%H:%M:%S')
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        date2 = date.replace(" ","_",1)
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/parking/{self.attr['camera_id']}/{date}_{track.id}.jpg"
        imgPath = '/home' + imgName
        # if abs(int(date[-2:]) - int(prev_time[-2:])) >= 3:
        if level==0 and (tid not in self.saved_tids):
            car_id = f'Car_id : {tid}'
            drawing.putTexts(frame, [car_id], 60, 50, size = 1.2, thick = 2, color=(6, 145, 255))
            frame1 = frame.copy()
            uuid_ = str(uuid.uuid4()) 
            mysql_values1 = [track.id,date, duration, self.attr['camera_id'], self.attr['camera_name'],self.attr['id_account'],self.attr['id_branch'], level]                            
            self.mysql_helper.insert_fast('parking', mysql_values1)                                                  
            mysql_values2 = (uuid_, 'Parking Violation in Restricted Area', date, date, 'NULL',                                                            
                        self.attr['id_account'], self.attr['id_branch'], level, self.attr['camera_name'],self.attr['camera_id'],imgPath, 'NULL', 'NULL')
            self.mysql_helper.insert_fast('tickets', mysql_values2)
            imgName = self.send_img(frame1, track.id,date2)
            prev_time = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
            self.saved_tids.append(tid)
        elif level==1 and (tid not in self.saved_tids_severe):
            car_id = f'Car_id : {tid}'
            drawing.putTexts(frame, [car_id], 60, 50, size = 1.2, thick = 2, color=(6, 145, 255))
            frame1 = frame.copy()
            uuid_ = str(uuid.uuid4()) 
            mysql_values1 = [track.id,date, duration, self.attr['camera_id'], self.attr['camera_name'],self.attr['id_account'],self.attr['id_branch'], level]                            
            self.mysql_helper.insert_fast('parking', mysql_values1)                                                  
            mysql_values2 = (uuid_, 'Parking Violation in Restricted Area', date, date, 'NULL',                                                            
                        self.attr['id_account'], self.attr['id_branch'], level, self.attr['camera_name'],self.attr['camera_id'],imgPath, 'NULL', 'NULL')
            self.mysql_helper.insert_fast('tickets', mysql_values2)
            imgName = self.send_img(frame1, track.id,date2)
            prev_time = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
            self.saved_tids_severe.append(tid)
        elif level==2 and (tid not in self.saved_tids_critical):
            car_id = f'Car_id : {tid}'
            drawing.putTexts(frame, [car_id], 60, 50, size = 1.2, thick = 2, color=(0, 28, 185))
            frame1 = frame.copy()
            uuid_ = str(uuid.uuid4()) 
            mysql_values1 = [track.id,date, duration, self.attr['camera_id'], self.attr['camera_name'],self.attr['id_account'],self.attr['id_branch'], level]                            
            self.mysql_helper.insert_fast('parking', mysql_values1)                                                  
            mysql_values2 = (uuid_, 'Parking Violation in Restricted Area', date, date, 'NULL',                                                            
                        self.attr['id_account'], self.attr['id_branch'], level, self.attr['camera_name'],self.attr['camera_id'],imgPath, 'NULL', 'NULL')
            self.mysql_helper.insert_fast('tickets', mysql_values2)
            imgName = self.send_img(frame1, track.id,date2)
            prev_time = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
            self.saved_tids_critical.append(tid)
        else:
            pass
            


            #self.send_es('gmtc_searcher', track.id, date, imgName) 
            
    def getVelocity(self, track, maxlen):
        x,y,w,h = track.attr['xywh']
        key = 'getVelocity'
        if key not in track.dict:
            track.dict[key] = deque(maxlen=maxlen)
        track.dict[key].append((x,y))
        if len(track.dict[key]) == maxlen:
            x0,y0 = track.dict[key][0]
            x1,y1 = track.dict[key][-1]
            dx = x1 - x0
            dy = y1 - y0
            dist = (dx**2 + dy**2)**.5/w
            direct = math.degrees(math.atan2(dy,dx))
            return dist, int(direct)
        else:
            return (None, None)

    def isPark(self,track, rois, speedThres, dt,maxlen):
        x,y,w,h = track.attr['xywh']
        key = 'getVelocity'
        if key not in track.dict:
            track.dict[key] = deque(maxlen=maxlen)
        track.dict[key].append((x,y))
        if 'park' not in track.dict:
            track.dict['park'] = 0
            track.dict['park_roi'] = 0

        if len(track.dict[key]) == maxlen:
            x0,y0 = track.dict[key][0]
            x1,y1 = track.dict[key][-1]
            dx = x1 - x0
            dy = y1 - y0
            speed = (dx**2 + dy**2)**.5/w
            if speed is not None:
                speed_new = int(speed * 15)
                if speed_new < speedThres:
                    track.dict['park'] += dt
                    if point_in_poly(x,y, rois):
                        track.dict['park_roi'] += dt
                else:
                    track.dict['park'] -= dt
                    track.dict['park_roi'] -= dt
                    track.dict['park'] = max(0, track.dict['park'])
                    track.dict['park_roi'] = max(0, track.dict['park_roi'])


    def traffic(self, frame,yoloTracks):
        dt = time.time() - self.now_t
        self.now_t = time.time()        
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')


        if (self.attr['rois'] is not None):
            drawing.draw_rois(frame, self.attr['rois'], (33,229,233))

        
        for i, track in enumerate(yoloTracks):
            x1,y1,x2,y2 = track.attr['xyxy']
            x,y,w,h = track.attr['xywh']
            speed, direction = self.getVelocity(track,60)
            texts = [track.id[-3:]]
            tid = int(texts[-1].replace('_',''))
            self.isPark(track, self.attr['rois'], 1, dt,60)
            severity = 0
            _cls = names[track.attr['cls']]

            if point_in_poly(x,y2,self.attr['rois']):
                if speed is not None:
                    speed_new = int(speed * 15)
                    if speed_new < 10:

                        if track.dict['park'] > 0:
                            if 60 < track.dict['park'] < 3600:
                                metric = 'mins'
                                new_count = track.dict['park'] / 60
                            elif 3600 < track.dict['park'] < 86400:
                                metric = 'hrs'
                                new_count = track.dict['park'] / 3600
                            elif track.dict['park'] > 86400:
                                metric = 'days'
                                new_count = track.dict['park'] / 86400
                            else:
                                metric = 'secs'
                                new_count = track.dict['park']
                            
                            texts.append('Parked for {} {}'.format(int(new_count), metric))
                            #label_w_id = str(_cls.title())+' ID: '+str(tid)i
                            label_w_id = _cls.title()+':'+str(tid)
                            
                            drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(0,0,0), label= label_w_id, line_thickness=2, line_tracking=False, polypoints=False)
                            drawing.putTexts(frame, [texts[1]], x2-10, y1, size=0.5, thick=1, color=(255,255,255))
                            duration = int(track.dict['park'])
                            drawing.draw_rois(frame, self.attr['rois'], (51,51,255))
                            if duration >= self.attr['atributes'][1]['time']:

                                if duration >= int(3 * self.attr['atributes'][1]['time']):
                                    t1 = " Critical Parking Violation Detected   "
                                    drawing.putTexts(frame, [t1], 500, 50, size = 1.2, thick = 2, color=(0, 28, 185))
                                    severity = 2
                                    text = "Critical Parking Violation"
                                    drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(0, 28, 185), label= label_w_id, line_thickness=2, line_tracking=False, polypoints=False)
                                    self.send_alert(frame, track, date, duration, severity, text, tid)



                                elif duration >= int(2 * self.attr['atributes'][1]['time']):
                                    t2 = " Prolonged Parking Violation Detected "
                                    drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(6, 145, 255), label= label_w_id, line_thickness=2, line_tracking=False, polypoints=False)
                                    drawing.putTexts(frame, [t2], 500, 50, size = 1.2, thick = 2, color=(6, 145, 255))
                                    severity = 1
                                    text = "Prolonged Parking Violation"
                                    self.send_alert(frame, track, date, duration, severity, text, tid)



                                else:
                                    #if duration <= int(1.7 * self.attr['atributes'][1]['time']):
                                    t3 = "      Parking Violation Detected      "
                                    drawing.putTexts(frame, [t3], 500, 50, size = 1.2, thick = 2, color=(6, 145, 255))
                                    severity = 0
                                    text = "Parking Violation"
                                    self.send_alert(frame, track, date, duration, severity, text, tid)


    def run(self, frame, yoloDets, vehicleTracks):
        self.traffic(frame,vehicleTracks)
        self.outstream.write(frame)





    # def send_es(self, index, track, date_es,imgName):
    #     data = {}
    #     data['description'] = f'{track} at {date_es}'
    #     data['filename'] = imgName.replace('/resources/', '')
    #     data['time'] = date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
    #     data['cam_name'] = self.attr['camera_name']
    #     data['algo'] = self.attr['algo_name']
    #     data['algo_id'] = self.attr['algo_id']
    #     self.es.index(index=index, body=data)

