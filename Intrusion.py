import os
import cv2
import numpy as np
import uuid
import use.drawing as drawing
import matplotlib.pyplot as plt 
from collections import deque
from use.point_in_poly import point_in_poly
from use.nms import filter_overlap
#from use.alert_video import AlertVideo
#from algo_helper.Face_client import Face
from use.alert_video import AlertVideo

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class Intrusion:
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        self.outstream = outstream
        
        mysql_fields = [
            ["time","datetime"],
            ["camera_name","varchar(40)"],
            ["zone", "int(10)"],
            ["track_id", "varchar(45)"],
            ['cam_id','varchar(40)'],
            ['id_account','varchar(40)'],
            ['id_branch','varchar(40)'],
            ['id','varchar(40)']]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('intrude', mysql_fields)
        self.timer = timer
        self.attr = algos_dict
        self.last_sent = self.timer.now_t
        self.es = elastic
        #print("ELASTIC SEARCH VALUE:",self.es)
        self.castSize = (640, 480)
        self.pts = [deque(maxlen=70) for _ in range(5000)]
        self.track_id = []
        self.alertvideo = AlertVideo()
        self.tcount = 0
        self.saved_tids = deque([], maxlen=10)
        #self.roi = [[[597, 191], [700, 173], [1274, 379], [1269, 704], [925, 714]]]
        #self.pts = {}
        #self.gender = Gender()
#        self.alertTrinity = AlertTrinity()
#        output_file = 'Video7_Intrusion.avi'
#        self.out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 10,(640,360))
    """
    def send_es(self, index, date, imgName):
        #print("SEND_ES INDEX :",index)
        #print("DATE :",date)
        #print("SEND_ES IMAGE NAME :",imgName)
        data = {}
        data['description'] = f'Intrusion  detected at {date} at {self.attr["camera_name"]}'
        #data['filename'] = imgName
        data['filename'] = imgName.replace('/resources/', '')
        #data['videoname'] = videoName.replace('/resources/','')        
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        print("*******************************************")
        print("DATA:",data)
        #self.es.index(index=index, body=data)
        print("After index function is called :" , self.es.index)
        print("________________________________________")
        print("DATA:",data)
        print("Index:", index)
    """
    def send_img(self, frame,track_id):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/intrude/{self.attr['camera_id']}/{date}_{track_id}_zone0.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_video(self, frame,track_id, date):
        
        # date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        videoName= f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/intrude/{self.attr['camera_id']}/{date}_{track_id}_video.mp4"
        videoPath = '/home' + videoName
        resizedFrame = cv2.resize(frame, (640,360))
        self.alertvideo.trigger_alert(videoPath,videoName,10)
        return videoName

    def send_alert(self,track,frame, date, uuid0, tid):
        # self.tcount += 1
        # if self.track_id == [] or track_id not in self.track_id:
        # print(uuid0)
        if not tid in self.saved_tids:
            self.track_id.append(tid)
            mysql_values = [date, self.attr['camera_name'], 0,track.id,self.attr['camera_id'],self.attr['id_account'], self.attr['id_branch'], uuid0]
            #mysql_values1 = (date, 0, self.attr['camera_name'], uuid0, self.attr['camera_id'],
            #self.attr['id_account'], self.attr['id_branch'], track_id)
            date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
            date2 = date.replace(" ","_",1)
            #mysql_values = (uuid_, 'loitering', date, date, 'NULL',                                                                                        self.attr['id_account'], self.attr['id_branch'], 0, 'NULL','NULL',self.attr['camera_name']) 
            imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/intrude/{self.attr['camera_id']}/{date2}_{tid}_zone0.jpg"
            imgPath =  '/home'+ imgName
            mysql_values2 = (uuid0, 'Intrusion Alert', date, date, 'NULL', self.attr['id_account'], self.attr['id_branch'], 1, self.attr['camera_name'], 'NULL', imgPath,'NULL', self.attr['camera_id'])
            #mysql_values2 = (uuid0, 'intrude', date, date, 'NULL', self.attr['id_account'], self.attr['id_branch'], 1,'NULL','NULL',self.attr['camera_name'])
            self.mysql_helper.insert_fast('intrude', mysql_values)
            #if self.tcount ==1:
            self.mysql_helper.insert_fast('tickets', mysql_values2)
            imgName = self.send_img(frame,tid)
            videoName = self.send_video(frame,tid, date2)
            #self.send_es(('gmtc_'+ self.attr['id_branch']),date, imgName)
            self.saved_tids.append(tid)
            #self.alertTrinity.trigger_alert(self.attr['algo_name'], 1)


    def run(self, frame, yoloTracks):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        peopleCount = 0
        
        if (self.attr['rois'] is not None):
            drawing.draw_rois(frame, self.attr['rois'], (255,0,0))

        for i, track in enumerate(yoloTracks): 
            x1,y1,x2,y2 = track.attr['xyxy']
            x,y,w,h = track.attr['xywh']
            # print(track.id)
            texts = [track.id[-3:]]
            tid2 = int(texts[-1].replace('_',''))
            tid = str(track.id)
            if w > 1000:
                continue
            if point_in_poly(x, y, self.attr['rois']):
                uuid0 = str(uuid.uuid4())
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                #drawing.plot_one_box(frame, (x1,y1), (x2, y2), color = (0, 0, 255), label=str(track_id[-3:]), line_thickness=1, line_tracking=False, polypoints=None) 
                drawing.plot_one_box(frame, (x1,y1), (x2, y2), color = (0, 0, 0), label=str([tid2]), line_thickness=2, line_tracking=False, polypoints=None)
                #drawing.putTexts(frame, [tid2], x1,y1, size=0.8, thick=1, color=(0,255,255))
                drawing.putTexts(frame, ["INTRUSION DETECTED!!"], 500, 50, size=1.5, thick=2, color=(0,0,255))
                self.send_alert(track, frame, date, uuid0, tid)
        self.alertvideo.updateVideo(frame)
        #img = frame
        #img = cv2.resize(img,(640,360)
        #self.out.write(img)
        self.outstream.write(frame)
       
