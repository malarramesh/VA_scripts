import os
import cv2
import uuid
from datetime import datetime
from datetime import timedelta
import use.drawing as drawing
import use.point_in_poly as point_in_poly
from use.nms import filter_overlap
from use.alert_video import AlertVideo
#from use.alert_thk import alert_thk
d = datetime.now()
prev_time = d.strftime('%Y-%-m-%-d_%H:%M:%S')
MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')

class Repeater:
    def __init__(self, repeat):
        self.repeat = repeat
        self.dets = []

    def update(self):
        dets = []
        for det in self.dets:
            det['count'] += 1
            if det['count'] <= self.repeat:
                dets.append(det)
        self.dets = dets

    def addNew(self, dets):
        for det in dets:
            det['count'] = 0
            self.dets.append(det)

    def run(self, dets):
        self.update()
        self.addNew(dets)
        dets_out = filter_overlap(self.dets, .5)
        return dets_out

class Loiter:
    def __init__(self, timer, outstream, mysql_helper, algos_dict,elastic):
        self.outstream = outstream
        self.timer = timer
        mysql_fields = [
            ['time','datetime'],
            ['dwell','int(11)'],
            ["track_id","varchar(40)"],
            ["camera_name","varchar(40)"],
            ["cam_id","varchar(40)"],
            ["id_branch","varchar(40)"],
            ["id_account","varchar(40)"],
            ['id',"varchar(45)"]
            ]
        self.mysql_helper = mysql_helper
        self.mysql_helper.add_table('loitering', mysql_fields)
        self.attr = algos_dict
        self.time_thres = 20
        #self.es = elastic
        self.castSize = (640, 480)
        self.tracked = []
        self.alertvideo = AlertVideo()
        self.tcount = 0
        # size = (1280,720)
        # self.out = cv2.VideoWriter("loiter_4.avi", cv2.VideoWriter_fourcc(*'MJPG'), 15, size)
        # self.count = 0


    def send_es(self, index, date,duration, imgName):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        #print("ES_INDEX VALUE :::",index)
        #print("ES_DATE:",date)
        #print("ES_DURATION:",duration)
        #print("ES_IMGNAME:",imgName)
        data = {}
        data['description'] = f'loitering for {duration:.0f}s from {date} at {self.attr["camera_name"]}'
        data['filename'] = imgName.replace('/resources/', '')
        data['time'] = date
        data['cam_name'] = self.attr['camera_name']
        data['algo'] = self.attr['algo_name']
        data['algo_id'] = self.attr['algo_id']
        print("Data:",data)
        self.es.index(index=index, body=data)
        print("after index function is called :",self.es.index)
        print("Index:",index)

    def send_img(self, frame,id_, date):
        #date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #print("ID ::::" , id_)
        #date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/loitering/{self.attr['camera_id']}/{date}_{id_}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName

    def send_video(self, frame,id_):
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        videoName= f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/loitering/{self.attr['camera_id']}/{date}_{id_}_video.mp4"
        videoPath = '/home' + videoName
        resizedFrame = cv2.resize(frame, (640,360))
        self.alertvideo.trigger_alert(videoPath,videoName,10)
        drawing.saveVideo(videoPath, resizedFrame)
        return videoName

    def send_alert(self, frame, track, date, level):
        self.tcount += 1
        #print(date)
        global prev_time
        #date1 = str(datetime.now())
        #print(date1)
        date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
        if abs(int(date[-2:]) - int(prev_time[-2:])) >= 2:
            if level is not None:
                uuid_ = str(uuid.uuid4())
                mysql_values = [date, track.dict['loiter'], track.id, self.attr['camera_name'],
                                    self.attr['camera_id'], self.attr['id_branch'], self.attr['id_account'], uuid_]
                self.mysql_helper.insert_fast('loitering', mysql_values)
                date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
                date2 = date.replace(" ","_",1)
                imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/loitering/{self.attr['camera_id']}/{date2}_{track.id}.jpg"
                imgPath = '/home' + imgName
                mysql_values = (uuid_, 'loitering', date, date, 'NULL',
                    self.attr['id_account'], self.attr['id_branch'], 0, self.attr['camera_name'], self.attr['camera_id'], imgPath, 'NULL', 'NULL')
                self.mysql_helper.insert_fast('tickets', mysql_values)
                imgName = self.send_img(frame,track.id, date2)
                videoName = self.send_video(frame,track.id)
                prev_time = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
                #self.send_es(('gmtc_'+'{'+self.attr['id_branch']+'}'), date, track.dict['loiter'], imgName)
                #self.send_es(('gmtc_'+self.attr['id_branch']), date,track.dict['loiter'], imgName)

    def get_alert_level(self, track):
        duration = track.dict['loiter']
        level = None
        if duration > self.time_thres:
            isLoiter = True
            if 'sql_loiter0' not in track.tag:
                level = 1
                track.tag.add('sql_loiter0')
            elif duration > 2 * self.time_thres:
                if 'sql_loiter1' not in track.tag:
                    level = 2
                    track.tag.add('sql_loiter1')
            elif duration > 4 * self.time_thres and 'sql_loiter2' not in track.tag:
                level = 3
                track.tag.add('sql_loiter2')
        else:
            isLoiter = False
        return isLoiter, level

    def run(self, frame, personTracks):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        drawing.draw_rois(frame, self.attr['rois'], (255,0,0))
        for track in personTracks:
            x1, y1, x2, y2 = track.attr['xyxy']
            x, y, w, h = track.attr['xywh']
            if track.attr['conf'] < 0.6:  ## NEED SOME FINE TUNING
                continue
            
            drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(51,255,51), label= None, line_thickness=2, line_tracking=False, polypoints=False)
            if 'loiter' not in track.dict:
                track.dict['loiter'] = 0
            if not point_in_poly.point_in_poly(x, y2, self.attr['rois']):
                continue
            
            track.dict['loiter'] += self.timer.dt
            isLoiter, level = self.get_alert_level(track)
            # draw
            if isLoiter:
                #self.send_alert(frame, track, date,level)
                # if track.dict['loiter'] > 30:
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # if track.dict['loiter'] > 60:
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # if track.dict['loiter'] > 90:
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                #if track.dict['loiter'] >= self.attr['atributes'][1]['time']:
                #    label_w_id = str(track.attr['cls'])+' loitering'
                #    drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(51,51,255), label= label_w_id, line_thickness=3, line_tracking=False, polypoints=False)
                if track.dict['loiter'] > 1:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                drawing.putTexts(frame, ['Loitering Detected'], 500, 50, size=1.3, thick=2, color=(255, 255,255))
                self.send_alert(frame, track, date,level) 
        
        
        # if self.count>1000:
        #     self.out.release()
        #     print("Video is ready")
        # else:
        #     self.out.write(frame)
        #     self.count+=1
        #     print("Video is being written")
        

        self.alertvideo.updateVideo(frame)   ## FOR VIDEO ALERT
        self.outstream.write(frame)
