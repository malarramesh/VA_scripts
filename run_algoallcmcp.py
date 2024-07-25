# ubuntu@office-am-4:~/hydrabad/src$ vi run_algo_all_HYD.py 
import os
import multiprocessing as mp
import time
import json
import ast

# LOCAL LIB
from use.mysql2 import Mysql
#import run_algocmcp_new
import run_algocmcp
# import trial_run
#from algo_helper.Face_client2 import KnownData

MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_IP = os.environ.get('MYSQL_IP')
#ALGO_DICT = {39:'fire', 32:'clothing',19:'blurred', 33:'train',2:'loiter', 12:'pcount',20:'mask',14:'heatmap',22:'queue',0:'fr',26:'traffic', 0:'fr', 16:'aod', 17:'intrusion',4:'parking', 43:'crowd'}
#ALGO_DICT = {4:'parking',32:'clothing',38:'collapse',2:'loiter', 12:'peoplecount',26:'vehicle_type', 0:'fr', 16:'aod', 13:'anpr',5:'speed',8:'WrongTurn',17:'intrusion',20:'mask',43:'crowd',59:'enterExit'}
#ALGO_DICT = {12:'pcount',43:'crowd',38:'collapse',16:'aod',26:'vcount',0:'fr',2:'loiter',17:'intrusion'}
#ALGO_DICT = {17:'intrusion'}

#ALGO_DICT = {23:'Helmet'}
ALGO_DICT = {22:"queue",0:'fr', 19:'violence', 38: 'collapse', 17:'intrusion',2:'loiter',
             43:'crowd',26:'vcount',13:'anpr',16:'aod',51:'personTrack',4:'parking',
             32:'clothing',92: 'Jaywalk',5:'speed', 59:'entryExit',71:'ppe',29:'crash',39:'fire',
             111:'pf_edge_crossing', 112:'jump_on_track',12:'pcount'}#,48: 'purseSnatching',82:'suicideTendency',83:'smokersGroup',84:'DrinkersGroupIdetification'}
#ALGO_DICT = {0:'fr',2:'loiter', 90:'malebehaviour', 4:'parking', 5:'speed', 9:'graffiti',13:'anpr', 16:'aod',17:'intrusion',89:'malemovement', 29:'accident', 35:'weapon', 43:'crowd',47:'disrobing', 70:'congestion', 75:'signal', 76:'nudity',77:'stalkingWomanArea',78:'stalkingWomanNight',79:'stalkingWomanScooty', 41:'wavinghand', 87: 'acidAttack', 48: 'purseSnatching', 38: 'womenCollapse', 40: 'pullingHair', 50: 'pushingWomenQueue', 19: 'violenceAgainstWomen', 81:'injuryAndBlood',82:'suicideTendency', 84:'DrinkersGroupIdetification', 85:'teasingWomen',83:'smokersGroup', 80: 'gunshotAudio', 88:'zigzagDriving',93:'abnormalRunning',94:'gambling', 96: 'strandedGirl'}
HTTP_OUT_IP = os.environ.get('SERVER_IP')
HTTP_OUT_PORT = 8090
STREAM_IP = '54.156.101.153'
#STREAM_IP = 'localhost'
STREAM_PORT = 8090

mysql = Mysql({"ip":MYSQL_IP, "user":'graymatics', "pwd":'graymatics', "db":MYSQL_DB, "table":""})

def get_rtsp_dict():
    def edit_path(rtsp):
        if rtsp[:4] in {'rtsp', 'http'}:
            return rtsp
        else:
            return rtsp.replace('/usr/src/app/resources/', '/home/videos/').replace('/home/nodejs/app/resources/', '/home/videos/')

    cmd = 'select id, name, rtsp_in, pic_width, pic_height from cameras'
    things = mysql.run_fetch(cmd)
    rtsp_dict = {}
    for camera_id, name, rtsp_in, pic_width, pic_height in things:
        rtsp_dict[camera_id] = [name, edit_path(rtsp_in), (pic_width, pic_height)]
    return rtsp_dict

def reset_stream_url(num, id_, camera_id, algo_id):
    #mysql.run(f'update relations set http_out=concat("http://{HTTP_OUT_IP}:{PORT}/stream",id,".mjpeg")')
    mysql.run(f'update relations set http_out="http://{HTTP_OUT_IP}:{HTTP_OUT_PORT}/stream{num}.mjpeg" where id="{id_}" and camera_id="{camera_id}" and algo_id="{algo_id}"')

def convert_roi(roi_id, cam_size):
    if roi_id is not None:
        rois = [[[int(i['x']/cam_size[0]*1280), int(i['y']/cam_size[1]*720)] for i in ast.literal_eval(roi_id)]]
    else:
        rois = [[[0,0],[1280,0],[1280,720],[0,720]]]
    return rois


def get_cam_dict(rtsp_dict, flag_stream_out):
    stream_num = 0
    cmd = 'select id,camera_id,algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt, http_out from relations'
    things = mysql.run_fetch(cmd)
    cam_dict = {}
    for id_, camera_id, algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt, http_out in things:
        if id_account != "34b9202f-3a4e-41c8-9cb6-a385a549d438":
            continue
        if algo_id not in ALGO_DICT:
            print(f'algo_id: {algo_id} not present')
            continue
        algo_name = ALGO_DICT[algo_id]
        reset_stream_url(stream_num, id_, camera_id, algo_id)
        if camera_id not in cam_dict:
            cam_dict[camera_id] = {}
            cam_dict[camera_id]['camera_id'] = camera_id
            cam_dict[camera_id]['rtsp_in'] = rtsp_dict[camera_id][1]
            cam_dict[camera_id]['algos'] = {}
        
        cam_dict[camera_id]['algos'][algo_name] = {}
        cam_dict[camera_id]['algos'][algo_name]['rois'] = convert_roi(roi_id, rtsp_dict[camera_id][2])
        cam_dict[camera_id]['algos'][algo_name]['atributes'] = json.loads(atributes)
        cam_dict[camera_id]['algos'][algo_name]['algo_name'] = algo_name
        cam_dict[camera_id]['algos'][algo_name]['algo_id'] = algo_id
        cam_dict[camera_id]['algos'][algo_name]['camera_id'] = camera_id
        cam_dict[camera_id]['algos'][algo_name]['camera_name'] = rtsp_dict[camera_id][0]
        cam_dict[camera_id]['algos'][algo_name]['id_account'] = id_account
        cam_dict[camera_id]['algos'][algo_name]['id_branch'] = id_branch
        if flag_stream_out:
            cam_dict[camera_id]['algos'][algo_name]['stream_in'] = f"http://{STREAM_IP}:{STREAM_PORT}/feed{stream_num}.ffm"
        else:
            cam_dict[camera_id]['algos'][algo_name]['stream_in'] = None
            
        stream_num += 1
        
    return cam_dict


if __name__ == '__main__':
    flag_stream_out = True

    rtsp_dict = get_rtsp_dict()
    cam_dict = get_cam_dict(rtsp_dict, flag_stream_out)
    mysql.close()
    # knownface_obj = KnownData(dir0="Sembawang/", npy_prefix="/*_k_arcface_ires100_trt.npy")
    #knownface_obj = None
    
    print('--algos setting--')
    for i, (camera_id, values) in enumerate(cam_dict.items()):
        #camera_id_for_stream_out = ["7bb0ed02-4f1c-4b6a-b0ac-119b98fe5c7f","2c04c3a8-0b91-4e2d-a847-026a76bd5167","56ed3ca2-31f7-4091-90c7-f6eff8f4c997","8191a372-a72c-4d62-aee8-44fba11dd245","0bb9506a-564b-4759-8b63-28c7e02d2f93","830dd268-0f8f-4d6a-a9ca-3f927138f16a","4e41ae4d-4e72-4ea1-91ec-f0ad40d31281"]
        #if camera_id in camera_id_for_stream_out:
        #    flag_stream_out = True
        # count no. of cameras
        #print("********************************")
        # print()
        #print("Total count of camera - ", i)
        # print()
        #print("*********************************")
        # Testing system usage for 2 cameras
        #if i > 0:
        #    continue
        # Testing violence one stored video
        # if camera_id != "4133e108-e8ad-471e-bf69-b922e7baeb56":
        #     continue
        print(f'starting camera_id {camera_id} ...\n')
        #p = mp.Process(target=run_algocmcp_new.main, args=(values, ))
        p = mp.Process(target=run_algocmcp.main, args=(values, ))
        # p = mp.Process(target=trial_run.main, args=(values, ))
        p.daemon = True
        p.start()

    while True:
        time.sleep(9999)
        
