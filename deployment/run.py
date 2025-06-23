
from functools import partial
import json
from datetime import datetime as dt
from pathlib import Path



def timestampStr(dd = dt.now()):
    return dd.strftime("%Y-%m-%dT%H:%M:%S")



'''
[ALPRResult(detection=DetectionResult(label='License Plate', confidence=0.75244140625, bounding_box=BoundingBox(x1=270, y1=775, x2=340, y2=810)), ocr=OcrResult(text='X2E447', confidence=0.9631446003913879)), ALPRResult(detection=DetectionResult(label='License Plate', confidence=0.58203125, bounding_box=BoundingBox(x1=1157, y1=731, x2=1233, y2=764)), ocr=OcrResult(text='XIR177', confidence=0.9548702239990234))]

'''

def alpr(fastalpr,frame):
    import cv2
    from VehicleDetectionTracker.VehicleDetectionTracker import encode_image_base64
    #we should only have one vehicle here
    #so if we have a result we use the highest one 
    alpr_results = fastalpr.predict(frame)
    #print(alpr_results)
    if not alpr_results is None and len(alpr_results) > 0 and not alpr_results[0].ocr is None:
        results = alpr_results[0]
        res = {"ocr-text":alpr_results[0].ocr.text,"confidence":alpr_results[0].ocr.confidence}
        
        '''
        x1  = results.detection.bounding_box.x1
        y1 = results.detection.bounding_box.y1
        x2  = results.detection.bounding_box.x2
        y2  = results.detection.bounding_box.y2
        
        res["alpr_base64"] = encode_image_base64(frame[y1:y2,x1:x2])
        print(res)
        '''
        return ["alpr",res]
        #{"ocr-text":alpr_results[0].ocr.text,"confidence":alpr_results[0].ocr.confidence}]
        
    return [None,None]



def collect_callback(calllist,arg):
    return [cc(arg) for cc in calllist]


def exec_callback(calllist,arg):
    for cc in calllist:
        cc(arg)

def result_callback(outpath, result):
    
    
    with open (outpath,'a') as fp:
        json.dump(result_collect(result),fp)
        fp.write(",\n")
        
        
        
def result_collect(result):
    

    collected  = {
        "number_of_vehicles_detected": result["number_of_vehicles_detected"],
        "create-time":timestampStr(),
        "detected_vehicles": [
            {
                "vehicle_id": vehicle["vehicle_id"],
                "vehicle_type": vehicle["vehicle_type"],
                "detection_confidence": vehicle["detection_confidence"],
                "color_info": vehicle["color_info"],
                "model_info": vehicle["model_info"],
                "speed_info": vehicle["speed_info"],
                "vehicle_frame_timestamp": timestampStr(vehicle["vehicle_frame_timestamp"])
            }
            for vehicle in result['detected_vehicles'  ]
        ]
        }
     
    return collected
                
                

def callbacke_save_annotations(outpath, result):
    if "annotated_frame_base64" in result:
        import cv2
        ts = timestampStr(result['detected_vehicles'][0]["vehicle_frame_timestamp"])
        #ts = ts.replace(" ","").replace("-","_")
        from VehicleDetectionTracker.VehicleDetectionTracker import decode_image_base64
        annotated_frame_base64 = result["annotated_frame_base64"]
        annotated_frame = decode_image_base64(annotated_frame_base64)
        imgp = Path(outpath)
        imgp.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(imgp.joinpath(f"{ts}.jpg")),annotated_frame)
        
        
        
def callbacke_detections(outpath, result):
    if "annotated_frame_base64" in result:
        import cv2
        ts = timestampStr(result['detected_vehicles'][0]["vehicle_frame_timestamp"])
        #ts = ts.replace(" ","").replace("-","_")
        from VehicleDetectionTracker.VehicleDetectionTracker import decode_image_base64
        imgp = Path(outpath,ts)
        imgp.mkdir(parents=True, exist_ok=True)
        for vehicle in result['detected_vehicles']:
            vehicle_frame_base64 = vehicle["vehicle_frame_base64"]
            vehicle_frame = decode_image_base64(vehicle_frame_base64)
            cv2.imwrite(str(imgp.joinpath(f"{vehicle['vehicle_id']}.jpg")),vehicle_frame)



                
def pushResults(url, result):
    collected = result_collect(result)
    import httpx
    

        

def test():
    
    from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker

    motor_types = ['car','bus','truck']
    
    
    callback_eval_list = []
    callbacks = None
    try:
        from fast_alpr import ALPR
    
        # You can also initialize the ALPR with custom plate detection and OCR models.
        fastalpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="global-plates-mobile-vit-v2-model",
        )
    
        callback_eval_list.append(partial(alpr,fastalpr))
        
    except:
        pass
        
    
    if len(callback_eval_list) > 0:
        callbacks = partial(collect_callback,callback_eval_list)
        
    
    
    #video_path = "[[YOUR_STREAMING_SOURCE]]"
    video_path = '/app/testvideo.mp4'
    outpath = "/app/res.json"
    outimagepath = "/app/images"
    
    
    
    #vehicle_detection = VehicleDetectionTracker(model_path="/app/yolov8l14June2025.pt",callback_classifier=callbacks)
    vehicle_detection = VehicleDetectionTracker(model_path="/app/yolo11l.pt",callback_classifier=callbacks)
    
    '''
    result_callback = lambda result: print({
        "number_of_vehicles_detected": result["number_of_vehicles_detected"],
        "detected_vehicles": [
            {
                "vehicle_id": vehicle["vehicle_id"],
                "vehicle_type": vehicle["vehicle_type"],
                "detection_confidence": vehicle["detection_confidence"],
                "color_info": vehicle["color_info"],
                "model_info": vehicle["model_info"],
                "speed_info": vehicle["speed_info"]
            }
            for vehicle in result['detected_vehicles'  ]
        ]
    })
    '''
    #import torch, ultralytics
    #torch.serialization.add_safe_globals([ultralytics.nn.modules.Conv,ultralytics.nn.tasks.DetectionModel,torch.nn.modules.container.Sequential])
    
    
    #with torch.serialization.add_safe_globals([ultralytics.nn.modules.Conv]):
    #    with torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel,torch.nn.modules.container.Sequential]):
        
    vehicle_detection.process_video(video_path, result_callback = partial(exec_callback, [partial(result_callback,outpath),
                                                                                          partial(callbacke_save_annotations,outimagepath),
                                                                                          partial(callbacke_detections,outimagepath)]))




def main(pushurl = ""):
    from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker    
    callbacks = partial(collect_callback,[alpr])
    
    
    
    
    
    