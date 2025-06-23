
from ultralytics/ultralytics:8.3.156-cpu

# now we add tracking 
RUN mkdir /app

workdir /app


COPY deployment/ultralytics/requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# cached till here ! takes for ever

COPY VehicleDetectionTracker /app/VehicleDetectionTracker
copy deployment/run.py /app/run.py
#copy yolov8l14June2025.pt /app/yolov8l14June2025.pt
ARG testfile
copy $testfile /app/testvideo.mp4
COPY deployment/ultralytics/yolo11l.pt /app/yolo11l.pt


copy deployment/fast-alpr/global_mobile_vit_v2_ocr.onnx /root/.cache/fast-plate-ocr/global-plates-mobile-vit-v2-model/
copy deployment/fast-alpr/global_mobile_vit_v2_ocr_config.yaml /root/.cache/fast-plate-ocr/global-plates-mobile-vit-v2-model/
copy deployment/fast-alpr/yolo-v9-t-384-license-plates-end2end.onnx /root/.cache/open-image-models/yolo-v9-t-384-license-plate-end2end/

#copy deployment/fast-alpr/global_mobile_vit_v2_ocr.onnx /app/
#copy deployment/fast-alpr/global_mobile_vit_v2_ocr_config.yaml /app/
#copy deployment/fast-alpr/yolo-v9-t-384-license-plates-end2end.onnx /app/


#CMD ["python3","/app/run.py"]