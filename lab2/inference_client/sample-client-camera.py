# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. // SPDX-License-Identifier: MIT-0
import time
import numpy as np
import cv2
import grpc
import edge_agent_pb2 as pb2 
from edge_agent_pb2_grpc import ( 
    EdgeAgentStub
)



def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=1,
):
    return (
          "nvarguscamerasrc ! "
          "video/x-raw(memory:NVMM), "
          "width=(int)%d, height=(int)%d, "
          "format=(string)NV12, framerate=(fraction)%d/1 ! "
          "nvvidconv flip-method=1 ! "
          "videocrop top=1300 bottom=200 left=0 right=0 !"
          "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
          "videoconvert ! "
          "video/x-raw, format=(string)BGR ! appsink"
          % (
               capture_width,
               capture_height,
               framerate,
               display_width,
               display_height,
            )
          )

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if cap.isOpened():
    ret_val, img = cap.read()
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    print("ret_val="+str(ret_val))
#    print("img="+str(img))
    cv2.imwrite("frame.bmp",img)
    cap.release()
    print("start client")
    #channel = grpc.insecure_channel("unix-abstract:aws.lookoutvision.inference-server")
    with grpc.insecure_channel("unix:///tmp/aws.iot.lookoutvision.EdgeAgent.sock") as channel:
        print("channel set")
        stub = EdgeAgentStub(channel)
        h, w, c = img.shape
        print("shape="+str(img.shape))
        response = stub.DetectAnomalies(
            pb2.DetectAnomaliesRequest(
                model_component="ComponentCircuitBoard",
                bitmap=pb2.Bitmap(
                    width=w,
                    height=h,
                    byte_data=bytes(img.tobytes())
                    ) 
                )
        )
        print(response)


else:
    print("Unable to open camera")
