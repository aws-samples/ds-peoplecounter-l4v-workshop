# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. // SPDX-License-Identifier: MIT-0
import time
import numpy as np
import cv2
import grpc
from inference_server_pb2_grpc import InferenceServerStub
import inference_server_pb2 as pb2 
import sys

img = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('frame.bmp',img)
print("start client")
channel = grpc.insecure_channel("unix-abstract:aws.lookoutvision.inference-server")
print("channel set")
stub = InferenceServerStub(channel)
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

