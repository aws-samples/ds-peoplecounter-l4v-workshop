# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. // SPDX-License-Identifier: MIT-0
import time
import numpy as np
import cv2
import grpc
import edge_agent_pb2 as pb2 
from edge_agent_pb2_grpc import ( 
    EdgeAgentStub
)
import sys

img = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('frame.bmp',img)
print("start client")
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

