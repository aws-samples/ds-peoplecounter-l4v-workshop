# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. // SPDX-License-Identifier: MIT-0
import time
import numpy as np
import cv2
import grpc
import edge_agent_pb2 as pb2 
from edge_agent_pb2_grpc import ( 
    EdgeAgentStub
)
from pypylon import pylon
import platform
import sys


print ("usage: capture-subject-basler <deviceSerialNumber> <componentName>")

info = pylon.DeviceInfo()
print("getting camera serial number "+sys.argv[1])
info.SetSerialNumber(sys.argv[1])
converter = pylon.ImageFormatConverter()

tl_factory = pylon.TlFactory.GetInstance()
camera = pylon.InstantCamera()
camera.Attach(tl_factory.CreateFirstDevice(info)) # change this to use device serial number
camera.Open()
camera.StartGrabbing(1)
grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_Return)
if grab.GrabSucceeded():
    img = grab.GetArray()
    print(f'Size of image: {img.shape}')
    image = converter.Convert(grab)
    img = image.GetArray()
    cv2.namedWindow('title', cv2.WINDOW_NORMAL)
    cv2.imshow('title', img)
    cv2.waitKey(0)
    print("start client")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #channel = grpc.insecure_channel("unix-abstract:aws.lookoutvision.inference-server")
    with grpc.insecure_channel("unix:///tmp/aws.iot.lookoutvision.EdgeAgent.sock") as channel:
        print("channel set")
        stub = EdgeAgentStub(channel)
        h, w, c = img.shape
        print("shape="+str(img.shape))
        # Change Component Name below
        response = stub.DetectAnomalies(
        pb2.DetectAnomaliesRequest(
            model_component=sys.argv[2],
            bitmap=pb2.Bitmap(
                width=w,
                height=h,
                byte_data=bytes(img.tobytes())
                )
            )
        )
        print(response)


camera.Close()
cv2.destroyAllWindows()

