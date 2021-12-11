# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. // SPDX-License-Identifier: MIT-0
# This loads the model, to warm up the inference engine - not needed if the model is set to 'Auto-load: true' in the AWS IoT GreenGrass Recipe
import time
import grpc
from inference_server_pb2_grpc import InferenceServerStub
import inference_server_pb2 as pb2 

channel = grpc.insecure_channel("unix-abstract:aws.lookoutvision.inference-server")
stub = InferenceServerStub(channel)
stub.StartModel(pb2.StartModelRequest(model_component="ComponentCircuitBoard"))
