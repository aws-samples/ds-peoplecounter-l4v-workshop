import time
import numpy as np
import cv2
import grpc
import edge_agent_pb2 as pb2 
from edge_agent_pb2_grpc import ( 
    EdgeAgentStub
)  
import sys
from awscrt import io, mqtt, auth, http
from awsiot import mqtt_connection_builder
import time as t
import json

ENDPOINT = "<your_iot_endpoint_here>"
CLIENT_ID = "<your_thing_name>"
PATH_TO_CERTIFICATE = "/greengrass/v2/thingCert.crt"
PATH_TO_PRIVATE_KEY = "/greengrass/v2/privKey.key"
PATH_TO_AMAZON_ROOT_CA_1 = "/greengrass/v2/rootCA.pem"
TOPIC = "l4v/testclient"

img = cv2.imread(sys.argv[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('frame.bmp',img) # this lets you verify the image compared is what you think it was
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
   event_loop_group = io.EventLoopGroup(1)
   host_resolver = io.DefaultHostResolver(event_loop_group)
   client_bootstrap = io.ClientBootstrap(event_loop_group, host_resolver)
   mqtt_connection = mqtt_connection_builder.mtls_from_path(
                    endpoint=ENDPOINT,
                    cert_filepath=PATH_TO_CERTIFICATE,
                    pri_key_filepath=PATH_TO_PRIVATE_KEY,
                    client_bootstrap=client_bootstrap,
                    ca_filepath=PATH_TO_AMAZON_ROOT_CA_1,
                    client_id=CLIENT_ID,
                    clean_session=False,
                    keep_alive_secs=6
                    )
   print("Connecting to {} with client ID '{}'...".format(
            ENDPOINT, CLIENT_ID))
   # Make the connect() call
   connect_future = mqtt_connection.connect()
   # Future.result() waits until a result is available
   connect_future.result()
   print("Connected!")
   # Publish message to server desired number of times.
   print('Begin Publish')
   data = "{} [{}]".format(str(response),1)
   message = {"message" : data, "is_anomalous": response.detect_anomaly_result.is_anomalous }
   mqtt_connection.publish(topic=TOPIC, payload=json.dumps(message), qos=mqtt.QoS.AT_LEAST_ONCE)
   print("Published: '" + json.dumps(message) + "' to the topic: " + TOPIC)
   t.sleep(0.1)
   print('Publish End')
   disconnect_future = mqtt_connection.disconnect()
   disconnect_future.result()

   print("is_anomalous:"+str(response.detect_anomaly_result.is_anomalous))
   print(str(response))

