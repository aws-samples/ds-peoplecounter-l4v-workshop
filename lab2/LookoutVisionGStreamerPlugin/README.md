# Build Amazon Lookout For Vision(L4V) GStreamer plugins from source

On the edge device, clone this repo (https://github.com/aws-samples/ds-peoplecounter-l4v-workshop). Inside the repo root, run following commands

```
mkdir build
cd build
cmake -DUSE_SHARED_MEMORY=ON ..
make
```

This will create the binary libgstlookoutvision.so inside build directory. Upload it to s3 location 
s3://<BUCKETNAME>/artifacts/aws.greengrass.lookoutvision.GStreamerPlugin/1.0.0/libgstlookoutvision.so. Replace <BUCKETNAME> with your s3 bucket name.


* Create a private Greengrass component aws.greengrass.lookoutvision.GStreamerPlugin with this recipe from the recipe directory. Replace <BUCKETNAME> in recipe with the s3 bucket name used in previous step.
* Add the component aws.greengrass.lookoutvision.GStreamerPlugin to your Greengrass deployment. Once deployed, verify that it is in the FINISHED state

# Command
```
sudo /greengrass/v2/bin/greengrass-cli component list
```
# Command prints installed components (should contain aws.greengrass.lookoutvision.GStreamerPlugin)
```
Components currently running in Greengrass:
Component Name: aws.greengrass.lookoutvision.GStreamerPlugin
    Version: 1.0.0
    State: FINISHED
    Configuration: {}
```
* Consume the lookoutvision element in a custom GStreamer pipeline. Replace values for server-url and model-component properties of lookoutvision element.
```
gst-launch-1.0 -e videotestsrc num-buffers=1 pattern="ball" ! 'video/x-raw, format=RGB, width=1280, height=720' ! videoconvert ! lookoutvision server-url="unix-abstract:aws.lookoutvision.inference-server" model-component="test-without-gpu-code" ! videoconvert ! jpegenc ! filesink location=./anomaly.jpg --gst-plugin-path=/greengrass/v2/
```
# Output will contain inference result of the form
Is Anomalous? 1, Confidence: 0.52559


* Create a private Greengrass component aws.greengrass.lookoutvision.GStreamerPlugin with this recipe in mqtt-publish-sample. Replace <BUCKETNAME> in recipe with the s3 bucket name used in previous step.
* Add the component aws.greengrass.lookoutvision.GStreamerPlugin to your Greengrass deployment. Once deployed, verify that it is in the FINISHED state

# Command
sudo /greengrass/v2/bin/greengrass-cli component list

# Command prints installed components (should contain aws.greengrass.lookoutvision.GStreamerPlugin)
Components currently running in Greengrass:
Component Name: aws.greengrass.lookoutvision.GStreamerPlugin
    Version: 1.0.0
    State: FINISHED
    Configuration: {}

* Consume the lookoutvision element in a custom GStreamer pipeline. Replace values for server-url and model-component properties of lookoutvision element.

gst-launch-1.0 -e videotestsrc num-buffers=1 pattern="ball" ! 'video/x-raw, format=RGB, width=1280, height=720' ! videoconvert ! lookoutvision server-url="unix-abstract:aws.lookoutvision.inference-server" model-component="test-without-gpu-code" ! videoconvert ! jpegenc ! filesink location=./anomaly.jpg --gst-plugin-path=/greengrass/v2/

# Output will contain inference result of the form
Is Anomalous? 1, Confidence: 0.52559




