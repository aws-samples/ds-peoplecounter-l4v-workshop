
# Abstract
AWS continually evolves our edge computing offerings to provide customers with the technology they need to extend AWS services to edge devices, such as consumer products or manufacturing equipment, and enable them to act intelligently. This helps customers avoid unnecessary cost and latency, and empower customers with the ability to manage edge devices securely and efficiently.

You can use AWS IoT Greengrass to extend a wide range of AWS cloud technologies to your edge devices so they can act locally on the data they generate, while still using the cloud for real-time data analytics, data storage and visualization, and training and fine-tuning machine learning models. In addition to providing technology solutions from the edge to the cloud, AWS works with a variety of device providers across the globe to provide customers with the right hardware to choose from for their particular use case.

NVIDIA DeepStream SDK is an accelerated framework to build managed intelligent video analytics apps and services. The NVIDIA Jetson product family enables customers to extend server-class compute performance to devices operating at the edge. By using DeepStream in combination with TensorRT and CUDA on the Jetson platform , customers can build and deploy high throughput, low latency solutions.

In this Kmine artifact, we will demonstrate how you can integrate NVIDIA DeepStream on Jetson Modules with AWS IoT Services, so that you can start building innovative solutions with the AWS technologies and infrastructures to best meet your unique business requirements.

# Lab 1 - Build a People Counter

In this Lab, we have divided the lab into three parts. 

1. Prepare and compile people detecting model
2. Build shared library for SageMaker NEO specific nvinfer deepstream plug-in, and use this plugin to run the DeepStream application locally
3. Deploy and manage deepstream application using AWS IoT Greengrass

![Architecture Diagram](./static/NEO_deepstream.jpg)

---

## Pre-requisites
### Step 1: Set up device 

If there isn't a device set up at your table open, go to the back and grab a device, tripod, wifi adapter, and power adapter and connect them together! Also grab a handful of green aliens. Don't forget if you are putting a tripod on, make sure the swivel ball adjustment is attached so the camera can point down for Lab 2. If there are no more devices, you'll have to pair up with someone at your table to share the device.

Once you plug in the wifi adapter and the power bricker to your device, vice should boot up, note your mac address labeled on the wifi antenna. Go up front to the screen and you can find your device's IPv4 address. If you need help, raise your help and a workshop support person will help you by attaching a screen to the device.

### Step 2: Get your device's IP address

Let's get started by checking the device's IP address.

In front of each attendee is a NVIDIA Jetson Nano equipped with a camera. Check the bottom of the device for a sticker with the MAC address. If there is an IP address as well, you may use that. If not, write down the MAC address and check with the monitor up front for your IP address. You will SSH to the box.

### Step 3: Set up SSH on your computer

If you are running MacOS or Linux, you can use the command line ssh build in. If you are using windows, we recommend you install Putty (https://www.putty.org/)

### Step 4: Test your connection

Once you have SSH set up, and have your devices IP address, try to log in. The password is written at the front of the room.

:::code{showCopyAction=true showLineNumbers=false language=bash}
ssh nvidia@XXX.XXX.XXX
nvidia@XXX.XXX.XXX's password:
:::

---

## Part 1: People detection model preparation and compilation

In this example, we use the pretrained MXNet GluonCV YoloV3 model with mobilenet1.0 backbone that was pre-trained on COCO dataset with 80 classes as a people detector. In this step, we are going to compile the model using SageMaker NEO, and get a framework non-dependent output.

For the ease of the preparation, we have a notebook prepared for you. We recommend running this notebook on SageMaker notebook instance, so you do not install extra libraries on your local machine and you can easily shut the notebook instance off once you are done.

### Step 1: Create SageMaker Notebook instance
Navigate to SageMaker console, click on Notebook -> Notebook instances -> create new notebook instances. You can keep the default instance type, and name it however you want to, such as `iot-306-demo`.
During the notebook creation process, for GitHub repository, you can type in 
```
https://github.com/aws-samples/ds-peoplecounter-l4v-workshop.git
```
So that this repository that we have prepared for you readily available after the notebook instance starts.

### Step 2: Model preparation
The details steps for model preparation is included in the notebook `lab1/MXNetNeoPrep.ipynb (part 1)` in the GitHub. So please open navigate to this notebook once the instance is ready. The homepage should look like ![below](./static/notebook-instance-home.png). Please note that when you click on lab1, sometimes there is a few seconds of wait before content shows up, so don't be too alerted if it does not respond immediately. The top of the notebook should look like ![the screenshot](./static/model-prep-notebook.png) below.

You can run each code block in this notebook by hitting `shift + return`. And there is some explanation on each block on what the sample code is doing.

Please pause after part one. At the end of this section, you should have the prepared model for Greengrass deployment in the final S3 URL on your `MXNetNeoPrep.ipynb` notebook.

---

## Part 2: Compile shared library for SageMaker NEO specific nvinfer deepstream plug-in

In this part, we are going show how a DLR model can be used as part of an NVIDIA deepstream pipeline. We are going to leverage the customizable feature `Custom cuda engine creation` of [nvinfer component](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html) in DeepStream application.

### Step 1: Install DeepStream application
```
sudo apt-get update -y
sudo apt-get install -y libjson-glib-dev
sudo apt-get install -y libgstrtspserver-1.0-dev
sudo apt install -y \
libssl1.0.0 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstrtspserver-1.0-0 \
libjansson4=2.11-1 \
gcc \
make \
git \
python3
```
Navigate to Deepstream website, and download the tar file of [installation package](https://developer.nvidia.com/deepstream_sdk_v6.0.0_jetsontbz2). Or, to save time, we have uploaded this in a S3 bucket for us to download without signing in:
```
wget https://to-be-shared-in-general-1219-8488-4871.s3.us-west-2.amazonaws.com/deepstream_sdk_v6.0.0_jetson.tbz2
```
```
sudo tar -xvf deepstream_sdk_v6.0.0_jetson.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-6.0
sudo ./install.sh
sudo ldconfig
```

### Step 2: Clone neo-ai-dlr official release
```
cd ~
git clone https://github.com/neo-ai/neo-ai-dlr.git
cd neo-ai-dlr
git submodule update --init --recursive
```

### Step 3: Clone plug-in source code to Jetson Nano
Please run the following command on your Jetson Nano.
```
cd ~
git clone https://github.com/aws-samples/ds-peoplecounter-l4v-workshop.git
cp -r ds-peoplecounter-l4v-workshop/lab1/deepstream_plugin neo-ai-dlr/examples/
cd ~/neo-ai-dlr/examples/deepstream_plugin
make
cp libnvdsinfer_custom_impl_neodlr.so ~/ds-peoplecounter-l4v-workshop/lab1/deepstream-occupancy-analytics/
```

### Step 4: Download compiled model
Please run the following command on your Jetson Nano. Please note you can copy the first part directly from your AWS console log in page:
```
export ISENGARD_PRODUCTION_ACCOUNT=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export AWS_ACCESS_KEY_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export AWS_SESSION_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

sudo apt-get install -y awscli
cd ~
aws s3 cp <LINK PRINTED IN YOUR NOTEBOOK (last block in section 1, e.g. s3://xxxx/MXNet-yolo3-mobilenet10-Jetson-Nano-2021-11-27-23-57-11-656/model_zipped/model.zip)> ~/
unzip model.zip
```


### Step 5: Copy the deepstream application to deepstream sample source folder so we can have all the dependencies
```
sudo cp -r ~/ds-peoplecounter-l4v-workshop/lab1/deepstream-occupancy-analytics /opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/
cd /opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/deepstream-occupancy-analytics
```
Optional: For the convinience of this lab, the application execution file is already made available for you. But you can compile again by yourself:
```
sudo su
export CUDA_VER=10.2
make
```
You will see the libnvdsinfer_custom_impl_neodlr.so newly compiled.

### Step 6: Run deepstream application
In this demo, we have set the deepstream source to be a file source of the video feed of Ryan's front lawn. Let's move it to root folder, where we point `uri=file:///opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/deepstream-occupancy-analytics/ryanhouse-ds-ringvideo.mp4` in deepstream configuration file `config/test5_config_file_src_infer_tlt_neo.txt`. 

And for the sink, we configured it to be `out.mp4` stored in the same folder.

```
sudo su
export COMPILED_MODEL_PATH=/home/nvidia/model
./deepstream-test5-analytics -c config/test5_config_file_src_infer_tlt_neo.txt
```

At the beginning, you will observe the PERF measurement to be 0, the reason is that the NEO compiled model takes significantly longer to run inferencing against the first frame due to the initialization of the graph. After around 1-2 minutes, you will see the pipeline actually running.

Once the pipeline finishes, you can upload the `out.mp4` to cloud storage and download locally to have a look.

(If you are doing the in-person workshop with a device, you can skip the awscliv2 install step)
```
pushd ~/Downloads
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
popd
```
Continue here if you already have a configured device

```
export AWS_ACCESS_KEY_ID=XXXXX
export AWS_SECRET_ACCESS_KEY=XXXXXXXXXX
export AWS_SESSION_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
sudo apt-get install aws-cli
aws s3 cp out.mp4 s3://<YOUR_BUCKET>/deepstream-people-counter/output/out.mp4
```

---

## Part 3: Deploy and manage deepstream application using AWS IoT Greengrass

### Step 1: Package the necessary deepstream application files as zip

If you have not yet installed aws cli, please install now (you can skip this if you have a pre-configured device in person):
```
pushd ~/Downloads
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
popd
```
Then we zip up the necessary deepstream application files to be deployed and managed by AWS IoT Greengrass. For `<YOUR_BUCKET>`, for the convinience of this demo, please copy the default bucket that you use in the Jupyter notebook in step 1.
```
sudo apt-get install -y zip curl
cd /opt/nvidia/deepstream/deepstream-6.0/sources/apps/sample_apps/deepstream-occupancy-analytics
zip app.zip deepstream-test5-analytics config/* libnvdsinfer_custom_impl_neodlr.so

export AWS_ACCESS_KEY_ID=XXXXX
export AWS_SECRET_ACCESS_KEY=XXXXXXXXXX
export AWS_SESSION_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
aws s3 cp app.zip s3://<YOUR_BUCKET>/deepstream-people-counter/artifact/app.zip
aws s3 cp app.zip s3://sagemaker-us-west-2-121984884871/deepstream-people-counter/artifact/app.zip
sudo --preserve-env=COMPILED_MODEL_PATH /greengrass/v2/packages/artifacts-unarchived/iot-greengrass-managed-deepstream-application/1.0.8/app/deepstream-test5-analytics -c /greengrass/v2/packages/artifacts-unarchived/iot-greengrass-managed-deepstream-application/1.0.8/app/config/test5_config_file_src_infer_tlt_neo.txt
```

### Step 2: Provision SageMaker executive role with Greengrass permissions
In this step, we need to navigate to AWS IAM console, find the currently SageMaker role used by your notebook instance, and provides Greengrass permissions, so that we can create a Greengrass component from the notebook.

### Step 3: Construct a Greengrass recipe file
Pick up from where we left in the Jupyter notebook on SageMaker instance. Run from `section 2` on the notebook instance.

By the end of the notebook, you should have a component created on AWS IoT Greengrass service. You can navigate on AWS console to IoT -> Greengrass -> Components to see this component that we have created.

### Step 4: Prepare a sample video to run inference
cp ~/Downloads/ ds-peoplecounter-l4v-workshop/lab1/deepstream-occupancy-analytics/ryanhouse-ds-ringvideo.mp4 /tmp

### Step 5: Install Greengrass on Jetson Nano
We have prepared AWS IoT Greengrass installation scripts for you. This script automatically download Greengrass and install on Jetson device. At the same time, it also provisions the thing named `jetsonDeepstreamDemo` which belongs to `jetsonDeepstreamDemoGroup` on AWS IoT service, so it has the necessary certificates to authenticate with AWS services.
```
cd ~/Downloads/ds-peoplecounter-l4v-workshop
chmod 777 install_greengrass_jetson_nano.sh
./install_greengrass_jetson_nano.sh
vim /etc/sudoers
```
And add the following line
```
ggc_user ALL=(ALL) NOPASSWD:ALL
```

### Step 6: Deploy DeepStream with Greengrass
Continue on the SageMaker notebook section 2, create a Greengrass recipe, Greengrass component, Greengrass deployment.

Once this part is done, you can navigate to 
```
/greengrass/v2/logs/iot-greengrass-managed-deepstream-application.log
```

Please note that originally, the code will show a PERF of 0, because of the graph initialization, and later it is going to reach 12-14 FPS at peak performance.

### Step 7: Visualize your deepstream inferencing output results
Upon a successful run of this DeepStream application, you will see a file named `output.mp4` in the same folder. You can further scp this file to your local machine and use any video player to see the overlay bounding boxes on the people detected and the corresponding increase of the people counter when they cross the pre-determined lines. Below is a sample video file that you will see as the output of this entire lab.

![Example Output](out.mov)

# End of Lab 1
Now you have successfully brought your own NEO compiled people detection model in DeepStream pipeline. Feel free to change the line positions so that you can adapt to your own live video feed environments. What are some next steps? Maybe create a new DeepStream application with other object counting use cases. Have fun!