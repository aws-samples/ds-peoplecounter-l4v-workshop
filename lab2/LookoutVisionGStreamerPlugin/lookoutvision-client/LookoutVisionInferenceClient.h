// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. // SPDX-License-Identifier: MIT-0
#ifndef __LOOKOUTVISION_INFERENCE_CLIENT_H__
#define __LOOKOUTVISION_INFERENCE_CLIENT_H__

#include <glib.h>
#include "Inference.grpc.pb.h"
#include "gst/lookoutvisionmeta/gstlookoutvisionresult.h"

class LookoutVisionInferenceClient{
public:
    typedef enum _OperationStatus {
        SUCCESSFUL = 0,
        FAILED = -1
    } OperationStatus;

    LookoutVisionInferenceClient(std::string server_url);
    ~LookoutVisionInferenceClient();
    void setServerUrl(std::string server_url);
    GstLookoutVisionResult* DetectAnomalies(std::string model_component, guint8* frame, size_t bytes_size, size_t width,
                                            size_t height);
    OperationStatus StartModel(std::string model_component);
    OperationStatus StopModel(std::string model_component);

private:
    static const int POLLING_TIMEOUT_IN_SECONDS;
    static const int POLLING_INTERVAL_IN_SECONDS;
    static const std::string SHM_NAME;
    size_t shm_size = 1920*1920*3;
    int shm_fd;
    uint8_t* shm_data;
    std::string server_url;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<AWS::LookoutVision::Edge::InferenceServer::Stub> stub;

    void writeSHM(guint8* buf, size_t bytes_size);
    bool waitForModelStatusWithTimeout(std::string model_component,
                                       AWS::LookoutVision::Edge::ModelStatus expected_status, int timeout_in_seconds);
    AWS::LookoutVision::Edge::ModelStatus* getModelStatus(std::string model_component);
};

#endif
