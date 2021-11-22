
#include <iostream>
#include <sstream>
#include <iomanip>
#include "neodlr_plugin.h"
#include "nvdsinfer_context.h"
#include "nvdsinfer_custom_impl.h"

// #define COMPILED_MODEL_PATH '/home/yuxin/Downloads/model'
#define PICTURE_HEIGHT 224
#define PICTURE_WIDTH 224

extern "C" bool NvDsInferNeoDlrCudaEngineGet(nvinfer1::IBuilder* const builder,
                                             const NvDsInferContextInitParams* const initParams,
                                             nvinfer1::DataType dataType,
                                             nvinfer1::ICudaEngine*& cudaEngine);

extern "C" bool NvDsInferNeoDlrCudaEngineGet(nvinfer1::IBuilder* const builder,
                                             const NvDsInferContextInitParams* const initParams,
                                             nvinfer1::DataType dataType,
                                             nvinfer1::ICudaEngine*& cudaEngine) {
  nvinfer1::INetworkDefinition* network = builder->createNetwork();

  // TODO: Edit input shapes to match your model.
  nvinfer1::ITensor* input =
      network->addInput("data", nvinfer1::DataType::kFLOAT, nvinfer1::DimsCHW(3, PICTURE_HEIGHT, PICTURE_WIDTH));
  std::vector<nvinfer1::ITensor*> inputs = {input};

  // Create DLR plugin. Set path to compiled model (folder containing .json, .params, .so, libdlr.so)
  auto* dlr_plugin = new NeoDLRLayer(std::getenv("COMPILED_MODEL_PATH"));
  nvinfer1::IPluginV2Layer* dlr_layer =
      network->addPluginV2(inputs.data(), inputs.size(), *dlr_plugin);
  // nvinfer1::IPluginV2Layer* dlr_layer =
  //     network->addPluginV2(&input, 1, *dlr_plugin);

  // Mark output(s)
  for (int i = 0; i < dlr_layer->getNbOutputs(); i++) {
    std::cout << "output" << i << "\n";
    nvinfer1::ITensor* output = dlr_layer->getOutput(i);
    std::string output_name = "dlr_output_" + std::to_string(i);
    output->setName(output_name.c_str());
    network->markOutput(*output);
  }

  cudaEngine = builder->buildCudaEngine(*network);
  network->destroy();
  return true;
}

