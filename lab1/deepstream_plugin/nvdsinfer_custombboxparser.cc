#include <cstring>
#include <iostream>
#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <cmath>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))
#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

extern "C"
bool NvDsInferParseCustomGluonYoloV3 (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  static NvDsInferDims *cidLayerDims;
  static NvDsInferDims *scoreLayerDims;
  static NvDsInferDims *bboxLayerDims;
  static NvDsInferDims temp;
  static bool classMismatchWarn = false;
  int numClassesToParse;
  float *outputCidBuf = (float *) outputLayersInfo[0].buffer;
  float *outputCovBuf = (float *) outputLayersInfo[1].buffer;
  float *outputBboxBuf = (float *) outputLayersInfo[2].buffer;
  for (int c = 0; c < outputLayersInfo[0].inferDims.d[0]; c++)
  {
    float *outputX1 = outputBboxBuf + (c * 4);
    float *outputY1 = outputX1 + 1;
    float *outputX2 = outputX1 + 2;
    float *outputY2 = outputX1 + 3;

    float threshold = detectionParams.perClassPreclusterThreshold[c];
    if (outputCovBuf[c] >= threshold)
    {
      NvDsInferObjectDetectionInfo object;
      object.classId = outputCidBuf[c];
      object.detectionConfidence = outputCovBuf[c];

      /* Clip object box co-ordinates to network resolution */
      object.left = CLIP(*outputX1, 0, networkInfo.width - 1);
      object.top = CLIP(*outputY1, 0, networkInfo.height - 1);
      object.width = (*outputX2) - (*outputX1);
      object.height = (*outputY2) - (*outputY1);
      objectList.push_back(object);
    }
  }
  return true;
}
