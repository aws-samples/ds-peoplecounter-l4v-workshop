#ifndef _NVDSINFER_CUSTOM_IMPL_H_
#define _NVDSINFER_CUSTOM_IMPL_H_

#include <string>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic pop

#include "nvdsinfer.h"


/*
 * C interfaces
 */

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * Holds the detection parameters required for parsing objects.
 */
typedef struct
{
  /** Holds the number of classes requested to be parsed, starting with
   class ID 0. Parsing functions may only output objects with
   class ID less than this value. */
  unsigned int numClassesConfigured;
  /** Holds a per-class vector of detection confidence thresholds
   to be applied prior to clustering operation.
   Parsing functions may only output an object with detection confidence
   greater than or equal to the vector element indexed by the object's
   class ID. */
  std::vector<float> perClassPreclusterThreshold;
  /* Per class threshold to be applied post clustering operation */
  std::vector<float> perClassPostclusterThreshold;

  /** Deprecated. Use perClassPreclusterThreshold instead. Reference to
   * maintain backward compatibility. */
  std::vector<float> &perClassThreshold = perClassPreclusterThreshold;
} NvDsInferParseDetectionParams;

/**
 * Type definition for the custom bounding box parsing function.
 *
 * @param[in]  outputLayersInfo A vector containing information on the output
 *                              layers of the model.
 * @param[in]  networkInfo      Network information.
 * @param[in]  detectionParams  Detection parameters required for parsing
 *                              objects.
 * @param[out] objectList       A reference to a vector in which the function
 *                              is to add parsed objects.
 */
typedef bool (* NvDsInferParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);

/**
 * Validates a custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static void checkFunc_ ## customParseFunc (NvDsInferParseCustomFunc func = customParseFunc) \
        { checkFunc_ ## customParseFunc (); }; \
    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
           NvDsInferNetworkInfo  const &networkInfo, \
           NvDsInferParseDetectionParams const &detectionParams, \
           std::vector<NvDsInferObjectDetectionInfo> &objectList);

/**
 * Type definition for the custom bounding box and instance mask parsing function.
 *
 * @param[in]  outputLayersInfo A vector containing information on the output
 *                              layers of the model.
 * @param[in]  networkInfo      Network information.
 * @param[in]  detectionParams  Detection parameters required for parsing
 *                              objects.
 * @param[out] objectList       A reference to a vector in which the function
 *                              is to add parsed objects and instance mask.
 */
typedef bool (* NvDsInferInstanceMaskParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferInstanceMaskInfo> &objectList);

/**
 * Validates a custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static void checkFunc_ ## customParseFunc (NvDsInferInstanceMaskParseCustomFunc func = customParseFunc) \
        { checkFunc_ ## customParseFunc (); }; \
    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
           NvDsInferNetworkInfo  const &networkInfo, \
           NvDsInferParseDetectionParams const &detectionParams, \
           std::vector<NvDsInferInstanceMaskInfo> &objectList);

/**
 * Type definition for the custom classifier output parsing function.
 *
 * @param[in]  outputLayersInfo  A vector containing information on the
 *                               output layers of the model.
 * @param[in]  networkInfo       Network information.
 * @param[in]  classifierThreshold
                                 Classification confidence threshold.
 * @param[out] attrList          A reference to a vector in which the function
 *                               is to add the parsed attributes.
 * @param[out] descString        A reference to a string object in which the
 *                               function may place a description string.
 */
typedef bool (* NvDsInferClassiferParseCustomFunc) (
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString);

/**
 * Validates the classifier custom parser function definition. Must be called
 * after defining the function.
 */
#define CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(customParseFunc) \
    static void checkFunc_ ## customParseFunc (NvDsInferClassiferParseCustomFunc func = customParseFunc) \
        { checkFunc_ ## customParseFunc (); }; \
    extern "C" bool customParseFunc (std::vector<NvDsInferLayerInfo> const &outputLayersInfo, \
           NvDsInferNetworkInfo  const &networkInfo, \
           float classifierThreshold, \
           std::vector<NvDsInferAttribute> &attrList, \
           std::string &descString);

typedef struct _NvDsInferContextInitParams NvDsInferContextInitParams;

#ifdef __cplusplus
}
#endif

#endif

