#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "cuda failed " #expr ": err=%d(%s) line=%d\n",           \
              (int)_err, cudaGetErrorString(_err), __LINE__);                  \
      abort();                                                                 \
    }                                                                          \
  } while (0)

using namespace nvinfer1;
class Logger : public ILogger {
  void log(Severity severity, const char *msg) override {
    // suppress info-level messages
    if (severity <= Severity::kINFO)
      std::cout << msg << std::endl;
  }
} gLogger;

struct InferDeleter {
  template <typename T> void operator()(T *obj) const {
    if (obj != nullptr)
      obj->destroy();
  }
};

template <typename T> using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

int main() {
  int max_batch_size = 64;
  SampleUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(gLogger), {}};

  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(gLogger));
  nvinfer1::NetworkDefinitionCreationFlags flags;
  flags = 1 << static_cast<int>(
              nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(flags));
  auto parser = SampleUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, gLogger));
  auto parsed = parser->parseFromFile(
      "xxx.onnx", static_cast<int>(ILogger::Severity::kVERBOSE));
  assert(parsed);
  auto config =
      SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

  auto profile = builder->createOptimizationProfile();
  {
    auto input = network->getInput(0);
    auto dims = input->getDimensions();
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN,
                           Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(
        input->getName(), OptProfileSelector::kOPT,
        Dims4{max_batch_size, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(
        input->getName(), OptProfileSelector::kMAX,
        Dims4{max_batch_size, dims.d[1], dims.d[2], dims.d[3]});
  }
  config->addOptimizationProfile(profile);

  auto engine = SampleUniquePtr<nvinfer1::ICudaEngine>(
      builder->buildEngineWithConfig(*network, *config));
  assert(engine.get());
  auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(
      engine->createExecutionContextWithoutDeviceMemory());
  assert(context);
  auto size = engine->getDeviceMemorySize();
  void *workspace_ptr;
  CUDA_CHECK(cudaMalloc(&workspace_ptr, size));
  for (int i = 0; i < 300; ++i)
    context->setDeviceMemory(workspace_ptr);
  CUDA_CHECK(cudaFree(workspace_ptr));

  return 0;
}