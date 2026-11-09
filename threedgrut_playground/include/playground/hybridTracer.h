// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <cuda_runtime.h>
#include <optix.h>
#include <string>

#include <3dgrt/optixTracer.h>
#include <3dgrt/pipelineDefinitions.h>
#include <playground/cutexture.h>
#include <playground/pipelineDefinitions.h>
#include <playground/pipelineParameters.h>

// intermediate struct to pass pbr material data from python
// when tracing this struct gets converted to PBRMaterial, see
// playground_params.h
struct CPBRMaterial {
  unsigned int material_id;
  torch::Tensor diffuseMap;
  torch::Tensor emissiveMap;
  torch::Tensor metallicRoughnessMap;
  torch::Tensor normalMap;
  torch::Tensor diffuseFactor;
  torch::Tensor emissiveFactor;
  float metallicFactor;
  float roughnessFactor;
  unsigned int alphaMode;
  float alphaCutoff;
  float transmissionFactor;
  float ior;
};

struct Textures {
  std::shared_ptr<CudaTexture2DFloat4Object> diffuse;
  std::shared_ptr<CudaTexture2DFloat4Object> emissive;
  std::shared_ptr<CudaTexture2DFloat2Object> metallicRoughness;
  std::shared_ptr<CudaTexture2DFloat4Object> normal;
};

struct OptixDenoiserWrapper {
  OptixDenoiser _denoiser = nullptr;
  CUdeviceptr _denoiserScratchPtr = 0;
  size_t _denoiserScratchSz = 0;
  CUdeviceptr _denoiserStatePtr = 0;
  size_t _denoiserStateSz = 0;
  unsigned int _width = 0;
  unsigned int _height = 0;

  void setup(unsigned int width, unsigned int height,
             OptixDeviceContext optixContext, cudaStream_t stream);
  void release(cudaStream_t stream = 0);
};

class HybridOptixTracer : public OptixTracer {

protected:
  struct PlaygroundState {
    OptixDeviceContext context;
    OptixTraversableHandle
        gasHandle; /// Handle to the acceleration structure after it is built
    CUdeviceptr
        gasBuffer; /// Scratch buffer for building acceleration structure
    size_t gasBufferSz;
    CUdeviceptr
        gasBufferTmp; /// Holds the acceleration structure mem after it is built
    size_t gasBufferTmpSz;

    CUdeviceptr paramsDevice;
    size_t paramsDeviceSz;

    // The following are used in the construction of the BVH and tracing
    uint32_t vNum;        /// Number of mesh vertices
    uint32_t fNum;        /// Number of indexed mesh faces
    CUdeviceptr fPrimVrt; /// Buffer containing mesh vertices
    size_t fPrimVrtSz;
    CUdeviceptr fPrimTri; /// Buffer containing indexed triangles
    size_t fPrimTriSz;

    OptixPipeline pipelineTriGSTracing;
    OptixShaderBindingTable sbtTriGSTracing;
    OptixModule moduleTriGSTracing;

    std::vector<PBRMaterial> materials; /// dynamic array of materials, cached
                                        /// so we don't re-upload every frame
    std::vector<Textures> textures;     /// caches textures allocated on gpu

    // Denoiser
    OptixDenoiserWrapper denoiser;
  } *_playgroundState;

  // One Gaussian LOD slot (slot >= 1): its own instance AS and instance buffer. The unit-primitive GAS the instances
  // point to is the base tracer's shared one in _state->iasBuffer.
  struct LodSlot
  {
    OptixTraversableHandle handle = 0;
    CUdeviceptr gasBuffer = 0; /// IAS output
    size_t gasBufferSz = 0;
    CUdeviceptr instances = 0; /// OptixInstance array, the IAS build input
    size_t instancesSz = 0;
    OptixAabb aabb = {};
    uint32_t count = 0;      /// number of instances = length of the slot's index table
    CUdeviceptr unitGas = 0; /// _state->iasBuffer when the slot was built; the instances point into it
    float inflation = -1.0f; /// std the proxies were inflated by (the kernel's widening bound), -1 if not built
  };
  std::array<LodSlot, MaxLodPhysicalSlots> _lodSlots; // slots 1..MaxLodSlots-1, then the cap variants

  void releaseLodSlots(bool nothrow);

  void reallocateMeshPrimGeomBuffer(cudaStream_t stream);
  void reallocatePlaygroundParamsDevice(size_t sz, cudaStream_t stream);

public:
  HybridOptixTracer(
      const std::string &threedgrtPath, const std::string &playgroundPath,
      const std::string &cuda_path, const std::string &pipeline,
      const std::string &backwardPipeline, const std::string &primitive,
      float particleKernelDegree, float particleKernelMinResponse,
      float particleKernelMaxAlpha, bool particleKernelDensityClamping,
      int particleRadianceSphDegree, bool enableNormals, bool enableHitCounts);

  ~HybridOptixTracer() override;

  void buildMeshBVH(torch::Tensor meshVerts, torch::Tensor meshFaces,
                    unsigned int rebuild, bool allow_update);

  // Builds the instance AS of LOD slot `slot` (1 <= slot < MaxLodSlots) over the given Gaussians, into the slot's own
  // storage. `mogScl` holds the scales already inflated by `inflation`, which is kept with the slot: the kernel never
  // widens the slot's Gaussians beyond it. The main BVH (params.handle) is left untouched.
  void buildLodBVH(int slot, torch::Tensor mogPos, torch::Tensor mogRot,
                   torch::Tensor mogScl, torch::Tensor mogDns, float inflation);

  // Frees every LOD slot
  void clearLodBVH();

  // The inflation each slot 0..MaxLodSlots-1 was built with, or -1 for a slot that is not built or whose instances
  // point to a unit-primitive GAS that has moved (slot 0 is the main BVH: always -1)
  std::vector<float> lodSlotInflations() const;

  void syncMaterials(PlaygroundPipelineParameters &params,
                     const std::vector<CPBRMaterial> &materials,
                     cudaStream_t cudaStream);

  // Returns (rayRad, rayDns, rayHit, rayNrm, rayHitsCount, aovFirstHitT,
  // aovFirstNormal, aovFirstPrimType, aovFirstHitTransmittance, aovNumBounces,
  // aovGaussDepth)
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             torch::Tensor, torch::Tensor, torch::Tensor>
  traceHybrid(uint32_t frameNumber, torch::Tensor rayToWorld,
              torch::Tensor rayOri, torch::Tensor rayDir,
              torch::Tensor particleDensity, torch::Tensor particleFeatures,
              int sphDegree, float minTransmittance, torch::Tensor rayMaxT,
              uint32_t playgroundOpts, torch::Tensor triangles,
              torch::Tensor vNormals, torch::Tensor vTangents,
              torch::Tensor vHasTangents, torch::Tensor primType,
              torch::Tensor matUV, torch::Tensor matID,
              const std::vector<CPBRMaterial> &materials,
              bool shouldSyncMaterials, torch::Tensor refractiveIndex,
              torch::Tensor envmap, torch::Tensor envmapOffset,
              const unsigned int maxPBRBounces, torch::Tensor rayPixelId,
              torch::Tensor raySampleIndex, torch::Tensor coneParams,
              torch::Tensor lodParams, torch::Tensor lodNodeIndex,
              torch::Tensor pixelMode);

  torch::Tensor denoise(torch::Tensor rayRad);
};
