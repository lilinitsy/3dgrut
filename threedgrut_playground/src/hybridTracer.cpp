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

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push, 0)
#include <torch/extension.h>
#pragma warning(pop)
#else
#include <torch/extension.h>
#endif

#include <3dgrt/cuoptixMacros.h>
#include <3dgrt/optixTracer.h>
#include <3dgrt/particlePrimitives.h>
#include <3dgrt/pipelineParameters.h>
#include <3dgrt/tensorBuffering.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <memory>
#include <nvrtc.h>
#include <optix.h>
#include <playground/hybridTracer.h>
#include <playground/meshBuffers.h>
#include <playground/pipelineParameters.h>

//------------------------------------------------------------------------------
// HybridOptixTracer
//------------------------------------------------------------------------------

HybridOptixTracer::HybridOptixTracer(
    const std::string &threedgrtPath, const std::string &playgroundPath,
    const std::string &cuda_path, const std::string &pipeline,
    const std::string &backwardPipeline, const std::string &primitive,
    float particleKernelDegree, float particleKernelMinResponse,
    float particleKernelMaxAlpha, bool particleKernelDensityClamping,
    int particleRadianceSphDegree, bool enableNormals, bool enableHitCounts)
    : OptixTracer(threedgrtPath, cuda_path, pipeline, backwardPipeline,
                  primitive, particleKernelDegree, particleKernelMinResponse,
                  particleKernelMaxAlpha, particleKernelDensityClamping,
                  particleRadianceSphDegree, enableNormals, enableHitCounts) {

  char log[2048]; // For error reporting from OptiX creation functions
  std::vector<std::string> defines = generateDefines(
      particleKernelDegree, particleKernelDensityClamping,
      particleRadianceSphDegree, enableNormals, enableHitCounts);
  const uint32_t sharedFlags =
      (_state->gPrimType == MOGTracingSphere
           ? PipelineFlag_SpherePrim
           : ((_state->gPrimType == MOGTracingCustom) ||
                      (_state->gPrimType == MOGTracingInstances)
                  ? PipelineFlag_HasIS
                  : 0));

  _playgroundState = new PlaygroundState();
  memset(_playgroundState, 0, sizeof(PlaygroundState));
  // Use same context as Gaussians program
  _playgroundState->context = _state->context;
  _playgroundState->fNum = 0;
  _playgroundState->vNum = 0;

  _playgroundState->moduleTriGSTracing = nullptr;
  _playgroundState->pipelineTriGSTracing = nullptr;
  _playgroundState->sbtTriGSTracing = {};
  createPipeline(_playgroundState->context, playgroundPath, threedgrtPath,
                 cuda_path, defines, "playgroundKernel",
                 sharedFlags | PipelineFlag_HasRG | PipelineFlag_HasCH |
                     PipelineFlag_HasAH | PipelineFlag_HasMS,
                 &_playgroundState->moduleTriGSTracing,
                 &_playgroundState->pipelineTriGSTracing,
                 _playgroundState->sbtTriGSTracing, 32,
                 {threedgrtPath + "/include",
                  playgroundPath + "/src/kernels/cuda/"}, // extra includes
                 true                                     // hybridMeshParticles
  );
}

HybridOptixTracer::~HybridOptixTracer(void) {

  if (_playgroundState)
    _playgroundState->denoiser.release(); // Release if allocated
  releaseLodSlots(true);

  OPTIX_CHECK_NOTHROW(
      optixPipelineDestroy(_playgroundState->pipelineTriGSTracing));
  CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void *>(
      _playgroundState->sbtTriGSTracing.raygenRecord)));
  CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void *>(
      _playgroundState->sbtTriGSTracing.missRecordBase)));
  CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void *>(
      _playgroundState->sbtTriGSTracing.hitgroupRecordBase)));
  OPTIX_CHECK_NOTHROW(optixModuleDestroy(_playgroundState->moduleTriGSTracing));

  CUDA_CHECK_NOTHROW(
      cudaFree(reinterpret_cast<void *>(_playgroundState->fPrimVrt)));
  CUDA_CHECK_NOTHROW(
      cudaFree(reinterpret_cast<void *>(_playgroundState->fPrimTri)));
  CUDA_CHECK_NOTHROW(
      cudaFree(reinterpret_cast<void *>(_playgroundState->gasBuffer)));
  CUDA_CHECK_NOTHROW(
      cudaFree(reinterpret_cast<void *>(_playgroundState->gasBufferTmp)));
  CUDA_CHECK_NOTHROW(
      cudaFree(reinterpret_cast<void *>(_playgroundState->paramsDevice)));

  delete _playgroundState;
}

void HybridOptixTracer::releaseLodSlots(bool nothrow)
{
  for (LodSlot &slot : _lodSlots)
  {
    if (nothrow)
    {
      CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void *>(slot.gasBuffer)));
      CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void *>(slot.instances)));
    }
    else
    {
      cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();
      CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(slot.gasBuffer), cudaStream));
      CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void *>(slot.instances), cudaStream));
    }
    slot = LodSlot();
  }
}

void HybridOptixTracer::clearLodBVH() { releaseLodSlots(false); }

std::vector<float> HybridOptixTracer::lodSlotInflations() const
{
  std::vector<float> inflations(MaxLodPhysicalSlots, -1.0f);
  for (int s = 1; s < MaxLodPhysicalSlots; s++)
  {
    const LodSlot &slot = _lodSlots[s];
    if ((slot.handle != 0) && (slot.unitGas == _state->iasBuffer)) inflations[s] = slot.inflation;
  }
  return inflations;
}

// Builds slot `slot` with the base buildBVH, by swapping the slot's storage into _state for the call and the main BVH
// back afterwards. Per slot: the IAS (gasHandle, gasBuffer), its instance buffer (gPrimAABB), its bounds (gasAABB) and
// gNum, so a later refit of the main BVH still sees its own count. Shared: the scratch buffer gasBufferTmp, and the
// unit-primitive GAS in iasBuffer, which every buildBVH rebuilds in place. Its size depends only on the fixed one-AABB
// build input, so it is never reallocated after the first build and the instances of earlier slots stay valid;
// traceHybrid still checks that iasBuffer has not moved. Everything runs on the current stream, so no trace overlaps a
// rebuild. Rebuilding a built slot reuses its buffers when they are large enough.
void HybridOptixTracer::buildLodBVH(int slot, torch::Tensor mogPos, torch::Tensor mogRot, torch::Tensor mogScl, torch::Tensor mogDns, float inflation)
{
  TORCH_CHECK(slot >= 1 && slot < MaxLodPhysicalSlots, "LOD slot must be in [1, ", MaxLodPhysicalSlots, "), got ", slot);
  TORCH_CHECK(std::isfinite(inflation) && inflation >= 0.0f, "LOD slot ", slot, " inflation must be finite and >= 0, got ", inflation);
  TORCH_CHECK(_state->gPrimType == MOGTracingInstances, "LOD slots need the 'instances' primitive type");
  TORCH_CHECK(mogPos.size(0) > 0, "LOD slot ", slot, " has no Gaussians");
  LodSlot &lodSlot = _lodSlots[slot];
  auto swapSlot = [&]()
  {
    std::swap(_state->gasHandle, lodSlot.handle);
    std::swap(_state->gasBuffer, lodSlot.gasBuffer);
    std::swap(_state->gasBufferSz, lodSlot.gasBufferSz);
    std::swap(_state->gPrimAABB, lodSlot.instances);
    std::swap(_state->gPrimAABBSz, lodSlot.instancesSz);
    std::swap(_state->gasAABB, lodSlot.aabb);
    std::swap(_state->gNum, lodSlot.count);
  };
  swapSlot();
  try
  {
    buildBVH(mogPos, mogRot, mogScl, mogDns, 1, false);
    // The bounds reach the host by an async copy: finish it before _state->gasAABB goes back to the main BVH
    CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  }
  catch (...)
  {
    swapSlot();
    lodSlot.handle = 0;
    lodSlot.count = 0;
    lodSlot.inflation = -1.0f;
    throw;
  }
  swapSlot();
  lodSlot.unitGas = _state->iasBuffer;
  lodSlot.inflation = inflation;
}

void HybridOptixTracer::reallocateMeshPrimGeomBuffer(cudaStream_t cudaStream) {
  if (_playgroundState) {
    reallocateBuffer(&_playgroundState->fPrimVrt, _playgroundState->fPrimVrtSz,
                     sizeof(float3) * 3 * _playgroundState->fNum, cudaStream);
    reallocateBuffer(&_playgroundState->fPrimTri, _playgroundState->fPrimTriSz,
                     sizeof(int3) * _playgroundState->fNum, cudaStream);
  }
}

void HybridOptixTracer::reallocatePlaygroundParamsDevice(
    size_t sz, cudaStream_t cudaStream) {
  if (_playgroundState) {
    reallocateBuffer(&_playgroundState->paramsDevice,
                     _playgroundState->paramsDeviceSz, sz, cudaStream);
  }
}

void HybridOptixTracer::buildMeshBVH(torch::Tensor meshVerts,
                                     torch::Tensor meshFaces,
                                     unsigned int rebuild, bool allow_update) {

  const uint32_t vNum = meshVerts.size(0);
  const uint32_t fNum = meshFaces.size(0);
  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  if (!rebuild && (_playgroundState->fNum != fNum)) {
    std::cerr
        << "ERROR:: cannot refit GAS with a different number of mesh faces"
        << std::endl;
    rebuild = 1;
  }
  cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

  _playgroundState->vNum = vNum;
  _playgroundState->fNum = fNum;
  reallocateMeshPrimGeomBuffer(cudaStream);

  computeMeshFaceBuffer(
      fNum, getPtr<float3>(meshVerts), getPtr<int3>(meshFaces),
      reinterpret_cast<float3 *>(_playgroundState->fPrimVrt),
      reinterpret_cast<int3 *>(_playgroundState->fPrimTri), cudaStream);
  CUDA_CHECK_LAST();

  {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                               OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    if (allow_update) {
      accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    }
    accel_options.operation =
        rebuild ? OPTIX_BUILD_OPERATION_BUILD : OPTIX_BUILD_OPERATION_UPDATE;

    // Our build input is a simple list of non-indexed triangle vertices
    OptixBuildInput prim_input = {};
    uint32_t prim_input_flags = {
        OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    prim_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    prim_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    prim_input.triangleArray.numVertices = vNum;
    prim_input.triangleArray.vertexBuffers = &_playgroundState->fPrimVrt;
    prim_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    prim_input.triangleArray.numIndexTriplets = fNum;
    prim_input.triangleArray.indexBuffer = _playgroundState->fPrimTri;
    prim_input.triangleArray.flags = &prim_input_flags;
    prim_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(_playgroundState->context,
                                             &accel_options, &prim_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));
    if (_playgroundState->gasBufferTmpSz < gas_buffer_sizes.tempSizeInBytes) {
      CUDA_CHECK(cudaFreeAsync(
          reinterpret_cast<void *>(_playgroundState->gasBufferTmp),
          cudaStream));
      CUDA_CHECK(cudaMallocAsync(
          reinterpret_cast<void **>(&_playgroundState->gasBufferTmp),
          gas_buffer_sizes.tempSizeInBytes, cudaStream));
      _playgroundState->gasBufferTmpSz = gas_buffer_sizes.tempSizeInBytes;
    }

    if (rebuild &&
        (_playgroundState->gasBufferSz < gas_buffer_sizes.outputSizeInBytes)) {
      CUDA_CHECK(cudaFreeAsync(
          reinterpret_cast<void *>(_playgroundState->gasBuffer), cudaStream));
      CUDA_CHECK(cudaMallocAsync(
          reinterpret_cast<void **>(&_playgroundState->gasBuffer),
          gas_buffer_sizes.outputSizeInBytes, cudaStream));
      _playgroundState->gasBufferSz = gas_buffer_sizes.outputSizeInBytes;
    }

    OPTIX_CHECK(optixAccelBuild(
        _playgroundState->context,
        cudaStream, // CUDA stream
        &accel_options, &prim_input,
        1, // num build inputs
        _playgroundState->gasBufferTmp, gas_buffer_sizes.tempSizeInBytes,
        _playgroundState->gasBuffer, gas_buffer_sizes.outputSizeInBytes,
        &_playgroundState->gasHandle,
        nullptr, // emitted property list
        0        // num emitted properties
        ));
  }

  CUDA_CHECK_LAST();
}

void HybridOptixTracer::syncMaterials(
    PlaygroundPipelineParameters &params,
    const std::vector<CPBRMaterial> &materials, cudaStream_t cudaStream) {
  size_t numMaterials = materials.size();

  // Material Textures
  std::vector<CudaTexture2DFloat4Object> cuDiffuseTexture(numMaterials);
  std::vector<CudaTexture2DFloat4Object> cuEmissiveTexture(numMaterials);
  std::vector<CudaTexture2DFloat2Object> cuMetallicRoughnessTexture(
      numMaterials);
  std::vector<CudaTexture2DFloat4Object> cuNormalTexture(numMaterials);

  _playgroundState->materials.resize(numMaterials);
  _playgroundState->textures.resize(numMaterials);

  for (int i = 0; i < numMaterials; ++i) {

    auto &texture = _playgroundState->textures[i];

    torch::Tensor texDiffuse = materials[i].diffuseMap;
    int diffuseTexHeight = texDiffuse.size(0);
    int diffuseTexWidth = texDiffuse.size(1);
    texture.diffuse = std::make_shared<CudaTexture2DFloat4Object>();
    if (diffuseTexHeight > 0 && diffuseTexWidth > 0) {
      texture.diffuse->reset(texDiffuse.data_ptr<float>(), diffuseTexHeight,
                             diffuseTexWidth);
    }

    torch::Tensor texEmissive = materials[i].emissiveMap;
    int emissiveTexHeight = texEmissive.size(0);
    int emissiveTexWidth = texEmissive.size(1);
    texture.emissive = std::make_shared<CudaTexture2DFloat4Object>();
    if (emissiveTexHeight > 0 && emissiveTexWidth > 0) {
      texture.emissive->reset(texEmissive.data_ptr<float>(), emissiveTexHeight,
                              emissiveTexWidth);
    }

    torch::Tensor texMetallicRoughness = materials[i].metallicRoughnessMap;
    int metallicRoughnessTexHeight = texMetallicRoughness.size(0);
    int metallicRoughnessTexWidth = texMetallicRoughness.size(1);
    texture.metallicRoughness = std::make_shared<CudaTexture2DFloat2Object>();
    if (metallicRoughnessTexHeight > 0 && metallicRoughnessTexWidth > 0) {
      texture.metallicRoughness->reset(texMetallicRoughness.data_ptr<float>(),
                                       metallicRoughnessTexHeight,
                                       metallicRoughnessTexWidth);
    }

    torch::Tensor texNormal = materials[i].normalMap;
    int normalTexHeight = texNormal.size(0);
    int normalTexWidth = texNormal.size(1);
    texture.normal = std::make_shared<CudaTexture2DFloat4Object>();
    if (normalTexHeight > 0 && normalTexWidth > 0) {
      texture.normal->reset(texNormal.data_ptr<float>(), normalTexHeight,
                            normalTexWidth);
    }

    PBRMaterial &hostMat = _playgroundState->materials[i];
    hostMat.useDiffuseTexture = texture.diffuse->isTexInitialized();
    hostMat.diffuseTexture = texture.diffuse->tex();
    hostMat.useEmissiveTexture = texture.emissive->isTexInitialized();
    hostMat.emissiveTexture = texture.emissive->tex();
    hostMat.useMetallicRoughnessTexture =
        texture.metallicRoughness->isTexInitialized();
    hostMat.metallicRoughnessTexture = texture.metallicRoughness->tex();
    hostMat.useNormalTexture = texture.normal->isTexInitialized();
    hostMat.normalTexture = texture.normal->tex();
    float *diffuseFactor = materials[i].diffuseFactor.data_ptr<float>();
    hostMat.diffuseFactor = make_float4(diffuseFactor[0], diffuseFactor[1],
                                        diffuseFactor[2], diffuseFactor[3]);
    float *emissiveFactor = materials[i].emissiveFactor.data_ptr<float>();
    hostMat.emissiveFactor =
        make_float3(emissiveFactor[0], emissiveFactor[1], emissiveFactor[2]);
    hostMat.metallicFactor = materials[i].metallicFactor;
    hostMat.roughnessFactor = materials[i].roughnessFactor;
    hostMat.alphaMode = materials[i].alphaMode;
    hostMat.alphaCutoff = materials[i].alphaCutoff;
    hostMat.transmissionFactor = materials[i].transmissionFactor;
    hostMat.ior = materials[i].ior;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
HybridOptixTracer::traceHybrid(
    uint32_t frameNumber, torch::Tensor rayToWorld, torch::Tensor rayOri,
    torch::Tensor rayDir, torch::Tensor particleDensity,
    torch::Tensor particleFeatures, int sphDegree, float minTransmittance,
    torch::Tensor rayMaxT, uint32_t playgroundOpts, torch::Tensor triangles,
    torch::Tensor vNormals, torch::Tensor vTangents, torch::Tensor vHasTangents,
    torch::Tensor primType, torch::Tensor matUV, torch::Tensor matID,
    const std::vector<CPBRMaterial> &materials, bool shouldSyncMaterials,
    torch::Tensor refractiveIndex, torch::Tensor envmap,
    torch::Tensor envmapOffset, const unsigned int maxPBRBounces,
    torch::Tensor rayPixelId, torch::Tensor raySampleIndex,
    torch::Tensor coneParams, torch::Tensor lodParams,
    torch::Tensor lodNodeIndex, torch::Tensor pixelMode) {

  // ----- 3dgrt launch params -----
  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  const torch::TensorOptions int_opts =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor rayRad =
      torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
  torch::Tensor rayDns =
      torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
  torch::Tensor rayHit =
      torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 2}, opts);
  torch::Tensor rayNrm =
      torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
  torch::Tensor rayHitsCount =
      torch::zeros({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
  torch::Tensor traceState = torch::zeros(
      {rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, int_opts);
  // Per-ray AOVs: every launch slot writes all of them, so empty is safe
  torch::Tensor aovFirstHitT = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
  torch::Tensor aovFirstNormal = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 3}, opts);
  torch::Tensor aovFirstPrimType = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, int_opts);
  torch::Tensor aovFirstHitTransmittance = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);
  torch::Tensor aovNumBounces = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, int_opts);
  torch::Tensor aovGaussDepth = torch::empty({rayOri.size(0), rayOri.size(1), rayOri.size(2), 1}, opts);

  PlaygroundPipelineParameters paramsHost;
  paramsHost.handle = _state->gasHandle;
  paramsHost.aabb = _state->gasAABB;

  paramsHost.frameBounds.x = rayOri.size(2) - 1;
  paramsHost.frameBounds.y = rayOri.size(1) - 1;
  paramsHost.frameNumber = frameNumber;
  paramsHost.gPrimNumTri = _state->gPrimNumTri;

  paramsHost.minTransmittance = minTransmittance;
  paramsHost.hitMinGaussianResponse = _state->particleKernelMinResponse;
  paramsHost.alphaMinThreshold = 1.0f / 255.0f;
  paramsHost.sphDegree = sphDegree;

  std::memcpy(&paramsHost.rayToWorld[0].x, rayToWorld.cpu().data_ptr<float>(),
              3 * sizeof(float4));
  paramsHost.rayOrigin = packed_accessor32<float, 4>(rayOri);
  paramsHost.rayDirection = packed_accessor32<float, 4>(rayDir);

  paramsHost.particleDensity = getPtr<const ParticleDensity>(particleDensity);
  paramsHost.particleFeatures =
      getPtr<const TParticleFeatureElem>(particleFeatures);
  paramsHost.particleExtendedData =
      reinterpret_cast<const void *>(_state->gPipelineParticleData);

  paramsHost.rayFeatures = packed_accessor32<TRayFeatureElem, 4>(rayRad);
  paramsHost.rayDensity = packed_accessor32<float, 4>(rayDns);
  paramsHost.rayHitDistance = packed_accessor32<float, 4>(rayHit);
  paramsHost.rayNormal = packed_accessor32<float, 4>(rayNrm);
  paramsHost.rayHitsCount = packed_accessor32<float, 4>(rayHitsCount);

  // ----- Playground launch params -----
  paramsHost.rayMaxT = packed_accessor32<float, 3>(rayMaxT);
  paramsHost.triangles = packed_accessor32<int32_t, 2>(triangles);
  paramsHost.vNormals = packed_accessor32<float, 2>(vNormals);
  paramsHost.vTangents = packed_accessor32<float, 2>(vTangents);
  paramsHost.vHasTangents = packed_accessor32<bool, 2>(vHasTangents);
  paramsHost.primType = packed_accessor32<int32_t, 2>(primType);
  paramsHost.matUV = packed_accessor32<float, 3>(matUV);
  paramsHost.matID = packed_accessor32<int32_t, 2>(matID);
  paramsHost.refractiveIndex = packed_accessor32<float, 2>(refractiveIndex);
  paramsHost.maxPBRBounces = maxPBRBounces;
  paramsHost.playgroundOpts = playgroundOpts;
  paramsHost.triHandle = _playgroundState->gasHandle;
  paramsHost.trace_state = packed_accessor32<int32_t, 4>(traceState);
  paramsHost.useRayIds = rayPixelId.numel() > 0 ? 1u : 0u;
  paramsHost.rayPixelId = packed_accessor32<int32_t, 3>(rayPixelId);
  paramsHost.raySampleIndex = packed_accessor32<int32_t, 3>(raySampleIndex);
  paramsHost.aovFirstHitT = packed_accessor32<float, 4>(aovFirstHitT);
  paramsHost.aovFirstNormal = packed_accessor32<float, 4>(aovFirstNormal);
  paramsHost.aovFirstPrimType = packed_accessor32<int32_t, 4>(aovFirstPrimType);
  paramsHost.aovFirstHitTransmittance = packed_accessor32<float, 4>(aovFirstHitTransmittance);
  paramsHost.aovNumBounces = packed_accessor32<int32_t, 4>(aovNumBounces);
  paramsHost.aovGaussDepth = packed_accessor32<float, 4>(aovGaussDepth);
  TORCH_CHECK(coneParams.numel() == 5 && coneParams.scalar_type() == torch::kFloat32 && coneParams.device().is_cpu(), "coneParams must be a CPU float32 tensor with 5 elements");
  const float *cone = coneParams.data_ptr<float>();
  paramsHost.cone.pixelSpread = cone[0];
  paramsHost.cone.spreadPerAlpha = cone[1];
  paramsHost.cone.maxSigma = cone[2];
  paramsHost.cone.maxRoughness = cone[3];
  paramsHost.cone.splitColumn = (int)cone[4];
  // pixelMode: empty (off) or a [B,H,W] CUDA float32 tensor matching the rays, see PlaygroundPipelineParameters
  if (pixelMode.numel() == 0)
  {
    paramsHost.pixelMode = nullptr;
  }
  else
  {
    TORCH_CHECK(pixelMode.scalar_type() == torch::kFloat32 && pixelMode.is_cuda() && pixelMode.is_contiguous(), "pixelMode must be a contiguous CUDA float32 tensor");
    TORCH_CHECK(pixelMode.dim() == 3 && pixelMode.size(0) == rayOri.size(0) && pixelMode.size(1) == rayOri.size(1) && pixelMode.size(2) == rayOri.size(2), "pixelMode must be [B,H,W] like the rays");
    paramsHost.pixelMode = pixelMode.data_ptr<float>();
  }
  // lodParams: [enabled, S, band lower edges (S), index table offsets (S)], optionally followed by
  // [M, (cap, band slot) x M] cap variants, see LodParams. Each slot's inflation comes from the slot itself
  // (buildLodBVH), so the widening bound is always the one its proxies were built with
  TORCH_CHECK(lodParams.dim() == 1 && lodParams.numel() >= 2 && lodParams.scalar_type() == torch::kFloat32 && lodParams.device().is_cpu() && lodParams.is_contiguous(), "lodParams must be a contiguous 1D CPU float32 tensor with at least 2 elements");
  const float *lod = lodParams.data_ptr<float>();
  // Range-check S as a float first: casting a NaN or out-of-range float to int is undefined
  TORCH_CHECK(std::isfinite(lod[1]) && lod[1] >= 0.0f && lod[1] <= (float)MaxLodSlots && lod[1] == std::floor(lod[1]), "lodParams S must be an integer in [0, ", MaxLodSlots, "], got ", lod[1]);
  const int numLodSlots = (int)lod[1];
  const int64_t baseNumel = 2 + 2 * numLodSlots;
  int numCapVariants = 0;
  if (lodParams.numel() > baseNumel)
  {
    const float m = lod[baseNumel];
    TORCH_CHECK(std::isfinite(m) && m >= 0.0f && m <= (float)MaxLodCapVariants && m == std::floor(m), "lodParams M must be an integer in [0, ", MaxLodCapVariants, "], got ", m);
    numCapVariants = (int)m;
  }
  TORCH_CHECK(lodParams.numel() == baseNumel || lodParams.numel() == baseNumel + 1 + 2 * numCapVariants, "lodParams must hold 2 + 2 * S elements, or 3 + 2 * S + 2 * M with cap variants; got S = ", numLodSlots, " and ", lodParams.numel(), " elements");
  std::memset(&paramsHost.lod, 0, sizeof(LodParams));
  paramsHost.lod.enabled = lod[0] != 0.0f ? 1 : 0;
  if (paramsHost.lod.enabled)
  {
    TORCH_CHECK(numLodSlots >= 1, "enabled lodParams need at least one slot");
    TORCH_CHECK(lodNodeIndex.scalar_type() == torch::kInt32 && lodNodeIndex.is_cuda() && lodNodeIndex.is_contiguous(), "lodNodeIndex must be a contiguous CUDA int32 tensor");
    paramsHost.lod.numSlots = numLodSlots;
    paramsHost.lod.nodeIndex = lodNodeIndex.data_ptr<int32_t>();
    for (int s = 0; s < numLodSlots; s++)
    {
      const float lower = lod[2 + s];
      TORCH_CHECK(!std::isnan(lower), "LOD slot ", s, " band lower edge is NaN");
      paramsHost.lod.bandLower[s] = lower;
      if (s == 0) continue;
      const LodSlot &slot = _lodSlots[s];
      TORCH_CHECK(slot.handle != 0, "LOD slot ", s, " is enabled but not built");
      TORCH_CHECK(slot.unitGas == _state->iasBuffer, "LOD slot ", s, " points to a unit-primitive GAS that has moved; rebuild it");
      const float offset = lod[2 + numLodSlots + s];
      TORCH_CHECK(std::isfinite(offset) && offset >= 0.0f && offset == std::floor(offset) && (double)offset + slot.count <= (double)lodNodeIndex.numel(), "LOD slot ", s, " index table [", offset, ", ", offset, " + ", slot.count, ") is outside lodNodeIndex (", lodNodeIndex.numel(), " entries)");
      TORCH_CHECK(std::isfinite(slot.inflation) && slot.inflation >= 0.0f, "LOD slot ", s, " has no valid inflation (", slot.inflation, ")");
      paramsHost.lod.handles[s] = slot.handle;
      paramsHost.lod.aabbs[s] = slot.aabb;
      paramsHost.lod.indexOffset[s] = (int)offset;
      paramsHost.lod.inflation[s] = slot.inflation;
    }
    paramsHost.lod.numCapVariants = numCapVariants;
    for (int m = 0; m < numCapVariants; m++)
    {
      const float cap = lod[baseNumel + 1 + 2 * m];
      const float band = lod[baseNumel + 2 + 2 * m];
      TORCH_CHECK(std::isfinite(cap) && cap > 0.0f, "LOD cap variant ", m, " cap must be finite and > 0, got ", cap);
      TORCH_CHECK(std::isfinite(band) && band >= 1.0f && band < (float)numLodSlots && band == std::floor(band), "LOD cap variant ", m, " band slot must be an integer in [1, S), got ", band);
      const int b = (int)band;
      const int p = MaxLodSlots + m;
      const LodSlot &slot = _lodSlots[p];
      TORCH_CHECK(slot.handle != 0, "LOD cap variant ", m, " is enabled but not built");
      TORCH_CHECK(slot.unitGas == _state->iasBuffer, "LOD cap variant ", m, " points to a unit-primitive GAS that has moved; rebuild it");
      TORCH_CHECK(slot.count == _lodSlots[b].count, "LOD cap variant ", m, " holds ", slot.count, " Gaussians but its band slot ", b, " holds ", _lodSlots[b].count);
      TORCH_CHECK(slot.inflation == cap, "LOD cap variant ", m, " was built with inflation ", slot.inflation, ", not its cap ", cap);
      paramsHost.lod.capValue[m] = cap;
      paramsHost.lod.capBand[m] = b;
      paramsHost.lod.handles[p] = slot.handle;
      paramsHost.lod.aabbs[p] = slot.aabb;
      paramsHost.lod.indexOffset[p] = paramsHost.lod.indexOffset[b];
      paramsHost.lod.inflation[p] = slot.inflation;
    }
  }
  CudaTexture2DFloat4Object cuEnvMap = CudaTexture2DFloat4Object();
  int envmapHeight = envmap.size(0);
  int envmapWidth = envmap.size(1);
  if (envmapHeight > 0 && envmapWidth > 0) {
    cuEnvMap.reset(envmap.data_ptr<float>(), envmapHeight, envmapWidth);
    paramsHost.envmap = cuEnvMap.tex();
    paramsHost.envmapOffset = make_float2(envmapOffset.data_ptr<float>()[0],
                                          envmapOffset.data_ptr<float>()[1]);
  }

  cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();

  syncMaterials(paramsHost, materials, cudaStream);
  unsigned int numMaterials = materials.size();
  CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&paramsHost.materials),
                             sizeof(PBRMaterial) * numMaterials, cudaStream));
  CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(paramsHost.materials),
                             _playgroundState->materials.data(),
                             sizeof(PBRMaterial) * numMaterials,
                             cudaMemcpyHostToDevice, cudaStream));
  paramsHost.numMaterials = numMaterials;

  reallocatePlaygroundParamsDevice(sizeof(paramsHost), cudaStream);
  CUDA_CHECK(cudaMemcpyAsync(
      reinterpret_cast<void *>(_playgroundState->paramsDevice), &paramsHost,
      sizeof(paramsHost), cudaMemcpyHostToDevice, cudaStream));

  OPTIX_CHECK(optixLaunch(_playgroundState->pipelineTriGSTracing, cudaStream,
                          _playgroundState->paramsDevice,
                          sizeof(PlaygroundPipelineParameters),
                          &_playgroundState->sbtTriGSTracing, rayRad.size(2),
                          rayRad.size(1), rayRad.size(0)));

  CUDA_CHECK_LAST();

  return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor>(
                        rayRad, rayDns, rayHit, rayNrm, rayHitsCount, aovFirstHitT,
                        aovFirstNormal, aovFirstPrimType, aovFirstHitTransmittance, aovNumBounces,
                        aovGaussDepth);
}

torch::Tensor HybridOptixTracer::denoise(torch::Tensor rayRad) {

  cudaStream_t cudaStream = at::cuda::getCurrentCUDAStream();
  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto &denoiser = _playgroundState->denoiser;
  denoiser.setup(rayRad.size(2), rayRad.size(1), _playgroundState->context,
                 cudaStream);

  // Output buffer
  torch::Tensor rayRadDenoised =
      torch::empty({rayRad.size(0), rayRad.size(1), rayRad.size(2), 3}, opts);

  OptixDenoiserParams denoiserParams = {};
#if OPTIX_VERSION > 70700
  denoiserParams.temporalModeUsePreviousLayers = 0;
#elif OPTIX_VERSION > 70500
  denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
#endif
  denoiserParams.hdrIntensity = (CUdeviceptr)0;
  denoiserParams.blendFactor = 0.0f;

  // -------------------------------------------------------
  OptixImage2D inputLayer = {};
  inputLayer.data = reinterpret_cast<CUdeviceptr>(getPtr<float3>(rayRad));
  /// Width of the image (in pixels)
  inputLayer.width = rayRad.size(2);
  /// Height of the image (in pixels)
  inputLayer.height = rayRad.size(1);
  /// Stride between subsequent rows of the image (in bytes).
  inputLayer.rowStrideInBytes = rayRad.size(2) * sizeof(float3);
  /// Stride between subsequent pixels of the image (in bytes).
  /// For now, only 0 or the value that corresponds to a dense packing of pixels
  /// (no gaps) is supported.
  inputLayer.pixelStrideInBytes = sizeof(float3);
  /// Pixel format.
  inputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

  // -------------------------------------------------------
  OptixImage2D outputLayer;
  outputLayer.data =
      reinterpret_cast<CUdeviceptr>(getPtr<float3>(rayRadDenoised));
  /// Width of the image (in pixels)
  outputLayer.width = rayRad.size(2);
  /// Height of the image (in pixels)
  outputLayer.height = rayRad.size(1);
  /// Stride between subsequent rows of the image (in bytes).
  outputLayer.rowStrideInBytes = rayRad.size(2) * sizeof(float3);
  /// Stride between subsequent pixels of the image (in bytes).
  /// For now, only 0 or the value that corresponds to a dense packing of pixels
  /// (no gaps) is supported.
  outputLayer.pixelStrideInBytes = sizeof(float3);
  /// Pixel format.
  outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT3;

  // -------------------------------------------------------
  OptixDenoiserGuideLayer denoiserGuideLayer = {};

  OptixDenoiserLayer denoiserLayer = {};
  denoiserLayer.input = inputLayer;
  denoiserLayer.output = outputLayer;

  OPTIX_CHECK(
      optixDenoiserInvoke(denoiser._denoiser, cudaStream, &denoiserParams,
                          denoiser._denoiserStatePtr, denoiser._denoiserStateSz,
                          &denoiserGuideLayer, &denoiserLayer, 1,
                          /*inputOffsetX*/ 0,
                          /*inputOffsetY*/ 0, denoiser._denoiserScratchPtr,
                          denoiser._denoiserScratchSz));

  return rayRadDenoised;
}

// Denoiser

void OptixDenoiserWrapper::setup(unsigned int width, unsigned int height,
                                 OptixDeviceContext optixContext,
                                 cudaStream_t stream) {
  if ((width == _width) && (height == _height)) {
    return;
  }
  release(stream);
  OptixDenoiserOptions denoiserOptions = {};
  OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_LDR,
                                  &denoiserOptions, &_denoiser));
  OptixDenoiserSizes denoiserReturnSizes;
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(_denoiser, width, height,
                                                  &denoiserReturnSizes));
  _denoiserScratchSz =
      std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes,
               denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
  CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&_denoiserScratchPtr),
                             _denoiserScratchSz, stream));
  _denoiserStateSz = denoiserReturnSizes.stateSizeInBytes;
  CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&_denoiserStatePtr),
                             _denoiserStateSz, stream));
  OPTIX_CHECK(optixDenoiserSetup(_denoiser, 0, width, height, _denoiserStatePtr,
                                 _denoiserStateSz, _denoiserScratchPtr,
                                 _denoiserScratchSz));
  _width = width;
  _height = height;
}

void OptixDenoiserWrapper::release(cudaStream_t stream) {
  if (_denoiser) {
    OPTIX_CHECK_NOTHROW(optixDenoiserDestroy(_denoiser));
    CUDA_CHECK_NOTHROW(
        cudaFreeAsync(reinterpret_cast<void *>(_denoiserScratchPtr), stream));
    CUDA_CHECK_NOTHROW(
        cudaFreeAsync(reinterpret_cast<void *>(_denoiserStatePtr), stream));
  }
  _denoiser = nullptr;
  _width = _height = _denoiserScratchSz = _denoiserStateSz =
      _denoiserScratchPtr = _denoiserStatePtr = 0;
}