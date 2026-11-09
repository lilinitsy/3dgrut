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

#include <optix.h>

#include <3dgrt/pipelineParameters.h>
#include <3dgrt/tensorAccessor.h>
#include <playground/cutexture.h>
#include <playground/pipelineDefinitions.h>

struct alignas(16) PBRMaterial {
  cudaTextureObject_t diffuseTexture;           // 8 bytes -> offset 8
  cudaTextureObject_t emissiveTexture;          // 8 bytes -> offset 16
  cudaTextureObject_t metallicRoughnessTexture; // 8 bytes -> offset 24
  cudaTextureObject_t normalTexture;            // 8 bytes -> offset 32

  float4 diffuseFactor;     // 16 bytes -> offset 48
  float3 emissiveFactor;    // 12 bytes -> offset 60
  float metallicFactor;     // 4 bytes  -> offset 64
  float roughnessFactor;    // 4 bytes  -> offset 68
  float transmissionFactor; // 4 bytes  -> offset 72
  float ior;                // 4 bytes  -> offset 76
  float alphaCutoff;        // 4 bytes  -> offset 80
  unsigned int alphaMode; // see GltfAlphaMode         // 4 bytes  -> offset 84

  bool useDiffuseTexture;           // 1 byte -> offset 85
  bool useEmissiveTexture;          // 1 byte -> offset 86
  bool useMetallicRoughnessTexture; // 1 byte -> offset 87
  bool useNormalTexture;            // 1 byte -> offset 88

  float2 pad0; // 8 bytes -> offset 96
               // --> mem 16 byte aligned
};

// Cone tracing parameters
struct ConeParams
{
  float pixelSpread;    // camera ray spread (rad, as a Gaussian std)
  float spreadPerAlpha; // spread added per unit GGX alpha
  float maxSigma;       // max cone std (world)
  float maxRoughness;   // roughness above which the cone is not used
  int splitColumn;      // cones are used for pixels with launch x >= splitColumn
};

// Gaussian LOD cone tracing: one instance AS per width band ("slot"), all indexing one particle buffer
constexpr int MaxLodSlots = 10;
// Cap variants: for a per-pixel mode with several cone caps, the slot whose band holds cap c is also built inflated
// by exactly c, so a pixel capped at c traces what a frame capped at c would (its top slot is not inflated to the
// band edge or to the frame's largest cap). They live in physical slots MaxLodSlots .. MaxLodSlots + 7.
constexpr int MaxLodCapVariants = 8;
constexpr int MaxLodPhysicalSlots = MaxLodSlots + MaxLodCapVariants;

struct LodParams
{
  OptixTraversableHandle handles[MaxLodPhysicalSlots]; // slot 0 unused: it traces params.handle
  OptixAabb aabbs[MaxLodPhysicalSlots];                // scene bounds of each slot's AS, slot 0 unused (params.aabb)
  const int32_t *nodeIndex;                            // concatenated slot index tables: AS-local hit id -> particle
  float bandLower[MaxLodSlots];                        // lower edge of each slot's width band, slot 0 = 0
  float inflation[MaxLodPhysicalSlots]; // std each slot's proxies were inflated by: its widening bound, slot 0 unused
  int indexOffset[MaxLodPhysicalSlots]; // start of each slot's table in nodeIndex, slot 0 unused
  int numSlots;
  int enabled;
  // Cap variant m (physical slot MaxLodSlots + m) serves pixels capped at exactly capValue[m], in place of slot capBand[m]
  float capValue[MaxLodCapVariants];
  int capBand[MaxLodCapVariants];
  int numCapVariants;
};

struct PlaygroundPipelineParameters : PipelineParameters {
  PackedTensorAccessor32<float, 3> rayMaxT; ///< ray max t for termination
  cudaTextureObject_t envmap; // for envmaps and solid background color
  float2 envmapOffset;        // rotates env map along (theta, phi) axis

  // -- Playground specific launch params --
  OptixTraversableHandle
      triHandle; // Handle to BVH of mesh primitives: mirrors, glasses, pbr..

  unsigned int playgroundOpts; // see PlaygroundRenderOptions
  unsigned int maxPBRBounces;  // Maximum PBR ray iterations (reflections,
                               // transmissions & refractions)
  PackedTensorAccessor32<int32_t, 4>
      trace_state; // Scratch buffer, stores current render pass per ray
  PackedTensorAccessor32<int32_t, 2>
      triangles; // Primitive index -> vertex indices

  // Per vertex attributes
  PackedTensorAccessor32<float, 2> vNormals; // vertex normals
  PackedTensorAccessor32<bool, 2>
      vHasTangents; // has precomputed vertex tangents;
  PackedTensorAccessor32<float, 2> vTangents; // vertex tangents

  // Materials
  PackedTensorAccessor32<float, 3>
      matUV; // [F,3,2]: triangular face X vertex X 2d uv
  PackedTensorAccessor32<int32_t, 2> matID; // id of material to use, per vertex
  PBRMaterial *materials;                   // dynamic array of materials
  unsigned int numMaterials;

  // Per triangle attributes
  PackedTensorAccessor32<int32_t, 2> primType; // see PlaygroundPrimitiveTypes
  PackedTensorAccessor32<float, 2>
      refractiveIndex; // glass refraction, higher -> thicker glass

  // [B,H,W] per-ray RNG ids; valid only if useRayIds
  PackedTensorAccessor32<int32_t, 3> rayPixelId;
  PackedTensorAccessor32<int32_t, 3> raySampleIndex;
  unsigned int useRayIds;

  // [B,H,W,C] per-ray AOV outputs, written once at the end of __raygen__rg
  PackedTensorAccessor32<float, 4> aovFirstHitT;   // C=1, -1 if no hit
  PackedTensorAccessor32<float, 4> aovFirstNormal; // C=3, zeros if no hit
  PackedTensorAccessor32<float, 4> aovFirstHitTransmittance; // C=1
  PackedTensorAccessor32<float, 4> aovGaussDepth;            // C=1
  PackedTensorAccessor32<int32_t, 4> aovFirstPrimType;       // C=1, 0 if no hit
  PackedTensorAccessor32<int32_t, 4> aovNumBounces;          // C=1

  // PGRNDRenderConeTracing
  ConeParams cone;

  // Gaussian LOD for cones, used only when cone tracing is active and lod.enabled
  LodParams lod;

  // Per-pixel render mode, [B,H,W] float32, or nullptr (then splitColumn decides as before). A value n > 0 renders n
  // Monte Carlo samples looped in raygen and averaged (sample i uses the random numbers of frame frameNumber + i, or
  // of sample index + i with ray ids); a value -c < 0 renders one cone with its std capped at min(c, cone.maxSigma);
  // 0 renders nothing (black, alpha 0). Cone values need PGRNDRenderConeTracing.
  const float *pixelMode;
};
