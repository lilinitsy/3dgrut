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

  // PGRNDRenderConeTracing: x = camera ray spread (rad, as a Gaussian std), y = spread added per unit GGX alpha, z = max cone std (world)
  float3 coneParams;
};
