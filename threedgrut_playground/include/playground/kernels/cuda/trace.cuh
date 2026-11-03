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
#ifdef __PLAYGROUND__MODE__

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f 1.57079632679489661923f
#endif
#ifndef M_1_PIf
#define M_1_PIf 0.318309886183790671538f
#endif

#include <3dgrt/kernels/cuda/3dgrtTracer.cuh>
#include <optix.h>
#include <playground/kernels/cuda/mathUtils.cuh>
#include <playground/pipelineDefinitions.h>
#include <playground/pipelineParameters.h>

extern "C" {
#ifndef __PLAYGROUND__PARAMS__
__constant__ PlaygroundPipelineParameters params;
#define __PLAYGROUND__PARAMS__ 1
#endif
}

struct HybridRayPayload {
  // Path tracer control
  float3 rayOri;           // next ray origin to use
  unsigned int numBounces; // current number of reflectance bounces
  float3 rayDir; // next ray dir to use, if ray was reflected, refracted, etc
  float t_hit;   // ray t of latest intersection
  unsigned int rndSeed; // random seed for current ray

  // Volumetric radiance integration params
  RayData *rayData;

  // PBR  params
  float3 accumulatedColor; // Amount of RGB color accumulated so far by ray
  float accumulatedAlpha;  // Amount of density accumulated by the ray so far.
                           // Solid mesh faces count as opaque.
  float3
      directLight; // Total light to be reflected off the surfaces seen so far
  unsigned int pbrNumBounces; // Current number of PBR ray iterations, to limit
                              // reflections, refractions, etc
  float3 pathThroughput; // How much of the original light energy remains along
                         // current path
  float3 bsdfValue;      // How much light is scattered in the sampled direction
                    // (reflected or transmitted), recomputed each iteration
  float3 nextEmissive; // Light emitted due to last PBR surface hit
                       // recomputed each iteration
  bool rayMissed;      // True if ray missed the mesh primitives

  // Per-ray AOVs, captured in __closesthit__ch
  bool aovHasFirstHit;   // True once the first closest hit was recorded
  float aovFirstHitT;    // t of the first hit (before any eps shift), or -1
  float3 aovFirstNormal; // normal at the first hit, or zeros
  int aovFirstPrimType;  // primitive type of the first hit, or 0 (None)
  unsigned int aovNumRedirections; // Mirror / Glass hits + PBR BSDF samples

  // Ray cone (PGRNDRenderConeTracing): Gaussian std at the segment origin, its growth per unit t, and the growth
  // the last PBR hit adds once the cone reaches it
  float coneSigma;
  float coneSpread;
  float coneSpreadAdd;
};

constexpr float epsT = 1e-9; // Minimal offset to ray t to avoid zero t
constexpr float TRACE_MESH_TMIN = 1e-5;
constexpr float TRACE_MESH_TMAX = 1e5;

static __device__ __forceinline__ void *unpackPointer(unsigned int i0,
                                                      unsigned int i1) {
  const unsigned long long uptr =
      static_cast<unsigned long long>(i0) << 32 | i1;
  void *ptr = reinterpret_cast<void *>(uptr);
  return ptr;
}

static __device__ __forceinline__ void packPointer(void *ptr, unsigned int &i0,
                                                   unsigned int &i1) {
  const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

static __device__ __forceinline__ unsigned int getNextTraceState() {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels

  return params.trace_state[idx.z][ry][rx][0];
}

static __device__ __forceinline__ HybridRayPayload *getRayPayload() {
  const unsigned int u0 = optixGetPayload_0();
  const unsigned int u1 = optixGetPayload_1();
  return reinterpret_cast<HybridRayPayload *>(unpackPointer(u0, u1));
}

static __device__ __forceinline__ void
setNextTraceState(unsigned int traceState) {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels
  params.trace_state[idx.z][ry][rx][0] = traceState;
}

// Per-ray RNG ids (pixelId, sampleIndex), bit-cast to unsigned.
// Valid only if params.useRayIds.
static __device__ __forceinline__ uint2 getRayIds() {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels
  return make_uint2(
      static_cast<unsigned int>(params.rayPixelId[idx.z][ry][rx]),
      static_cast<unsigned int>(params.raySampleIndex[idx.z][ry][rx]));
}

static __device__ __forceinline__ void clearOutputBuffers() {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels

  params.rayFeatures[idx.z][ry][rx][0] = 0.0f;
  params.rayFeatures[idx.z][ry][rx][1] = 0.0f;
  params.rayFeatures[idx.z][ry][rx][2] = 0.0f;
  params.rayDensity[idx.z][ry][rx][0] = 0.0f;
}

static __device__ __forceinline__ void
writeRadianceDensityToOutputBuffer(float4 radiance) {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels

  params.rayFeatures[idx.z][ry][rx][0] = radiance.x;
  params.rayFeatures[idx.z][ry][rx][1] = radiance.y;
  params.rayFeatures[idx.z][ry][rx][2] = radiance.z;
  params.rayDensity[idx.z][ry][rx][0] = radiance.w;
}

static __device__ __forceinline__ void
accumulateRadianceToOutputBuffer(float3 radiance) {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels

  params.rayFeatures[idx.z][ry][rx][0] += radiance.x;
  params.rayFeatures[idx.z][ry][rx][1] += radiance.y;
  params.rayFeatures[idx.z][ry][rx][2] += radiance.z;
}

static __device__ __forceinline__ void
accumulateRadianceDensityToOutputBuffer(float4 radiance) {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels

  params.rayFeatures[idx.z][ry][rx][0] += radiance.x;
  params.rayFeatures[idx.z][ry][rx][1] += radiance.y;
  params.rayFeatures[idx.z][ry][rx][2] += radiance.z;
  params.rayDensity[idx.z][ry][rx][0] += radiance.w;
}

static __device__ __forceinline__ void
writeUpdatedRaysToBuffer(const float3 rayOri, const float3 rayDir) {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels
  params.rayOrigin[idx.z][ry][rx][0] = rayOri.x;
  params.rayOrigin[idx.z][ry][rx][1] = rayOri.y;
  params.rayOrigin[idx.z][ry][rx][2] = rayOri.z;
  params.rayDirection[idx.z][ry][rx][0] = rayDir.x;
  params.rayDirection[idx.z][ry][rx][1] = rayDir.y;
  params.rayDirection[idx.z][ry][rx][2] = rayDir.z;
}

static __device__ __forceinline__ void
writeAovsToBuffer(const HybridRayPayload &payload, float gaussDepth,
                  float firstHitTransmittance) {
  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels
  params.aovFirstHitT[idx.z][ry][rx][0] = payload.aovFirstHitT;
  params.aovFirstNormal[idx.z][ry][rx][0] = payload.aovFirstNormal.x;
  params.aovFirstNormal[idx.z][ry][rx][1] = payload.aovFirstNormal.y;
  params.aovFirstNormal[idx.z][ry][rx][2] = payload.aovFirstNormal.z;
  params.aovFirstPrimType[idx.z][ry][rx][0] = payload.aovFirstPrimType;
  params.aovFirstHitTransmittance[idx.z][ry][rx][0] = firstHitTransmittance;
  params.aovNumBounces[idx.z][ry][rx][0] =
      static_cast<int32_t>(payload.aovNumRedirections);
  params.aovGaussDepth[idx.z][ry][rx][0] = gaussDepth;
}

static __device__ __forceinline__ void
traceMesh(const float3 rayOri, const float3 rayDir, HybridRayPayload *payload) {
  setNextTraceState(PGRNDTracePrimitivesPass);

  unsigned int p0, p1;
  packPointer(payload, p0, p1);
  optixTrace(
      params.triHandle, rayOri, rayDir,
      TRACE_MESH_TMIN,               // Min intersection distance
      TRACE_MESH_TMAX,               // Max intersection distance
      0.0f,                          // rayTime -- used for motion blur
      OptixVisibilityMask(255),      // Specify always visible
      OPTIX_RAY_FLAG_DISABLE_ANYHIT, // |
                                     // OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
      0, // SBT offset   -- See SBT discussion
      1, // SBT stride   -- See SBT discussion
      0, // missSBTIndex -- See SBT discussion
      p0, p1);
}

// Convolve a gaussian with an isotropic Gaussian. The density preserves the integral of the ray.
static __device__ __inline__ void widenParticle(gaussianParticle_Parameters_0 &particle, const float3 &rayDirection, const float sigma)
{
  const float3 d = make_float3(dot(particle.rotationT_0.rows[0], rayDirection), dot(particle.rotationT_0.rows[1], rayDirection), dot(particle.rotationT_0.rows[2], rayDirection));
  const float3 s2 = particle.scale_0 * particle.scale_0;
  const float3 w2 = s2 + make_float3(sigma * sigma);
  const float q = d.x * d.x / s2.x + d.y * d.y / s2.y + d.z * d.z / s2.z;
  const float qWide = d.x * d.x / w2.x + d.y * d.y / w2.y + d.z * d.z / w2.z;
  particle.density_0 *= sqrtf((s2.x / w2.x) * (s2.y / w2.y) * (s2.z / w2.z) * q / qWide);
  particle.scale_0 = make_float3(sqrtf(w2.x), sqrtf(w2.y), sqrtf(w2.z));
}

// same as traceVolumetricGS, but with every particle widened by the cone stddev at its distance, capped at some coneParams.z value.
static __device__ __inline__ void traceVolumetricGSCone(RayData &rayData, const float3 &rayOrigin, const float3 &rayDirection, const float tmin, const float tmax, const float coneSigma, const float coneSpread)
{
  float rayTransmittance = 1.0f - rayData.density;
  float2 minMaxT = intersectAABB(params.aabb, rayOrigin, rayDirection);
  minMaxT.x = fmaxf(minMaxT.x, tmin);
  minMaxT.y = fminf(minMaxT.y, tmax);
  float rayLastHitDistance = fmaxf(0.0f, minMaxT.x - epsT);
  float3 unusedNormal = make_float3(0.f);
  RayPayload rayPayload;
  while ((rayLastHitDistance <= minMaxT.y) && (rayTransmittance > params.minTransmittance))
  {
    trace(rayPayload, rayOrigin, rayDirection, rayLastHitDistance + epsT, minMaxT.y + epsT);
    if (rayPayload[0].particleId == RayHit::InvalidParticleId)
    {
      break;
    }
#pragma unroll
    for (int i = 0; i < PipelineParameters::MaxNumHitPerTrace; i++)
    {
      const RayHit rayHit = rayPayload[i];
      if ((rayHit.particleId != RayHit::InvalidParticleId) && (rayTransmittance > params.minTransmittance))
      {
        gaussianParticle_Parameters_0 particle = particleDensityParameters(rayHit.particleId, {{(gaussianParticle_RawParameters_0 *)params.particleDensity, nullptr, true}});
        const float t = fmaxf(dot(particle.position_0 - rayOrigin, rayDirection), 0.0f);
        widenParticle(particle, rayDirection, fminf(coneSigma + coneSpread * t, params.coneParams.z));
        float alpha, depth;
        float3 canonicalIntersection;
        if (particleDensityHit(rayOrigin, rayDirection, particle, &alpha, &depth, &canonicalIntersection, false, &unusedNormal))
        {
          const float hitWeight = particleDensityIntegrateHit(alpha, &rayTransmittance, depth, &rayData.hitDistance, false, unusedNormal, &unusedNormal);
          particleFeaturesIntegrateFwdFromBuffer(rayDirection, canonicalIntersection, hitWeight, rayHit.particleId, const_cast<TParticleFeatureElem *>(params.particleFeatures), params.sphDegree, &rayData.features);
        }
        rayLastHitDistance = fmaxf(rayLastHitDistance, rayHit.distance);
      }
    }
  }
  rayData.density = 1 - rayTransmittance;
  rayData.rayLastHitDistance = rayLastHitDistance;
}

static __device__ __forceinline__ float4 traceGaussians(
    RayData &rayData, const float3 &rayOrigin, const float3 &rayDirection,
    const float tmin, const float tmax, HybridRayPayload *payload) {

  const uint3 idx = optixGetLaunchIndex();
  const int rx =
      fminf(idx.x, params.frameBounds.x); // Ray coordinates in pixels
  const int ry =
      fminf(idx.y, params.frameBounds.y); // Ray coordinates in pixels

  if (params.playgroundOpts & PGRNDRenderDisableGaussianTracing)
    return make_float4(0.0);

  // Copy RayData, to avoid writing the output buffer by this pass
  RayData prevRayData = rayData;
  setNextTraceState(PGRNDTraceRTGaussiansPass);
  if (params.playgroundOpts & PGRNDRenderConeTracing)
  {
    traceVolumetricGSCone(rayData, rayOrigin, rayDirection, tmin, tmax, payload->coneSigma, payload->coneSpread);
  }
  else
  {
    traceVolumetricGS(rayData, rayOrigin, rayDirection, tmin, tmax);
  }

  // The difference in the output buffer is the result of this trace path
  float4 accumulated_radiance =
      make_float4(rayData.features[0] - prevRayData.features[0],
                  rayData.features[1] - prevRayData.features[1],
                  rayData.features[2] - prevRayData.features[2],
                  rayData.density - prevRayData.density);

  return accumulated_radiance;
}

static __device__ __forceinline__ float3
getBackgroundColor(const float3 rayDir) {
  // Apply rotation to the ray direction based on offset: convert offset from
  // [0,1] range to rotation angles
  float rotY = params.envmapOffset.x * 2.0f * M_PIf; // Yaw   (around Y-axis)
  float rotX = 2.0 * params.envmapOffset.y * M_PIf;  // Pitch (around X-axis)

  // Apply yaw rotation
  float3 rotatedDir;
  rotatedDir.x = rayDir.x * cosf(rotY) - rayDir.z * sinf(rotY);
  rotatedDir.z = rayDir.x * sinf(rotY) + rayDir.z * cosf(rotY);
  rotatedDir.y = rayDir.y;

  // Apply pitch rotation
  float3 doubleRotatedDir;
  doubleRotatedDir.x = rotatedDir.x;
  doubleRotatedDir.y = rotatedDir.y * cosf(rotX) - rotatedDir.z * sinf(rotX);
  doubleRotatedDir.z = rotatedDir.y * sinf(rotX) + rotatedDir.z * cosf(rotX);

  // Compute texture coordinates from the rotated direction
  float theta = atan2(doubleRotatedDir.x, doubleRotatedDir.z);
  float phi = M_PIf * 0.5f - acosf(clamp(doubleRotatedDir.y, -1.0f, 1.0f));
  float u = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v = 0.5f * (1.0f + sin(phi));
  float4 env = tex2D<float4>(params.envmap, u, v);
  return make_float3(env.x, env.y, env.z);
}

#endif