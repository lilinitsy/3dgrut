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

  // Sample index within this pixel's in-kernel sample loop (params.pixelMode); added to the frame number in RNG keys
  unsigned int sampleIndex;
};

// This launch index's entry of params.pixelMode (see PlaygroundPipelineParameters); call only when it is set
static __device__ __forceinline__ float pixelModeValue()
{
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  return params.pixelMode[(idx.z * dim.y + idx.y) * dim.x + idx.x];
}

// Cones are used when the option is set and, with a per-pixel mode, this pixel's mode is a cone; without one, when this
// pixel is at or right of the split column
static __device__ __forceinline__ bool coneActive()
{
  if (!(params.playgroundOpts & PGRNDRenderConeTracing)) return false;
  if (params.pixelMode) return pixelModeValue() < 0.0f;
  return (int)optixGetLaunchIndex().x >= params.cone.splitColumn;
}

// Largest cone std for this pixel: params.cone.maxSigma, or the per-pixel cap when params.pixelMode is set
static __device__ __forceinline__ float coneCap()
{
  return params.pixelMode ? fminf(-pixelModeValue(), params.cone.maxSigma) : params.cone.maxSigma;
}

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

// same as traceVolumetricGS, but with every particle widened by the cone stddev at its distance, capped at params.cone.maxSigma.
static __device__ __inline__ void traceVolumetricGSCone(RayData &rayData, const float3 &rayOrigin, const float3 &rayDirection, const float tmin, const float tmax, const float coneSigma, const float coneSpread)
{
  const float cap = coneCap();
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
        widenParticle(particle, rayDirection, fminf(coneSigma + coneSpread * t, cap));
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

// Same as trace() of 3dgrtTracer.cuh, against the given acceleration structure instead of params.handle
static __device__ __inline__ void traceHandle(OptixTraversableHandle handle, RayPayload &rayPayload, const float3 &rayOri, const float3 &rayDir, const float tmin, const float tmax)
{
  uint32_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
  r0 = r2 = r4 = r6 = r8 = r10 = r12 = r14 = r16 = r18 = r20 = r22 = r24 = r26 = r28 = r30 = RayHit::InvalidParticleId;
  r1 = r3 = r5 = r7 = r9 = r11 = r13 = r15 = r17 = r19 = r21 = r23 = r25 = r27 = r29 = r31 = __float_as_int(RayHit::InfiniteDistance);
  optixTrace(handle, rayOri, rayDir, tmin, tmax, 0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | (PipelineParameters::SurfelPrimitive ? OPTIX_RAY_FLAG_NONE : OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES), 0, 1, 0, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31);
  rayPayload[0] = {r0, __uint_as_float(r1)};
  rayPayload[1] = {r2, __uint_as_float(r3)};
  rayPayload[2] = {r4, __uint_as_float(r5)};
  rayPayload[3] = {r6, __uint_as_float(r7)};
  rayPayload[4] = {r8, __uint_as_float(r9)};
  rayPayload[5] = {r10, __uint_as_float(r11)};
  rayPayload[6] = {r12, __uint_as_float(r13)};
  rayPayload[7] = {r14, __uint_as_float(r15)};
  rayPayload[8] = {r16, __uint_as_float(r17)};
  rayPayload[9] = {r18, __uint_as_float(r19)};
  rayPayload[10] = {r20, __uint_as_float(r21)};
  rayPayload[11] = {r22, __uint_as_float(r23)};
  rayPayload[12] = {r24, __uint_as_float(r25)};
  rayPayload[13] = {r26, __uint_as_float(r27)};
  rayPayload[14] = {r28, __uint_as_float(r29)};
  rayPayload[15] = {r30, __uint_as_float(r31)};
}

// Ray distance where the cone std coneSigma + coneSpread * t reaches sigma: -inf if already there, +inf if never
static __device__ __inline__ float lodSigmaCrossing(const float sigma, const float coneSigma, const float coneSpread)
{
  if (sigma <= coneSigma) return -RayHit::InfiniteDistance;
  return coneSpread > 0.0f ? (sigma - coneSigma) / coneSpread : RayHit::InfiniteDistance;
}

// Cone trace over the Gaussian LOD. [tmin, tmax] is split along t where the cone std crosses a band edge, and each piece
// is traced against its slot's AS, front to back, with one transmittance. Slot 0 is the main BVH and integrates as
// traceVolumetricGS (no widening). Other slots map AS-local hit ids through params.lod.nodeIndex and widen as
// traceVolumetricGSCone. A hit belongs to a piece when its proxy's closest approach lies in it (the intersection
// program rejects the rest), so a Gaussian in two slots can be counted twice or not at all at a band edge. The width
// is taken at the Gaussian mean's projection, which can lie beyond the piece, so it is also clamped to the slot's
// inflation: the proxy then always bounds the widened Gaussian.
static __device__ __noinline__ void traceVolumetricGSConeLod(RayData &rayData, const float3 &rayOrigin, const float3 &rayDirection, const float tmin, const float tmax, const float coneSigma, const float coneSpread)
{
  float rayTransmittance = 1.0f - rayData.density;
  float rayLastHitDistance = 0.0f;
  float3 canonicalIntersection = make_float3(0.f);
  float3 unusedNormal = make_float3(0.f);
  RayPayload rayPayload;
  // A per-pixel cap below the frame's cap ends the slot list at the slot whose band holds it: the slots are built for the
  // frame's largest cap, and their inflation (>= the cap of every band they serve) still bounds the widened Gaussians
  const float cap = coneCap();
  int numSlots = params.lod.numSlots;
  while ((numSlots > 1) && (params.lod.bandLower[numSlots - 1] >= cap)) numSlots--;
  // A cap variant built for exactly this cap replaces the top slot, so this pixel traces what a frame capped here would
  int topPhysical = numSlots - 1;
  for (int m = 0; m < params.lod.numCapVariants; m++)
  {
    if ((params.lod.capValue[m] == cap) && (params.lod.capBand[m] == numSlots - 1)) topPhysical = MaxLodSlots + m;
  }
  float bandStart = tmin;
  for (int s = 0; (s < numSlots) && (rayTransmittance > params.minTransmittance); s++)
  {
    // The piece ends where the next slot's band starts. Slots whose band ends before the piece start are skipped,
    // so a slot's band effectively starts at the largest lower edge so far.
    const float bandEnd = (s + 1 < numSlots) ? fminf(tmax, lodSigmaCrossing(params.lod.bandLower[s + 1], coneSigma, coneSpread)) : tmax;
    if ((s > 0) && (bandEnd <= bandStart)) continue;
    const int p = (s == numSlots - 1) ? topPhysical : s; // physical slot: s, or the cap variant serving it
    const OptixTraversableHandle handle = s == 0 ? params.handle : params.lod.handles[p];
    float2 minMaxT = intersectAABB(s == 0 ? params.aabb : params.lod.aabbs[p], rayOrigin, rayDirection);
    minMaxT.x = fmaxf(minMaxT.x, bandStart);
    minMaxT.y = fminf(minMaxT.y, bandEnd);
    bandStart = fmaxf(bandStart, bandEnd);
    rayLastHitDistance = fmaxf(0.0f, minMaxT.x - epsT);
    while ((rayLastHitDistance <= minMaxT.y) && (rayTransmittance > params.minTransmittance))
    {
      traceHandle(handle, rayPayload, rayOrigin, rayDirection, rayLastHitDistance + epsT, minMaxT.y + epsT);
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
          if (s == 0)
          {
#ifdef ENABLE_NORMALS
            const float hitWeight = particleDensityProcessHitFwdFromBuffer(rayOrigin, rayDirection, rayHit.particleId, {{(gaussianParticle_RawParameters_0 *)params.particleDensity, nullptr, true}}, &rayTransmittance, &rayData.hitDistance, &canonicalIntersection, true, &rayData.normal);
#else
            const float hitWeight = particleDensityProcessHitFwdFromBuffer(rayOrigin, rayDirection, rayHit.particleId, {{(gaussianParticle_RawParameters_0 *)params.particleDensity, nullptr, true}}, &rayTransmittance, &rayData.hitDistance, &canonicalIntersection, false, nullptr);
#endif
            particleFeaturesIntegrateFwdFromBuffer(rayDirection, canonicalIntersection, hitWeight, rayHit.particleId, const_cast<TParticleFeatureElem *>(params.particleFeatures), params.sphDegree, &rayData.features);
#ifdef ENABLE_HIT_COUNTS
            rayData.hitCount += hitWeight > 0.f ? 1.0f : 0.f;
#endif
          }
          else
          {
            const unsigned int particleId = params.lod.nodeIndex[params.lod.indexOffset[p] + rayHit.particleId];
            gaussianParticle_Parameters_0 particle = particleDensityParameters(particleId, {{(gaussianParticle_RawParameters_0 *)params.particleDensity, nullptr, true}});
            const float t = fmaxf(dot(particle.position_0 - rayOrigin, rayDirection), 0.0f);
            widenParticle(particle, rayDirection, fminf(fminf(coneSigma + coneSpread * t, params.lod.inflation[p]), cap));
            float alpha, depth;
            if (particleDensityHit(rayOrigin, rayDirection, particle, &alpha, &depth, &canonicalIntersection, false, &unusedNormal))
            {
              const float hitWeight = particleDensityIntegrateHit(alpha, &rayTransmittance, depth, &rayData.hitDistance, false, unusedNormal, &unusedNormal);
              particleFeaturesIntegrateFwdFromBuffer(rayDirection, canonicalIntersection, hitWeight, particleId, const_cast<TParticleFeatureElem *>(params.particleFeatures), params.sphDegree, &rayData.features);
            }
          }
          rayLastHitDistance = fmaxf(rayLastHitDistance, rayHit.distance);
        }
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
  if (coneActive() && params.lod.enabled)
  {
    traceVolumetricGSConeLod(rayData, rayOrigin, rayDirection, tmin, tmax, payload->coneSigma, payload->coneSpread);
  }
  else if (coneActive())
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