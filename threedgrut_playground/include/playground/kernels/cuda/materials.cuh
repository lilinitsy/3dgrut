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

#include <optix.h>
#include <playground/kernels/cuda/dfgLut.cuh>
#include <playground/kernels/cuda/mathUtils.cuh>
#include <playground/kernels/cuda/rng.cuh>
#include <playground/kernels/cuda/trace.cuh>
#include <playground/pipelineParameters.h>

extern "C" {
#ifndef __PLAYGROUND__PARAMS__
__constant__ PlaygroundPipelineParameters params;
#define __PLAYGROUND__PARAMS__ 1
#endif
}

constexpr float PBR_EPS = 1e-6; // Minimal eps to avoid zero division

static __device__ __inline__ const PBRMaterial &
get_material(const unsigned int matId) {
  return params.materials[matId];
}

static __device__ __inline__ float3 get_diffuse_color(const float3 ray_d,
                                                      float3 normal) {
  const unsigned int triId = optixGetPrimitiveIndex();
  const unsigned int materialId = params.matID[triId][0];
  const auto material = get_material(materialId);

  const float2 uv0 =
      make_float2(params.matUV[triId][0][0], params.matUV[triId][0][1]);
  const float2 uv1 =
      make_float2(params.matUV[triId][1][0], params.matUV[triId][1][1]);
  const float2 uv2 =
      make_float2(params.matUV[triId][2][0], params.matUV[triId][2][1]);
  const float2 barycentric = optixGetTriangleBarycentrics();
  float2 texCoords = (1 - barycentric.x - barycentric.y) * uv0 +
                     barycentric.x * uv1 + barycentric.y * uv2;

  float3 diffuse;
  bool disableTextures = params.playgroundOpts & PGRNDRenderDisablePBRTextures;

  float3 diffuseFactor =
      make_float3(material.diffuseFactor.x, material.diffuseFactor.y,
                  material.diffuseFactor.z);
  if (!material.useDiffuseTexture || disableTextures) {
    diffuse = diffuseFactor;
  } else {
    cudaTextureObject_t diffuseTex = material.diffuseTexture;
    float4 diffuse_fp4 = tex2D<float4>(diffuseTex, texCoords.x, texCoords.y);
    diffuse = make_float3(diffuse_fp4.x, diffuse_fp4.y, diffuse_fp4.z);
    diffuse *= diffuseFactor;
  }

  float shade = fabsf(dot(ray_d, normal));
  return diffuse * shade;
}

/* Dot product, limited to positive values, clamped otherwise */
static __device__ __inline__ float positive_dot(const float3 a,
                                                const float3 b) {
  return clamp(dot(a, b), 0.0f, 1.0f);
}

/* Step function*/
static __device__ __inline__ float heaviside(float x) {
  return x > 0 ? 1.0 : 0.0;
}

static __device__ inline float3 normalize(float3 v) {
  const float norm = dot(v, v);
  return (norm > PBR_EPS) ? v * rsqrtf(norm) : v;
}

static __device__ __inline__ bool has_precomputed_tangents() {
  const unsigned int triId = optixGetPrimitiveIndex();
  const unsigned int v0_idx = params.triangles[triId][0];
  const unsigned int v1_idx = params.triangles[triId][1];
  const unsigned int v2_idx = params.triangles[triId][2];
  return params.vHasTangents[v0_idx][0] && params.vHasTangents[v1_idx][0] &&
         params.vHasTangents[v2_idx][0];
}

static __device__ __inline__ float3 get_smooth_tangent() {
  // Uses interpolated vertex tangents to get a smooth varying interpolated
  // tangent
  const unsigned int triId = optixGetPrimitiveIndex();
  const unsigned int v0_idx = params.triangles[triId][0];
  const unsigned int v1_idx = params.triangles[triId][1];
  const unsigned int v2_idx = params.triangles[triId][2];
  const float3 t0 =
      make_float3(params.vTangents[v0_idx][0], params.vTangents[v0_idx][1],
                  params.vTangents[v0_idx][2]);
  const float3 t1 =
      make_float3(params.vTangents[v1_idx][0], params.vTangents[v1_idx][1],
                  params.vTangents[v1_idx][2]);
  const float3 t2 =
      make_float3(params.vTangents[v2_idx][0], params.vTangents[v2_idx][1],
                  params.vTangents[v2_idx][2]);
  const float2 barycentric = optixGetTriangleBarycentrics();
  float3 interpolated_tangent = (1 - barycentric.x - barycentric.y) * t0 +
                                barycentric.x * t1 + barycentric.y * t2;
  interpolated_tangent /= length(interpolated_tangent);

  return interpolated_tangent;
}

// Computes TBN matrix and multiplies dir
static __device__ __inline__ float3 compute_normal_space(const float3 normal,
                                                         const float3 dir) {
  float3 tangent;
  float3 bitangent;

  if (has_precomputed_tangents()) {
    tangent = get_smooth_tangent();
  } else {
    if (fabs(normal.x) > fabs(normal.z))
      tangent = make_float3(-normal.y, normal.x, 0.0f);
    else
      tangent = make_float3(0.0f, -normal.z, normal.y);
  }
  tangent = normalize(tangent);
  bitangent = normalize(cross(normal, tangent));

  // Create TBN matrix
  float3 worldNormal;
  worldNormal.x = tangent.x * dir.x + bitangent.x * dir.y + normal.x * dir.z;
  worldNormal.y = tangent.y * dir.x + bitangent.y * dir.y + normal.y * dir.z;
  worldNormal.z = tangent.z * dir.x + bitangent.z * dir.y + normal.z * dir.z;
  return normalize(worldNormal);
}

static __device__ __inline__ float3
importance_sample_diffuse_ggx(const float3 normal,
                              const float random_theta_seed,
                              const float random_phi_seed) {
  const float theta =
      asin(random_theta_seed); // changed from asin(sqrtf(random_theta_seed));
  const float phi = 2.0 * pi() * random_phi_seed;
  // Sampled indirect diffuse direction in normal space
  const float3 local_diffuse_dir =
      make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
  float3 L = compute_normal_space(normal, local_diffuse_dir);
  return L;
}

static __device__ __inline__ float3 importance_sample_specular_ggx(
    const float3 normal, const float random_theta_seed,
    const float random_phi_seed, const float roughness) {
  const float alpha = roughness * roughness;
  const float theta =
      acos(sqrtf((1.0 - random_theta_seed) /
                 (1.0 + (alpha * alpha - 1.0) * random_theta_seed)));
  const float phi = 2.0 * pi() * random_phi_seed;

  const float3 local_H =
      make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
  float3 H = compute_normal_space(normal, local_H);
  return H;
}

static __device__ __inline__ bool alpha_test(const float alpha,
                                             const unsigned int alphaMode,
                                             const float alphaCutoff,
                                             unsigned int &rndSeed) {
  // Run alpha test according to:
  // https://github.com/KhronosGroup/glTF-Sample-Models/blob/main/2.0/AlphaBlendModeTest/README.md
  // If returns true, alpha test succeeds, if false the current hit should be
  // ignored ("no material")
  if (alphaMode == GLTFBlend) {
    return alpha > rnd(rndSeed);
  } else if (alphaMode == GLTFMask) {
    return alpha > alphaCutoff;
  } else { // Opaque
    return true;
  }
}

/* D: Normal distribution function approximation */
static __device__ __inline__ float
trowbridge_reitz_ggx(const float3 h, // halfway vector
                     const float3 normal, const float roughness) {
  const float alpha = roughness * roughness;
  const float alpha_sqr = alpha * alpha;
  const float n_dot_h = positive_dot(normal, h);
  const float denom = n_dot_h * n_dot_h * (alpha_sqr - 1.0) + 1.0;
  return alpha_sqr / fmaxf((pi() * denom * denom), PBR_EPS);
}

/* G: Geometry function: microfacet occlusion approximation */
static __device__ __inline__ float geometry_schlick_ggx(const float n_dot_v,
                                                        const float roughness) {
  const float alpha = 0.5 * roughness * roughness;
  const float denom = n_dot_v * (1.0 - alpha) + alpha;
  return n_dot_v / fmaxf(denom, PBR_EPS);
}

/* G: Geometry function: microfacet occlusion approximation */
static __device__ __inline__ float geometry_smith(const float normal_dot_wo,
                                                  const float normal_dot_wi,
                                                  const float roughness) {
  return geometry_schlick_ggx(normal_dot_wo, roughness) *
         geometry_schlick_ggx(normal_dot_wi, roughness);
}

/* F: Reflectance function: Fresnel Schlick approximation */
static __device__ __inline__ float3 fresnel_schlick(const float cosine,
                                                    const float3 F0) {
  return F0 + (1.0 - F0) * powf(1.0 - cosine, 5.0);
}

static __device__ __inline__ float3 pbr_refract(float3 wi, float3 normal,
                                                float eta) {
  const float normal_dot_wi = dot(normal, wi);
  const float k = 1.0 - eta * eta * (1.0 - normal_dot_wi * normal_dot_wi);
  return (k < 0.0) ? make_float3(0.0)
                   : eta * wi - (eta * normal_dot_wi + sqrtf(k)) * normal;
}

static __device__ __inline__ unsigned int getFrameNumber() {
  const uint3 idx = optixGetLaunchIndex();
  return params.frameNumber + idx.z;
}

static __device__ __inline__ float3
sampled_microfacet_brdf(const float3 wo, const float3 normal,
                        const float3 base_color, const float metalness,
                        const float roughness, const float transmission,
                        const float ior, float3 &next_ray_dir,
                        unsigned int &rndSeed, HybridRayPayload *payload) {
  // Outputs
  float3 out_factor;
  float3 L;

  // Constant values
  const float fresnel_reflect = 0.5;

  // Stochastic variables
  const uint3 idx = optixGetLaunchIndex();
  uint3 rngKey;
  if (params.useRayIds) {
    const uint2 ids = getRayIds();
    rngKey = make_uint3(ids.x, ids.y + payload->sampleIndex, payload->pbrNumBounces);
  } else {
    rngKey =
        make_uint3(idx.x, idx.y, getFrameNumber() + payload->sampleIndex + payload->pbrNumBounces);
  }
  const float3 rand = rnd_pcg3d(rngKey);
  //     const float3 rand = rnd3(rndSeed);   // Enable for an alternative
  //     randomization procedure
  const float random_phi_seed = rand.x;
  const float random_theta_seed = rand.y;
  const float ray_prob = rand.z;

  // Lobe choice in proportion to each lobe's expected weight (the same DFG-based weights as the cone path), each
  // sample divided by its lobe's probability. Every lobe with a nonzero weight keeps a nonzero probability, so the
  // estimate stays unbiased; a metal never spends a sample on its zero-weight diffuse lobe.
  float pTrans, pDiff;
  {
    const float3 f0w = lerp(make_float3(0.16f * fresnel_reflect * fresnel_reflect), base_color, metalness);
    const float2 dfg = dfgLookup(fmaxf(fabsf(dot(normal, wo)), 1e-4f), roughness);
    const float3 specW = f0w * dfg.x + make_float3(dfg.y);
    const float3 notSpec = (make_float3(1.0f) - specW) * base_color;
    const float3 diffW = notSpec * ((1.0f - metalness) * (1.0f - transmission));
    const float3 transW = notSpec * transmission;
    const float3 lum = make_float3(0.2126f, 0.7152f, 0.0722f);
    const float specL = fmaxf(dot(specW, lum), 0.0f), diffL = fmaxf(dot(diffW, lum), 0.0f);
    const float transL = fmaxf(dot(transW, lum), 0.0f), sumL = specL + diffL + transL;
    pTrans = sumL > 0.0f ? transL / sumL : 0.0f;
    pDiff = sumL > 0.0f ? diffL / sumL : 0.0f;
  }

  if (ray_prob < pTrans) { // Transmissive
    float frontFacing = dot(wo, normal);
    float3 forwardNormal;
    float eta;
    if (frontFacing >= 0.0) {
      forwardNormal = normal;
      eta = 1.0 / ior;
    } else {
      forwardNormal = -normal;
      eta = ior;
    }

    // Importance sampling transmittance
    const float3 H = importance_sample_specular_ggx(
        forwardNormal, random_theta_seed, random_phi_seed, roughness);
    L = pbr_refract(-wo, H, eta);

    const float fnormal_dot_wo = positive_dot(forwardNormal, wo);
    const float fnormal_dot_L = positive_dot(-forwardNormal, L);
    const float fnormal_dot_H = positive_dot(forwardNormal, H);
    const float wo_dot_H = positive_dot(wo, H);

    // F0 for dielectics in range [0.0, 0.16]
    // default FO is (0.16 * 0.5^2) = 0.04
    float3 f0 = make_float3(0.16 * fresnel_reflect * fresnel_reflect);
    // in case of metals, baseColor contains F0
    f0 = lerp(f0, base_color, metalness);

    float3 F = fresnel_schlick(wo_dot_H, f0);
    float D = trowbridge_reitz_ggx(H, forwardNormal, roughness);
    float G = geometry_smith(fnormal_dot_wo, fnormal_dot_L, roughness);
    out_factor = base_color * (make_float3(1.0) - F) * G * wo_dot_H /
                 fmaxf((fnormal_dot_H * fnormal_dot_wo), 0.001);
    out_factor *= transmission / pTrans;
  } else if (ray_prob < pTrans + pDiff) { // Diffuse
    // Importance sampling diffuse
    L = importance_sample_diffuse_ggx(normal, random_theta_seed,
                                      random_phi_seed);

    // Half vector
    const float3 H = normalize(wo + L);

    const float wo_dot_H = positive_dot(wo, H);
    // F0 for dielectics in range [0.0, 0.16]
    // default FO is (0.16 * 0.5^2) = 0.04
    float3 f0 = make_float3(0.16 * (fresnel_reflect * fresnel_reflect));
    // in case of metals, baseColor contains F0
    f0 = lerp(f0, base_color, metalness);
    float3 F = fresnel_schlick(wo_dot_H, f0);

    // If not specular, use as diffuse
    float3 not_spec = make_float3(1.0) - F;
    // no diffuse for metals
    not_spec *= (1.0 - metalness);

    out_factor = not_spec * base_color * ((1.0f - transmission) / pDiff);
  } else {
    // Specular
    // important sample GGX
    const float3 H = importance_sample_specular_ggx(normal, random_theta_seed,
                                                    random_phi_seed, roughness);
    L = -wo - 2.0 * dot(H, -wo) * H;

    const float normal_dot_wo = positive_dot(normal, wo);
    const float normal_dot_H = positive_dot(normal, H);
    const float normal_dot_L = positive_dot(normal, L);
    const float wo_dot_H = positive_dot(wo, H);

    // F0 for dielectics in range [0.0, 0.16]
    // default FO is (0.16 * 0.5^2) = 0.04
    float3 f0 = make_float3(0.16 * fresnel_reflect * fresnel_reflect);
    f0 = lerp(f0, base_color, metalness);

    // specular microfacet (cook-torrance) BRDF
    float3 F = fresnel_schlick(wo_dot_H, f0);
    float D = trowbridge_reitz_ggx(H, normal, roughness);
    float G = geometry_smith(normal_dot_wo, normal_dot_L, roughness);
    out_factor =
        F * G * wo_dot_H / fmaxf((normal_dot_H * normal_dot_wo), 0.001);
    out_factor /= fmaxf(1.0f - pTrans - pDiff, 1e-6f);
  }

  next_ray_dir = L;
  payload->pbrNumBounces = payload->pbrNumBounces + 1;
  return make_float3(out_factor.x, out_factor.y, out_factor.z);
}

// Parts of implementation adopted from Thorsten Thormählen's Path Tracer at:
// https://www.gsn-lib.org/apps/raytracing/index.php?name=example_transmission
static __device__ __inline__ void sampled_cook_torrance_brdf(
    const float3 ray_o, const float3 ray_d, const float hit_t, float3 normal,
    float3 &new_ray_dir, unsigned int &rndSeed, HybridRayPayload *payload) {
  const float3 wo =
      normalize(-ray_d); // Out light reflects back at the viewer's direction
  const float3 hit_point =
      ray_o + hit_t * ray_d; // 3D Point where the ray hit the surface

  const unsigned int triId = optixGetPrimitiveIndex();
  const unsigned int materialId = params.matID[triId][0];
  const auto material = get_material(materialId);

  const float2 uv0 =
      make_float2(params.matUV[triId][0][0], params.matUV[triId][0][1]);
  const float2 uv1 =
      make_float2(params.matUV[triId][1][0], params.matUV[triId][1][1]);
  const float2 uv2 =
      make_float2(params.matUV[triId][2][0], params.matUV[triId][2][1]);
  const float2 barycentric = optixGetTriangleBarycentrics();
  float2 texCoords = (1 - barycentric.x - barycentric.y) * uv0 +
                     barycentric.x * uv1 + barycentric.y * uv2;

  float3 diffuse;
  float3 emissive;
  float metallic;
  float roughness;
  float transmission;
  float ior;
  bool disablePBRTextures =
      params.playgroundOpts & PGRNDRenderDisablePBRTextures;

  float3 diffuseFactor =
      make_float3(material.diffuseFactor.x, material.diffuseFactor.y,
                  material.diffuseFactor.z);
  float alpha = material.diffuseFactor.w;
  if (!material.useDiffuseTexture || disablePBRTextures) {
    diffuse = diffuseFactor;
  } else {
    cudaTextureObject_t diffuseTex = material.diffuseTexture;
    float4 diffuse_fp4 = tex2D<float4>(diffuseTex, texCoords.x, texCoords.y);
    diffuse = make_float3(diffuse_fp4.x, diffuse_fp4.y, diffuse_fp4.z);
    diffuse *= diffuseFactor;
    alpha *= diffuse_fp4.w;
  }

  if (!material.useEmissiveTexture || disablePBRTextures) {
    emissive = material.emissiveFactor;
  } else {
    cudaTextureObject_t emissiveTex = material.emissiveTexture;
    float4 emissive_fp4 = tex2D<float4>(emissiveTex, texCoords.x, texCoords.y);
    emissive = make_float3(emissive_fp4.x, emissive_fp4.y, emissive_fp4.z) *
               material.emissiveFactor;
  }

  if (!material.useMetallicRoughnessTexture || disablePBRTextures) {
    metallic = material.metallicFactor;
    roughness = material.roughnessFactor;
  } else {
    cudaTextureObject_t metallicRoughnessTex =
        material.metallicRoughnessTexture;
    float2 metallic_roughness_fp2 =
        tex2D<float2>(metallicRoughnessTex, texCoords.x, texCoords.y);
    metallic = metallic_roughness_fp2.x * material.metallicFactor;
    roughness = metallic_roughness_fp2.y * material.roughnessFactor;
  }

  if (material.useNormalTexture && !disablePBRTextures) {
    cudaTextureObject_t normalTex = material.normalTexture;
    const float4 normal_fp4 =
        tex2D<float4>(normalTex, texCoords.x, texCoords.y);
    float3 normal_tex = make_float3(normal_fp4.x, normal_fp4.y, normal_fp4.z);
    normal = compute_normal_space(normal, normal_tex);
  }

  transmission = material.transmissionFactor;
  ior = material.ior;

  // gltf 2.0 alpha test - for "no material"
  if (!alpha_test(alpha, material.alphaMode, material.alphaCutoff, rndSeed)) {
    new_ray_dir = ray_d;
    return;
  }

  const float3 nextScatter =
      sampled_microfacet_brdf(wo, normal, diffuse, metallic, roughness,
                              transmission, ior, new_ray_dir, rndSeed, payload);
  const float3 nextEmissive = emissive;
  payload->bsdfValue = maxf3(nextScatter, make_float3(0.0));
  payload->nextEmissive = nextEmissive;
}


struct PBRSurface
{
  float3 diffuse;
  float3 emissive;
  float metallic;
  float roughness;
  float transmission;
  float ior;
  float alpha;
  unsigned int alphaMode;
  float alphaCutoff;
  float3 normal;
};

// Same fetch as the first half of sampled_cook_torrance_brdf. Used only by the cone path.
static __device__ __inline__ PBRSurface fetch_pbr_surface(const float3 normal)
{
  const unsigned int triId = optixGetPrimitiveIndex();
  const unsigned int materialId = params.matID[triId][0];
  const auto material = get_material(materialId);

  const float2 uv0 = make_float2(params.matUV[triId][0][0], params.matUV[triId][0][1]);
  const float2 uv1 = make_float2(params.matUV[triId][1][0], params.matUV[triId][1][1]);
  const float2 uv2 = make_float2(params.matUV[triId][2][0], params.matUV[triId][2][1]);
  const float2 barycentric = optixGetTriangleBarycentrics();
  const float2 texCoords = (1 - barycentric.x - barycentric.y) * uv0 + barycentric.x * uv1 + barycentric.y * uv2;
  const bool disablePBRTextures = params.playgroundOpts & PGRNDRenderDisablePBRTextures;

  PBRSurface surface;
  surface.normal = normal;
  const float3 diffuseFactor = make_float3(material.diffuseFactor.x, material.diffuseFactor.y, material.diffuseFactor.z);
  surface.alpha = material.diffuseFactor.w;
  surface.alphaMode = material.alphaMode;
  surface.alphaCutoff = material.alphaCutoff;
  if (!material.useDiffuseTexture || disablePBRTextures)
  {
    surface.diffuse = diffuseFactor;
  }
  else
  {
    const float4 diffuse_fp4 = tex2D<float4>(material.diffuseTexture, texCoords.x, texCoords.y);
    surface.diffuse = make_float3(diffuse_fp4.x, diffuse_fp4.y, diffuse_fp4.z) * diffuseFactor;
    surface.alpha *= diffuse_fp4.w;
  }

  if (!material.useEmissiveTexture || disablePBRTextures)
  {
    surface.emissive = material.emissiveFactor;
  }
  else
  {
    const float4 emissive_fp4 = tex2D<float4>(material.emissiveTexture, texCoords.x, texCoords.y);
    surface.emissive = make_float3(emissive_fp4.x, emissive_fp4.y, emissive_fp4.z) * material.emissiveFactor;
  }

  if (!material.useMetallicRoughnessTexture || disablePBRTextures)
  {
    surface.metallic = material.metallicFactor;
    surface.roughness = material.roughnessFactor;
  }
  else
  {
    const float2 metallic_roughness_fp2 = tex2D<float2>(material.metallicRoughnessTexture, texCoords.x, texCoords.y);
    surface.metallic = metallic_roughness_fp2.x * material.metallicFactor;
    surface.roughness = metallic_roughness_fp2.y * material.roughnessFactor;
  }

  if (material.useNormalTexture && !disablePBRTextures)
  {
    const float4 normal_fp4 = tex2D<float4>(material.normalTexture, texCoords.x, texCoords.y);
    surface.normal = compute_normal_space(normal, make_float3(normal_fp4.x, normal_fp4.y, normal_fp4.z));
  }

  surface.transmission = material.transmissionFactor;
  surface.ior = material.ior;
  return surface;
}

static __device__ __inline__ float cone_luminance(const float3 c)
{
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// Hybrid lobes: the specular and transmission lobes are cones, the diffuse lobe is a sampled direction as in the MC path.
// One lobe is chosen per hit by its luminance weight. Rough surfaces fall back to the sampled BRDF.
static __device__ __noinline__ void cone_cook_torrance_brdf(const float3 ray_d, const float3 normal, float3 &new_ray_dir, unsigned int &rndSeed, HybridRayPayload *payload)
{
  const PBRSurface surface = fetch_pbr_surface(normal);
  if (!alpha_test(surface.alpha, surface.alphaMode, surface.alphaCutoff, rndSeed))
  {
    new_ray_dir = ray_d;
    return;
  }
  const float3 wo = normalize(-ray_d);
  if (surface.roughness > params.cone.maxRoughness)
  {
    const float3 nextScatter = sampled_microfacet_brdf(wo, surface.normal, surface.diffuse, surface.metallic, surface.roughness, surface.transmission, surface.ior, new_ray_dir, rndSeed, payload);
    payload->bsdfValue = maxf3(nextScatter, make_float3(0.0));
    payload->nextEmissive = surface.emissive;
    return;
  }
  const float roughness = surface.roughness;
  const float frontFacing = dot(wo, surface.normal);
  const float3 n = frontFacing < 0.0f ? -surface.normal : surface.normal;
  const float nDotV = fmaxf(dot(n, wo), 1e-4f);
  const float3 f0 = make_float3(0.04f) * (1.0f - surface.metallic) + surface.diffuse * surface.metallic;
  const float2 dfg = dfgLookup(nDotV, roughness);
  const float3 specW = f0 * dfg.x + make_float3(dfg.y);
  const float3 notSpec = (make_float3(1.0f) - specW) * surface.diffuse * (1.0f - surface.metallic);
  const float3 diffW = notSpec * (1.0f - surface.transmission);
  const float3 transW = notSpec * surface.transmission;
  const float specL = cone_luminance(specW);
  const float diffL = cone_luminance(diffW);
  const float transL = cone_luminance(transW);
  const float sumL = specL + diffL + transL;
  const float pSpec = sumL > 0.0f ? specL / sumL : 1.0f;
  const float pDiff = sumL > 0.0f ? diffL / sumL : 0.0f;
  const float pTrans = sumL > 0.0f ? transL / sumL : 0.0f;

  const uint3 idx = optixGetLaunchIndex();
  uint3 rngKey;
  if (params.useRayIds)
  {
    const uint2 ids = getRayIds();
    rngKey = make_uint3(ids.x, ids.y + payload->sampleIndex, payload->pbrNumBounces);
  }
  else
  {
    rngKey = make_uint3(idx.x, idx.y, getFrameNumber() + payload->sampleIndex + payload->pbrNumBounces);
  }
  const float3 rand = rnd_pcg3d(rngKey);

  payload->nextEmissive = surface.emissive;
  payload->pbrNumBounces = payload->pbrNumBounces + 1;
  const bool chooseSpec = rand.z < pSpec || (pDiff <= 0.0f && pTrans <= 0.0f);
  if (chooseSpec)
  {
    payload->bsdfValue = specW / pSpec;
    payload->coneSpreadAdd = params.cone.spreadPerAlpha * roughness * roughness;
    new_ray_dir = 2.0f * dot(n, wo) * n - wo;
  }
  else if (rand.z < pSpec + pDiff || pTrans <= 0.0f)
  {
    const float3 L = importance_sample_diffuse_ggx(surface.normal, rand.y, rand.x);
    const float3 H = normalize(wo + L);
    const float3 F = fresnel_schlick(positive_dot(wo, H), f0);
    payload->bsdfValue = (make_float3(1.0f) - F) * ((1.0f - surface.metallic) * (1.0f - surface.transmission)) * surface.diffuse / pDiff;
    payload->coneSpreadAdd = 0.0f;
    new_ray_dir = L;
  }
  else
  {
    const float eta = frontFacing >= 0.0f ? 1.0f / surface.ior : surface.ior;
    const float3 refracted = pbr_refract(-wo, n, eta);
    const bool totalInternal = dot(refracted, refracted) == 0.0f;
    payload->bsdfValue = transW / pTrans;
    payload->coneSpreadAdd = params.cone.spreadPerAlpha * roughness * roughness;
    new_ray_dir = totalInternal ? 2.0f * dot(n, wo) * n - wo : refracted;
  }
}

#endif