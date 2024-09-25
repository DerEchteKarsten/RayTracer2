#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_debug_printf : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_KHR_shader_subgroup_ballot : enable

uint RTXDI_IntegerCompact(uint x)
{
    x = (x & 0x11111111) | ((x & 0x44444444) >> 1);
    x = (x & 0x03030303) | ((x & 0x30303030) >> 2);
    x = (x & 0x000F000F) | ((x & 0x0F000F00) >> 4);
    x = (x & 0x000000FF) | ((x & 0x00FF0000) >> 8);
    return x;
}

uvec2 RTXDI_LinearIndexToZCurve(uint index)
{
    return uvec2(
        RTXDI_IntegerCompact(index),
        RTXDI_IntegerCompact(index >> 1));
}



layout( push_constant ) uniform GConst {
	uvec2 sourceSize;
    uint numDestMipLevels;
    uint sourceMipLevel;
} c_Const;

layout(binding = 0, set = 0, r16f) uniform image2D u_IntegratedMips[32];

#ifdef INPUT_ENVIRONMENT_MAP
layout(binding = 1, set = 0) uniform sampler2D t_EnvironmentMap;
const float PI = 3.1415926535;

float calcLuminance(vec3 color)
{
    return dot(color.xyz, vec3(0.299f, 0.587f, 0.114f));
}

float getPixelWeight(uvec2 position)
{
    vec3 color = texture(t_EnvironmentMap, vec2(position)).rgb;
    float luma = max(calcLuminance(color), 0);

    // Do not sample invalid colors.
    if (isinf(luma) || isnan(luma))
        return 0;
    
    // Compute the solid angle of the pixel assuming equirectangular projection.
    // We don't need the absolute value of the solid angle here, just one at the same scale as the other pixels.
    float elevation = ((float(position.y) + 0.5) / float(c_Const.sourceSize.y) - 0.5) * PI;
    float relativeSolidAngle = cos(elevation);

    const float maxWeight = 65504.0; // maximum value that can be encoded in a float16 texture

    return clamp(luma * relativeSolidAngle, 0, maxWeight);
}
#endif

shared float s_weights[16];

// Warning: do not change the group size. The algorithm is hardcoded to process 16x16 tiles.
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;  

void main()
{
    uvec2 GroupIndex = uvec2(gl_WorkGroupID.xy);
    uint ThreadIndex = uint(gl_LocalInvocationID);

    uvec2 LocalIndex = RTXDI_LinearIndexToZCurve(ThreadIndex);
    uvec2 GlobalIndex = (GroupIndex * 16) + LocalIndex;

    // Step 0: Load a 2x2 quad of pixels from the source texture or the source mip level.
    vec4 sourceWeights;
#ifdef INPUT_ENVIRONMENT_MAP
    if (c_Const.sourceMipLevel == 0)
    {
        uvec2 sourcePos = GlobalIndex.xy * 2;

        sourceWeights.x = getPixelWeight(sourcePos + ivec2(0, 0));
        sourceWeights.y = getPixelWeight(sourcePos + ivec2(0, 1));
        sourceWeights.z = getPixelWeight(sourcePos + ivec2(1, 0));
        sourceWeights.w = getPixelWeight(sourcePos + ivec2(1, 1));

        
        imageStore(u_IntegratedMips[0], ivec2(sourcePos) + ivec2(0, 0), vec4(sourceWeights.x));
        imageStore(u_IntegratedMips[0], ivec2(sourcePos) + ivec2(0, 1), vec4(sourceWeights.y));
        imageStore(u_IntegratedMips[0], ivec2(sourcePos) + ivec2(1, 0), vec4(sourceWeights.z));
        imageStore(u_IntegratedMips[0], ivec2(sourcePos) + ivec2(1, 1), vec4(sourceWeights.w));
    }
    else
#endif
    {
        uvec2 sourcePos = GlobalIndex.xy * 2;

        sourceWeights.x = imageLoad(u_IntegratedMips[c_Const.sourceMipLevel], ivec2(sourcePos) + ivec2(0, 0)).r;
        sourceWeights.y = imageLoad(u_IntegratedMips[c_Const.sourceMipLevel], ivec2(sourcePos) + ivec2(0, 1)).r;
        sourceWeights.z = imageLoad(u_IntegratedMips[c_Const.sourceMipLevel], ivec2(sourcePos) + ivec2(1, 0)).r;
        sourceWeights.w = imageLoad(u_IntegratedMips[c_Const.sourceMipLevel], ivec2(sourcePos) + ivec2(1, 1)).r;
    }

    uint mipLevelsToWrite = c_Const.numDestMipLevels - c_Const.sourceMipLevel - 1;
    if (mipLevelsToWrite < 1) return;

    // Average those weights and write out the first mip.
    float weight = (sourceWeights.x + sourceWeights.y + sourceWeights.z + sourceWeights.w) * 0.25;

    imageStore(u_IntegratedMips[c_Const.sourceMipLevel + 1], ivec2(GlobalIndex.xy), vec4(weight));

    if (mipLevelsToWrite < 2) return;

    // The following sequence is an optimized hierarchical downsampling algorithm using wave ops.
    // It assumes that the wave size is at least 16 lanes, which is true for both NV and AMD GPUs.
    // It also assumes that the threads are laid out in the group using the Z-curve pattern.

    // Step 1: Average 2x2 groups of pixels.
    uint lane = gl_SubgroupInvocationID;
    weight = (weight 
        + subgroupBroadcast(weight, lane + 1)
        + subgroupBroadcast(weight, lane + 2)
        + subgroupBroadcast(weight, lane + 3)) * 0.25;

    if ((lane & 3) == 0)
    {
        imageStore(u_IntegratedMips[c_Const.sourceMipLevel + 2], ivec2(GlobalIndex.xy >> 1), vec4(weight));
    }

    if (mipLevelsToWrite < 3) return;

    // Step 2: Average the previous results from 2 pixels away.
    weight = (weight 
        + subgroupBroadcast(weight, lane + 4)
        + subgroupBroadcast(weight, lane + 8)
        + subgroupBroadcast(weight, lane + 12)) * 0.25;

    if ((lane & 15) == 0)
    {
        imageStore(u_IntegratedMips[c_Const.sourceMipLevel + 3], ivec2(GlobalIndex.xy >> 2), vec4(weight));

        // Store the intermediate result into shared memory.
        s_weights[ThreadIndex >> 4] = weight;
    }

    if (mipLevelsToWrite < 4) return;

    barrier();

    // The rest operates on a 4x4 group of values for the entire thread group
    if (ThreadIndex >= 16)
        return;

    // Load the intermediate results
    weight = s_weights[ThreadIndex];

    // Change the output texture addressing because we'll be only writing a 2x2 block of pixels
    GlobalIndex = (GroupIndex * 2) + (LocalIndex >> 1);

    // Step 3: Average the previous results from adjacent threads, meaning from 4 pixels away.
    weight = (weight 
        + subgroupBroadcast(weight, lane + 1)
        + subgroupBroadcast(weight, lane + 2)
        + subgroupBroadcast(weight, lane + 3)) * 0.25;

    if ((lane & 3) == 0)
    {
        imageStore(u_IntegratedMips[c_Const.sourceMipLevel + 4], ivec2(GlobalIndex.xy), vec4(weight));
    }

    if (mipLevelsToWrite < 5) return;

    // Step 4: Average the previous results from 8 pixels away.
    weight = (weight 
        + subgroupBroadcast(weight, lane + 4)
        + subgroupBroadcast(weight, lane + 8)
        + subgroupBroadcast(weight, lane + 12)) * 0.25;

    if (lane == 0)
    {
        imageStore(u_IntegratedMips[c_Const.sourceMipLevel + 5], ivec2(GlobalIndex.xy >> 1), vec4(weight));
    }
}