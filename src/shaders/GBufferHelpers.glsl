
struct RayDesc {
    vec3 Origin;
    vec3 Direction;
    float TMin;
    float TMax;
};

RayDesc setupPrimaryRay(uvec2 pixelPosition, PlanarViewConstants view)
{
    vec2 uv = (vec2(pixelPosition) + 0.5) * view.viewportSizeInv;
    vec4 clipPos = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, (1.0 / 256.0), 1);
    vec4 worldPos = clipPos * view.matClipToWorld;
    worldPos.xyz /= worldPos.w;

    RayDesc ray;
    ray.Origin = view.cameraDirectionOrPosition.xyz;
    ray.Direction = normalize(worldPos.xyz - ray.Origin);
    ray.TMin = 0;
    ray.TMax = 1000;
    return ray;
}

vec3 getMotionVector(
    PlanarViewConstants view,
    PlanarViewConstants viewPrev,
    vec3 objectSpacePosition,
    vec3 prevObjectSpacePosition,
    out float o_viewDepth)
{
    vec3 worldSpacePosition = vec4(objectSpacePosition, 1.0).xyz;
    vec3 prevWorldSpacePosition = vec4(prevObjectSpacePosition, 1.0).xyz;

    vec4 clipPos = vec4(worldSpacePosition, 1.0) * view.matWorldToClip;
    clipPos.xyz /= clipPos.w;
    vec4 prevClipPos = vec4(prevWorldSpacePosition, 1.0) * viewPrev.matWorldToClip;
    prevClipPos.xyz /= prevClipPos.w;

    o_viewDepth = clipPos.w;

    if (clipPos.w <= 0 || prevClipPos.w <= 0)
        return vec3(0);

    vec2 windowPos = clipPos.xy * view.clipToWindowScale + view.clipToWindowBias;
    vec2 prevWindowPos = prevClipPos.xy * viewPrev.clipToWindowScale + viewPrev.clipToWindowBias;

    vec3 motion;
    motion.xy = prevWindowPos.xy - windowPos.xy;
    motion.xy += (view.pixelOffset - viewPrev.pixelOffset);
    motion.z = prevClipPos.w - clipPos.w;
    return motion;
}

vec3 viewDepthToWorldPos(
    PlanarViewConstants view,
    ivec2 pixelPosition,
    float viewDepth)
{
    vec2 uv = (vec2(pixelPosition) + 0.5) * view.viewportSizeInv;
    vec4 clipPos = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1);
    vec4 viewPos = clipPos * view.matClipToView;
    viewPos.xy /= viewPos.z;
    viewPos.zw = vec2(1.0);
    viewPos.xyz *= viewDepth;
    return (viewPos * view.matViewToWorld).xyz;
}