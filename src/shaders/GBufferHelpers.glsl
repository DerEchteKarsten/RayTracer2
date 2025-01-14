#ifndef GBufferHelpers
#define GBufferHelpers
#define mul(a, b) (b * a)

struct RayDesc {
    vec3 Origin;
    vec3 Direction;
    float TMin;
    float TMax;
};

RayDesc setupPrimaryRay(uvec2 pixelPosition, PlanarViewConstants view)
{
    const vec2 pixelCenter = vec2(pixelPosition.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter/vec2(view.viewportSize);
    vec2 d = inUV * 2.0 - 1.0;
    vec2 dir = inUV * 2.0 - 1.0;
    vec4 target = view.matClipToView * vec4(dir.x, dir.y, 1, 1);
    vec4 direction = view.matViewToWorld*vec4(normalize(target.xyz), 0);

    RayDesc ray;
    ray.Origin = view.cameraDirectionOrPosition.xyz;
    ray.Direction = direction.xyz;
    ray.TMin = 0;
    ray.TMax = BACKGROUND_DEPTH;
    return ray;
}

vec3 getMotionVector(
    PlanarViewConstants view,
    PlanarViewConstants viewPrev,
    vec3 objectSpacePosition,
    vec3 prevObjectSpacePosition
)
{   
    vec3 worldSpacePosition = objectSpacePosition;
    vec3 prevWorldSpacePosition = prevObjectSpacePosition;

    vec4 clipPos = view.matWorldToClip * vec4(worldSpacePosition, 1.0);
    clipPos.xyz /= clipPos.w;
    vec4 prevClipPos = viewPrev.matWorldToClip * vec4(prevWorldSpacePosition, 1.0);
    prevClipPos.xyz /= prevClipPos.w;

    if (clipPos.w <= 0 || prevClipPos.w <= 0)
        return vec3(0);

    vec3 motion;
    motion.xy = (prevClipPos.xy - clipPos.xy) / 2.0 * view.viewportSize;
    motion.xy += (view.pixelOffset - viewPrev.pixelOffset);
    motion.z = prevClipPos.w - clipPos.w;
    return motion;
}

vec3 viewDepthToWorldPos(
    PlanarViewConstants view,
    ivec2 pixelPosition,
    float viewDepth)
{
    const vec2 pixelCenter = vec2(pixelPosition.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter/vec2(view.viewportSize);
    vec2 d = inUV * 2.0 - 1.0;
    vec2 dir = inUV * 2.0 - 1.0;
    vec4 target = view.matClipToView * vec4(dir.x, dir.y, 1, 1);
    vec4 direction = view.matViewToWorld*vec4(normalize(target.xyz), 0);

    return view.cameraDirectionOrPosition.xyz + direction.xyz * viewDepth;
}

vec3 convertMotionVectorToPixelSpace(
    PlanarViewConstants view,
    PlanarViewConstants viewPrev,
    ivec2 pixelPosition,
    vec3 motionVector)
{
    vec2 curerntPixelCenter = vec2(pixelPosition.xy) + 0.5;
    vec2 previousPosition = curerntPixelCenter + motionVector.xy;
    previousPosition *= viewPrev.viewportSize * view.viewportSizeInv;
    motionVector.xy = previousPosition - curerntPixelCenter;
    return motionVector;
}

#endif