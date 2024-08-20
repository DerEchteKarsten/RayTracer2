
struct PrimarySurfaceOutput
{
    RAB_Surface surface;
    vec3 motionVector;
    vec3 emissiveColor;
};

PrimarySurfaceOutput TracePrimaryRay(int2 pixelPosition)
{
    RayDesc ray = setupPrimaryRay(pixelPosition, g_Const.view);
    
    trace(ray);

    PrimarySurfaceOutput result;
    result.surface = RAB_EmptySurface();
    result.motionVector = vec3(0);
    result.emissiveColor = vec3(0);

    if (!p.missed)
    { 
        result.motionVector = getMotionVector(g_Const.view, g_Const.prevView, 
            p.hitPoint, p.hitPoint, result.surface.viewDepth);
        
        result.surface.worldPos = p.hitPoint;
        result.surface.normal = p.hitNormal;
        result.surface.geoNormal = p.hitNormal;
        result.surface.diffuseAlbedo = p.color.rgb;
        result.surface.specularF0 = vec3(p.metallicFactor);
        result.surface.roughness = p.roughness;
        result.surface.viewDir = -ray.Direction;
        result.surface.diffuseProbability = getSurfaceDiffuseProbability(result.surface);
        result.emissiveColor = p.emission;
    }

    return result;
}

RAB_Surface TraceRayToSurface(RayDesc ray)
{    
    trace(ray);

    RAB_Surface surface;

    if (!p.missed)
    {  
        surface.worldPos = p.hitPoint;
        surface.normal = p.hitNormal;
        surface.geoNormal = p.hitNormal;
        surface.diffuseAlbedo = p.color.rgb;
        surface.specularF0 = vec3(p.metallicFactor);
        surface.roughness = p.roughness;
        surface.viewDir = -ray.Direction;
        surface.diffuseProbability = getSurfaceDiffuseProbability(surface);
    }

    return surface;
}