
bool ShadeSurfaceWithLightSample(
    inout RTXDI_DIReservoir reservoir,
    RAB_Surface surface,
    RAB_LightSample lightSample,
    bool previousFrameTLAS,
    bool enableVisibilityReuse,
    out float3 diffuse,
    out float3 specular,
    out float lightDistance)
{
    diffuse = vec3(0);
    specular = vec3(0);
    lightDistance = 0;

    if (lightSample.solidAnglePdf <= 0)
        return false;

    bool needToStore = false;
    if (g_Const.restirDI.shadingParams.enableFinalVisibility == 1)
    {
        float3 visibility = vec3(0);
        bool visibilityReused = false;

        if (g_Const.restirDI.shadingParams.reuseFinalVisibility == 1 && enableVisibilityReuse)
        {
            RTXDI_VisibilityReuseParameters rparams;
            rparams.maxAge = g_Const.restirDI.shadingParams.finalVisibilityMaxAge;
            rparams.maxDistance = g_Const.restirDI.shadingParams.finalVisibilityMaxDistance;

            visibilityReused = RTXDI_GetDIReservoirVisibility(reservoir, rparams, visibility);
        }

        if (!visibilityReused)
        {
            RayDesc ray = setupVisibilityRay(surface, lightSample.position, 0.01);
            trace(ray);
            visibility = vec3(p.geometryIndex == ~0u);
            RTXDI_StoreVisibilityInDIReservoir(reservoir, visibility, g_Const.restirDI.temporalResamplingParams.discardInvisibleSamples == 1);
            needToStore = true;
        }

        lightSample.radiance *= visibility;
    }
    lightSample.radiance *= RTXDI_GetDIReservoirInvPdf(reservoir) / lightSample.solidAnglePdf;

    if (lightSample.radiance.x > 0.0 || lightSample.radiance.y > 0.0 || lightSample.radiance.z > 0.0)
    {
        SplitBrdf brdf = EvaluateBrdf(surface, lightSample.position);

        diffuse = brdf.demodulatedDiffuse * lightSample.radiance;
        specular = brdf.specular * lightSample.radiance;

        lightDistance = length(lightSample.position - surface.worldPos);
    }

    return needToStore;
}


void StoreShadingOutput(
    ivec2 pixelPosition,
    float viewDepth,
    float roughness,
    vec3 diffuse,
    vec3 specular,
    float lightDistance,
    bool isFirstPass
    )
{
    float diffuseHitT = lightDistance;
    float specularHitT = lightDistance;

    if (!isFirstPass)
    {
        vec4 priorDiffuse = imageLoad(DiffuseLighting, pixelPosition);
        vec4 priorSpecular = imageLoad(SpecularLighting, pixelPosition);

        if (calcLuminance(diffuse) > calcLuminance(priorDiffuse.rgb) || lightDistance == 0)
            diffuseHitT = priorDiffuse.w;

        if (calcLuminance(specular) > calcLuminance(priorSpecular.rgb) || lightDistance == 0)
            specularHitT = priorSpecular.w;
        
        diffuse += priorDiffuse.rgb;
        specular += priorSpecular.rgb;
    }

    imageStore(DiffuseLighting, pixelPosition, vec4(diffuse, diffuseHitT));
    imageStore(SpecularLighting, pixelPosition, vec4(specular, specularHitT));
}