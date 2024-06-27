#include "./common.glsl"

struct Reservoir
{
    uint Y; // index of most important light
    float W_y; // light weight
    float W_sum; // sum of all weights for all lights processed
    float M; // number of lights processed for this reservoir
};
 
bool UpdateReservoir(inout Reservoir reservoir, uint X, float w, float c, inout uint rngState)
{
    reservoir.W_sum += w;
    reservoir.M += c;
 
    if ( RandomValue(rngState) < w / reservoir.W_sum  )
    {
        reservoir.Y = X;
        return true;
    }
 
    return false;
}

layout(std430, set = 0, binding = 4) readonly buffer uLightBuffer { vec4 lights[]; };

vec3 polar_form(float theta, float thi) {
    return vec3(sin(theta)*cos(thi), sin(theta)*sin(thi), cos(theta));
}


bool IsReservoirValid(in Reservoir reservoir) {
    return true;
}

vec3 light_radiance(vec3 dir, vec3 normal, vec3 color, float area) {
    return max(dot(normal, dir), 0.0) * color * vec3(area);
}

mat3 angleAxis3x3(float angle, vec3 axis)
{
    float c, s;
    s = sin(angle);
    c = cos(angle);

    float t = 1 - c;
    float x = axis.x;
    float y = axis.y;
    float z = axis.z;

    return mat3(
        t * x * x + c,      t * x * y - s * z,  t * x * z + s * y,
        t * x * y + s * z,  t * y * y + c,      t * y * z - s * x,
        t * x * z - s * y,  t * y * z + s * x,  t * z * z + c
    );
}

vec3 getConeSample(inout uint randSeed, vec3 direction, float coneAngle) {
    float cosAngle = cos(coneAngle);

    // Generate points on the spherical cap around the north pole [1].
    // [1] See https://math.stackexchange.com/a/205589/81266
    float z = RandomValue(randSeed) * (1.0f - cosAngle) + cosAngle;
    float phi = RandomValue(randSeed) * 2.0f * PI;

    float x = sqrt(1.0f - z * z) * cos(phi);
    float y = sqrt(1.0f - z * z) * sin(phi);
    vec3 north = vec3(0.f, 0.f, 1.f);

    // Find the rotation axis `u` and rotation angle `rot` [1]
    vec3 axis = normalize(cross(north, normalize(direction)));
    float angle = acos(dot(normalize(direction), north));

    // Convert rotation axis and angle to 3x3 rotation matrix [2]
    mat3 R = angleAxis3x3(angle, axis);

    return vec3(x, y, z) * R;
}
const int random_lights = 8;

vec3 RIS(inout Reservoir reservoir, in vec3 origin, in vec3 normal, in vec3 color, inout uint rngState) {

    float pdf = 1.0 / lights.length();
    float p_hat = 0;
    
    //initial selection of 1 light of M
    for (uint i = 0; i < random_lights; i++)
    {
        uint lightIndex = uint(RandomValue(rngState) * (lights.length() - 1));
        vec3 light = lights[i].xyz; 
        p_hat = length(light_radiance(light, normal, color, 1.0 / lights.length())); //brdf
        
        float w = p_hat / pdf;
        
        UpdateReservoir(reservoir, lightIndex, w, 1, rngState);
    }
    vec3 radiance = vec3(0.0);
    if (IsReservoirValid(reservoir))
    {
        vec3 light = lights[reservoir.Y].xyz;
        vec3 o_pos, o_normal, o_color;
        bool hit = ray_cast(origin + normal * 0.0001, light, o_pos, o_normal, o_color) == INFINITY;

        float shadowFactor = float(hit); // is this ray occluded?
                    
        //pixel radiance with the selected light
        radiance = shadowFactor * light_radiance(light, normal, color, 1.0 / lights.length());
    
        p_hat = length(radiance);
            
        // calculate the weight of this light
        reservoir.W_y = p_hat > 0.0 ? (1.0 / p_hat) * reservoir.W_sum / reservoir.M : 0.0;
        
        // apply it to the radiance to get the final radiance.
        radiance  *=  reservoir.W_y;
    }
    return radiance;
}