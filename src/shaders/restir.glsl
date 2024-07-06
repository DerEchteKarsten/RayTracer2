struct RISReservoir
{
    vec3 Y; // index of most important light
    float W_y; // light weight
    float W_sum; // sum of all weights for all lights processed
    float M; // number of lights processed for this reservoir
};

bool UpdateReservoir(inout RISReservoir reservoir, vec3 X, float w, float c, inout uint rngState)
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


void UpdateReservoir(uint slot, Sample s_n, float w_n, inout uint rngState) {
    uint org = atomicAdd(values[slot].w, int(w_n * 1000)) + int(w_n * 1000);
    if (RandomValue(rngState) < w_n / (float(org) / 1000.0)) {
        values[slot].z = s_n;
    }
    return;
}

float pdf(Sample s, vec3 dir) {
    return clamp(dot(s.normal,dir), 0.0, 1.0)/PI;
}

float calcLuminance(vec3 color)
{
    return dot(color.xyz, vec3(0.299f, 0.587f, 0.114f));
}

float p_hat(Sample s) {
    return calcLuminance(s.radiance * dot(s.normal, s.dir) * s.color);
}

vec3 polar_form(float theta, float thi) {
    return vec3(sin(theta)*cos(thi), sin(theta)*sin(thi), cos(theta));
}


bool IsReservoirValid(in RISReservoir reservoir) {
    return true;
}

const int random_lights = 1;


float atan2(in float y, in float x)
{
	if (x > 0) {
		return atan(y/x);
	}
	if (x < 0 && y >= 0){
		return atan(y/x) + PI;
	}
	if (x < 0 && y < 0) {
		return atan(y/x) - PI;
	}
	if (x == 0 && y > 0) {
		return PI/2;
	}
	if (x == 0 && y < 0) {
		return -PI/2;
	}
	if (x == 0 && y == 0) {
		return 0;
	}
	return 0;
}

vec2 get_uv(vec3 dir) {
    float u = (0.5 + atan2(dir.x, dir.z)/(2*PI));
    float v = (0.5 - asin(dir.y)/PI);
    return vec2(u,v);
}

vec3 RIS(in vec3 origin, in vec3 normal, in vec3 color, inout uint rngState) {
    RISReservoir reservoir = RISReservoir(vec3(0.0), 0.0, 0.0, 0.0);
    float p_hat = 0;
    
    //initial selection of 1 light of M
    for (uint i = 0; i < random_lights; i++)
    {
        vec3 lightIndex = normalize(normal + RandomDirection(rngState));
        float pdf = dot(normal, lightIndex);
        vec3 radiance = texture(skybox, get_uv(lightIndex)).rgb * color; 
        p_hat = length(radiance); //brdf
        
        float w = p_hat / pdf;
        
        UpdateReservoir(reservoir, lightIndex, w, 1, rngState);
    }
    vec3 radiance = vec3(0.0);
    if (IsReservoirValid(reservoir))
    {
        vec3 radiance =  max(dot(normal, reservoir.Y), 0.0) * texture(skybox, get_uv(reservoir.Y)).rgb * color;
        HitInfo hit_info;
        bool hit = ray_cast(origin + normal * 0.0001, reservoir.Y, hit_info);

        float shadowFactor = float(hit); // is this ray occluded?
                    
        //pixel radiance with the selected light
        radiance = shadowFactor * radiance;
    
        p_hat = length(radiance);
            
        // calculate the weight of this light
        reservoir.W_y = p_hat > 0.0 ? (1.0 / p_hat) * reservoir.W_sum / reservoir.M : 0.0;
        
        // apply it to the radiance to get the final radiance.
        radiance  *=  reservoir.W_y;
    }
    return radiance;
}

