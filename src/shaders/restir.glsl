float calcLuminance(vec3 color)
{
    return dot(color.xyz, vec3(0.299f, 0.587f, 0.114f));
}

vec3 polar_form(float theta, float thi) {
    return vec3(sin(theta)*cos(thi), sin(theta)*sin(thi), cos(theta));
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
