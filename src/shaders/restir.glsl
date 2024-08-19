struct Sample {
    vec3 p_pos, p_normal;
    vec3 s_pos, s_normal;
    vec3 radiance;
};

struct Reservoir {
    Sample z;
    float weight_sum;
    int num_sampels;
    float weight;
};

void update(inout Reservoir res, Sample s_new, float weight, inout uint rngState) {
    res.weight_sum += weight;
    res.num_sampels += 1;
    if(RandomValue(rngState) < weight / res.weight_sum) {
        res.z = s_new;
    }
}

void merge(inout Reservoir self, Reservoir other, float p_hat, inout uint rngState) {
    int M0 = self.num_sampels;
    update(self, other.z, p_hat * other.weight * other.num_sampels, rngState);
    self.num_sampels = M0 + other.num_sampels;
}