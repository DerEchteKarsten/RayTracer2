#extension GL_GOOGLE_include_directive : enable

#include "./common.glsl"

struct Sample {
    vec3 origin;
    vec3 originNormal;
    vec3 samplePoint;
    vec3 samplePointNormal;
    vec3 radiance;
    uint random;
};

struct Reservoir {
    Sample s;
    float weightOfReservoir;
    uint numCandidateSamples;
    float weightOfSample;
};

void update_reservoir(inout Reservoir self, Sample new_sample, float new_weight, inout uint state) {
    self.weightOfReservoir += new_weight;
    self.numCandidateSamples += 1;

    if (RandomValue(state) < new_weight / self.weightOfReservoir) {
        self.s = new_sample;
    } 
}

void merge_reservoir(inout Reservoir self, Reservoir other, float target_pdf, inout uint state) {
    uint s = self.numCandidateSamples;
    update_reservoir(self, other.s, target_pdf * other.weightOfSample * other.numCandidateSamples, state);
    self.numCandidateSamples = s + other.numCandidateSamples;
}