layout(location = 0) out vec4 outColor;

layout (binding = 0, set=0, rgba32f) uniform readonly image2D inputImage;

#define AGX_LOOK 0

// Mean error^2: 3.6705141e-06
vec3 agxDefaultContrastApprox(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  
  return + 15.5     * x4 * x2
         - 40.14    * x4 * x
         + 31.96    * x4
         - 6.868    * x2 * x
         + 0.4298   * x2
         + 0.1191   * x
         - 0.00232;
}

vec3 agx(vec3 val) {
  const mat3 agx_mat = mat3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992,  0.878468636469772,  0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    
  const float min_ev = -12.47393f;
  const float max_ev = 4.026069f;

  // Input transform (inset)
  val = agx_mat * val;
  
  // Log2 space encoding
  val = clamp(log2(val), min_ev, max_ev);
  val = (val - min_ev) / (max_ev - min_ev);
  
  // Apply sigmoid function approximation
  val = agxDefaultContrastApprox(val);

  return val;
}

vec3 agxEotf(vec3 val) {
  const mat3 agx_mat_inv = mat3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    
  // Inverse input transform (outset)
  val = agx_mat_inv * val;
  
  // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
  // NOTE: We're linearizing the output here. Comment/adjust when
  // *not* using a sRGB render target
  val = pow(val, vec3(2.2));

  return val;
}

vec3 agxLook(vec3 val) {
  const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
  float luma = dot(val, lw);
  
  // Default
  vec3 offset = vec3(0.0);
  vec3 slope = vec3(1.0);
  vec3 power = vec3(1.0);
  float sat = 1.0;
 
#if AGX_LOOK == 1
  // Golden
  slope = vec3(1.0, 0.9, 0.5);
  power = vec3(0.8);
  sat = 0.8;
#elif AGX_LOOK == 2
  // Punchy
  slope = vec3(1.0);
  power = vec3(1.35, 1.35, 1.35);
  sat = 1.4;
#endif
  
  // ASC CDL
  val = pow(val * slope + offset, power);
  return luma + sat * (val - luma);
}

const float gamma = 2.2;

void main() {
    vec3 col = imageLoad(inputImage, ivec2(gl_FragCoord)).rgb;

    col = agx(col);
    col = agxLook(col);
    col = agxEotf(col);
    vec3 gamma_cor = pow(col, vec3(1.0 / gamma));
    outColor = vec4(col, 1.0);
}