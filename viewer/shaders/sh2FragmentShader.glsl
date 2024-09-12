precision highp float;
precision mediump sampler2DArray;

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vViewDir;

const float C0 = 0.28209479177387814;
const float C1 = 0.4886025119029199;
const float C2[5] = float[5](1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396);
const float C3[7] = float[7](-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435);
const int NR_COEFFS[4] = int[4](1, 3, 5, 7);
const int SH_DEG = 2;
const int NR_TOT_SH_COEFFS = (SH_DEG + 1) * (SH_DEG + 1); // 9

uniform sampler2DArray sh_0_coeffs_texture_3D;
uniform sampler2DArray sh_1_coeffs_texture_3D;
uniform sampler2DArray sh_2_coeffs_texture_3D;
uniform bool inverse_alpha;
uniform bool ignore_alpha;
uniform bool use_alpha_decay;
uniform vec2 values_range;

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float fastSigmoid(float x) {
    return 0.5 * (1.0 + tanh(0.5 * x));
}

float eval_sh(
    vec3 viewDir,
    float sh0Coeffs[4 * NR_COEFFS[0]],
    float sh1Coeffs[4 * NR_COEFFS[1]],
    float sh2Coeffs[4 * NR_COEFFS[2]],
    int channel
) {
    
    float x = viewDir.x;
    float y = viewDir.y;
    float z = viewDir.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;
    float result = 0.0;

    // deg 0
    result += C0 * sh0Coeffs[channel];
    
    // deg 1
    result -= C1 * y * sh1Coeffs[0 * 4 + channel];
    result += C1 * z * sh1Coeffs[1 * 4 + channel];
    result -= C1 * x * sh1Coeffs[2 * 4 + channel];

    // deg 2
    result += C2[0] * xy * sh2Coeffs[0 * 4 + channel];
    result += C2[1] * yz * sh2Coeffs[1 * 4 + channel];
    result += C2[2] * (2.0 * zz - xx - yy) * sh2Coeffs[2 * 4 + channel];
    result += C2[3] * xz * sh2Coeffs[3 * 4 + channel];
    result += C2[4] * (xx - yy) * sh2Coeffs[4 * 4 + channel];

    return result;
}

void main() {

    // read sh coeffs from textures

    float v_min = values_range[0];
    float v_max = values_range[1];

    // deg 0
    float sh_coeffs_deg_0[4 * NR_COEFFS[0]];
    for (int i = 0; i < NR_COEFFS[0]; i++) {
        // read coeffs batch
        vec4 sh_0_coeffs_batch = texture(sh_0_coeffs_texture_3D, vec3(vUv, float(i)));
        sh_0_coeffs_batch = sh_0_coeffs_batch * (v_max - v_min) + v_min;
        for (int j = 0; j < 4; j++) {
            sh_coeffs_deg_0[j] = sh_0_coeffs_batch[j];
        }
        // r0, g0, b0, a0
    }

    // deg 1
    float sh_coeffs_deg_1[4 * NR_COEFFS[1]];
    for (int i = 0; i < NR_COEFFS[1]; i++) {  // 3
        // read coeffs batch
        vec4 sh_1_coeffs_batch = texture(sh_1_coeffs_texture_3D, vec3(vUv, float(i)));
        sh_1_coeffs_batch = sh_1_coeffs_batch * (v_max - v_min) + v_min;
        // ri, gi, bi, ai
        for (int j = 0; j < 4; j++) {
            sh_coeffs_deg_1[i * 4 + j] = sh_1_coeffs_batch[j];
        }
        // r0, g0, b0, a0, ..., r2, g2, b2, a2 
    }

    // deg 2
    float sh_coeffs_deg_2[4 * NR_COEFFS[2]];
    for (int i = 0; i < NR_COEFFS[2]; i++) {  // 5
        // read coeffs batch
        vec4 sh_2_coeffs_batch = texture(sh_2_coeffs_texture_3D, vec3(vUv, float(i)));
        sh_2_coeffs_batch = sh_2_coeffs_batch * (v_max - v_min) + v_min;
        for (int j = 0; j < 4; j++) {
            sh_coeffs_deg_2[i * 4 + j] = sh_2_coeffs_batch[j];
        }
        // r0, g0, b0, a0, ..., r4, g4, b4, a4
    }
    
    // float sh_coeffs_deg_3[4 * NR_COEFFS[3]];

    // debug: const view dir
    // vec3 const_vViewDir = vec3(1.0, 0.0, 0.0);

    float channels[4];
    for (int channel_idx = 0; channel_idx < 4; channel_idx++) {
        float channel_val = eval_sh(
            vViewDir,
            sh_coeffs_deg_0,
            sh_coeffs_deg_1,
            sh_coeffs_deg_2,
            channel_idx
        );
        channels[channel_idx] = sigmoid(channel_val);
    }

    // apply alpha decay
    if (use_alpha_decay) {
        vec3 normal = vNormal;
        vec3 view_dir = vViewDir;
        float dot_product = max(0.0, dot(normal, -view_dir));
        float alpha = 1.0;
        // float threshold = 0.5;
        float threshold = 10.0;
        if (dot_product < threshold)
        {
            // Exponential decay from 1 to 0 for x < 0.1
            // Adjust the exponent to control the rate of decay
            // alpha = exp(-10.0 * (threshold - dot_product) / threshold);
            // alpha = (1.0 / threshold) * dot_product;
            alpha = (1.0 / (1.0 + exp(-threshold * dot_product))) * 2.0 - 1.0;
        }
        channels[3] = channels[3] * alpha;
    }

    if (ignore_alpha)
        channels[3] = 1.0;

    if (inverse_alpha)
        channels[3] = 1.0 - channels[3];

    gl_FragColor = vec4(channels[0], channels[1], channels[2], channels[3]);
}