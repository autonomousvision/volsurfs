precision highp float;

varying vec3 vNormal;
varying vec3 vViewDir;

void main() {
    vec3 color = vec3(1.0, 1.0, 1.0);
    vec3 normal = vNormal;
    vec3 view_dir = vViewDir;
    float dot_product = max(0.0, dot(normal, -view_dir));

    float alpha = 1.0;
    // float threshold = 0.5;
    float threshold = 10.0;
    // if (dot_product < threshold)
    // {
    //     // Exponential decay from 1 to 0 for x < 0.1
    //     // Adjust the exponent to control the rate of decay
    //     // alpha = exp(-10.0 * (threshold - dot_product) / threshold);
    //     // alpha = (1.0 / threshold) * dot_product;
    // }
    alpha = (1.0 / (1.0 + exp(-threshold * dot_product))) * 2.0 - 1.0;
    // color = color * alpha;

    gl_FragColor = vec4(1.0 - alpha, 0.0, 0.0, 1.0);
}