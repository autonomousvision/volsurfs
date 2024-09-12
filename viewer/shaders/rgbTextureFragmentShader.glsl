precision highp float;
precision mediump sampler2D;

varying vec2 vUv;
uniform sampler2D rgbaTexture;
uniform bool inverse_alpha;

void main() {
    vec4 color = texture2D(rgbaTexture, vUv);
    // vec3 viewDir = -1.0 * vViewDir;
    // float dotProduct = max(0.0, dot(vNormal, viewDir));
    // color = color * dotProduct;
    if (inverse_alpha) {
        gl_FragColor = vec4(color.rgb, 0.0);
    } else {
        gl_FragColor = vec4(color.rgb, 1.0);  // 1.0 - color.a
    }
}