precision highp float;

varying vec2 vUv;
uniform sampler2D rgbaTexture;

void main() {
    vec4 color = texture2D(rgbaTexture, vUv);
    gl_FragColor = vec4(color.rgb, 1.0 - color.a);
}