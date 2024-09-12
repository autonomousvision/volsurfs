precision highp float;

varying vec3 vNormal;
varying vec3 vViewDir;

void main() {
    vec3 color = vViewDir * 0.5 + 0.5;
    gl_FragColor = vec4(color, 1.0);
}