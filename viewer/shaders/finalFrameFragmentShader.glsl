precision highp float;
precision mediump sampler2D;

const int MAX_NR_MESHES = 9;

uniform vec3 bgColor;
uniform int nrMeshes;
uniform vec2 screenSize; // (width, height)
// uniform sampler2D bgDepthTexture; // float
// uniform sampler2D bgRGBATexture; // rgb
uniform sampler2D meshesRGBATextures[MAX_NR_MESHES]; // single meshes render buffers
// uniform samples2D meshDepthTexture; // single meshes depth buffers

void main() {

    // float depth_bg = texture(bgDepthTexture, gl_FragCoord.xy / screenSize).r;
    // float depth_mesh = texture(meshDepthTexture, gl_FragCoord.xy / screenSize).r;
    // vec3 c_bg = texture(bgRGBATexture, gl_FragCoord.xy / screenSize).rgb;
    vec3 c_bg = bgColor;
    // vec3 c_bg = vec3(0.0, 0.0, 0.0);

    // if (depth_bg < depth) {

    //     gl_FragColor = vec4(c_bg, 1.0);

    // } else { 

    vec3 c_meshes[MAX_NR_MESHES];
    float a_meshes[MAX_NR_MESHES];
    vec3 blended_color = c_bg;

    // read colors and alpha buffers and alpha blend them

    if (nrMeshes > 0) {
        c_meshes[0] = texture(meshesRGBATextures[0], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[0] = texture(meshesRGBATextures[0], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[0] * a_meshes[0] + (1.0 - a_meshes[0]) * blended_color;
    }
    if (nrMeshes > 1) {
        c_meshes[1] = texture(meshesRGBATextures[1], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[1] = texture(meshesRGBATextures[1], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[1] * a_meshes[1] + (1.0 - a_meshes[1]) * blended_color;
    }
    if (nrMeshes > 2) {
        c_meshes[2] = texture(meshesRGBATextures[2], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[2] = texture(meshesRGBATextures[2], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[2] * a_meshes[2] + (1.0 - a_meshes[2]) * blended_color;
    }
    if (nrMeshes > 3) {
        c_meshes[3] = texture(meshesRGBATextures[3], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[3] = texture(meshesRGBATextures[3], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[3] * a_meshes[3] + (1.0 - a_meshes[3]) * blended_color;
    }
    if (nrMeshes > 4) {
        c_meshes[4] = texture(meshesRGBATextures[4], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[4] = texture(meshesRGBATextures[4], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[4] * a_meshes[4] + (1.0 - a_meshes[4]) * blended_color;
    }
    if (nrMeshes > 5) {
        c_meshes[5] = texture(meshesRGBATextures[5], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[5] = texture(meshesRGBATextures[5], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[5] * a_meshes[5] + (1.0 - a_meshes[5]) * blended_color;
    }
    if (nrMeshes > 6) {
        c_meshes[6] = texture(meshesRGBATextures[6], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[6] = texture(meshesRGBATextures[6], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[6] * a_meshes[6] + (1.0 - a_meshes[6]) * blended_color;
    }
    if (nrMeshes > 7) {
        c_meshes[7] = texture(meshesRGBATextures[7], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[7] = texture(meshesRGBATextures[7], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[7] * a_meshes[7] + (1.0 - a_meshes[7]) * blended_color;
    }
    if (nrMeshes > 8) {
        c_meshes[8] = texture(meshesRGBATextures[8], gl_FragCoord.xy / screenSize).rgb;
        a_meshes[8] = texture(meshesRGBATextures[8], gl_FragCoord.xy / screenSize).a;
        blended_color = c_meshes[8] * a_meshes[8] + (1.0 - a_meshes[8]) * blended_color;
    }

    gl_FragColor = vec4(blended_color, 1.0);
}