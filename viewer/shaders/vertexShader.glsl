// #ifdef GL_FRAGMENT_PRECISION_HIGH
//     precision highp float;
// #else
//     precision highp float;
// #endif

// attribute vec3 position;  // Vertex position attribute in object space
// attribute vec2 uv;        // Texture coordinates attribute
// attribute vec3 normal;    // Normal vector attribute in object space

// uniform mat4 modelViewMatrix;  // Model-view transformation matrix
// uniform mat4 projectionMatrix; // Projection transformation matrix
// uniform vec3 cameraPosition;   // Camera position in world space
// uniform mat4 modelMatrix;      // Model transformation matrix

varying vec2 vUv;        // Pass UV to fragment shader
varying vec3 vNormal;    // Pass normal to fragment shader (in view space)
varying vec3 vViewDir;   // Pass view direction to fragment shader (in view space)

void main() {

    // rotate by 90 degrees clockwise on X axis
    mat4 rotationMatrix = mat4(
                                1.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, -1.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 1.0
                            );
    
    vUv = uv; // Pass texture coordinates to fragment shader
    vNormal = (rotationMatrix * vec4(normal, 1.0)).xyz;

    // 
    vec3 camera_position = (rotationMatrix * vec4(cameraPosition, 1.0)).xyz;
    vec3 point_position = (rotationMatrix * vec4(position, 1.0)).xyz;
    vViewDir = -normalize(camera_position - point_position);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}