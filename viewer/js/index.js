import _, { map, reject } from 'lodash';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import Stats from 'stats.js'
import * as dat from 'dat.gui';
import { loadShader } from './utils/shaderUtils.js';
import { loadCameras } from './utils/camerasUtils.js';
import { captureScreenshot } from './utils/screenshotUtils.js';
import { loadTexturedMeshes } from './utils/meshesUtils.js';

const MAX_NR_MESHES = 9;

// set scene name as get parameter
var url = new URL(window.location.href);
console.log(url);
if (url.searchParams.get('scene_name') != null) {
    var scene_name = url.searchParams.get('scene_name');
} else {
    var scene_name = "plushy";
}

if (url.searchParams.get('nr_surfs') != null) {
    var nr_surfs = Number(url.searchParams.get('nr_surfs'));
} else {
    var nr_surfs = 5;
}

if (url.searchParams.get('textures_mode') != null) {
    var textures_mode = url.searchParams.get('textures_mode');
} else {
    var textures_mode = "2048_linear";
}

if (url.searchParams.get('sh_deg') != null) {
    var sh_deg = Number(url.searchParams.get('sh_deg'));
} else {
    var sh_deg = 3;
}

const sceneName = scene_name;

const nrMeshes = nr_surfs;
console.log('nrMeshes:', nrMeshes);

const shDeg = sh_deg;
console.log('shDeg:', shDeg);

const texturesMode = textures_mode;

const sceneDir = sceneName + "_" + nrMeshes + "_" + texturesMode + "_" + shDeg + "_deg";
console.log('sceneDir:', sceneDir);



const sceneDirPath =  'assets/scenes/' + scene_name + '/' + sceneDir + '/';

let canvasContainer, canvasElement, statsElement, guiElement, loadingScreen;
let activeCamera, renderer, controls, stats;
let orbitCamera;
let datasetCameras;
let canRender = true;

let solidScene; // scene for rendering solid objects occluding volsurfs
let sceneDebug; // scene for debugging specific things
let sceneWireframe; // scene for meshes wireframe and more
let scenePost; // scene for post processing the final frame
let postCamera;
let postMaterial;
let scenesMeshes = [];
// let bgRenderTarget;
let meshesRenderTargets = [];
let prec = 0;

let rendererCfg = {
    benchmark_width: 1280,
    benchmark_height: 720,
    dataset_width: 1280,
    dataset_height: 720,
    width: 1280,
    height: 720,
	frames_buffering: 1,
    renderMode: "volsurfs",
    debugShader: "grazing_angles", 
    near: 0.1,
    far: 100,
    nrMeshes: 0,
    shDeg: shDeg,
    nrFrames: 0,
    frameTimes: [],
    maxFrameTimes: 100,
    bg_color: "black",
    antialias: false,
    render_single_mesh: false,
    mesh_idx: 0, // index of the mesh to render, if -1 render all meshes
    ignore_alpha: false,  // ignore alpha channel of the textures
    use_alpha_decay: true, // use alpha decay for rendering
    visualize_sh_coeffs: false, // visualize SH coefficients
    // texture_interp: "linear", // texture interpolation mode
}

let orbitCameraCfg = {
    isRotating: true,
    radius: 1.0,
    theta: 0,
    phi: 0,
    speed: 0.01
}

let shadersMaterials = {};

function initGUI() {
    
    const gui = new dat.GUI();
    
    const renderingFolder = gui.addFolder('rendering')

    // add a toggle switch between orbit camera rotating on/off
    const orbitCameraOptions = {
        'rotate': orbitCameraCfg.isRotating,
    };

    renderingFolder.add(orbitCameraOptions, "rotate").onChange((value) => {
        orbitCameraCfg.isRotating = value;
        console.log('orbit camera rotating:', value);
    });

    // // add a toggle switch between anti-aliasing on/off
    // const aaOptions = {
    //     'antialias': rendererCfg.antialias,
    // };

    // renderingFolder.add(aaOptions, "antialias").onChange((value) => {
    //     rendererCfg.antialias = value;
    //     console.log('antialias:', value);
    //     renderer = createRenderer();
    // });

    // add a toggle switch between dataset and benchmark resolutions
    const resolutionsOptions = {
        'resolution': 'benchmark',
    };

    renderingFolder.add(resolutionsOptions, "resolution", ["benchmark", "dataset"]).onChange((value) => {
        switch (value) {
            case 'benchmark':
                setCanvasSize(rendererCfg.benchmark_width, rendererCfg.benchmark_height);
                break;
            case 'dataset':
                setCanvasSize(rendererCfg.dataset_width, rendererCfg.dataset_height);
                break;
            default:
                console.error('unknown resolution');
                break;
        }
    });

    // Dropdown menu for shader selection
    const shaderOptions = {
        'volsurfs': 'volsurfs',
        'wireframe': 'wireframe',
        'view_dirs': 'view_dirs',
        'grazing_angles': 'grazing_angles',
        'uvs': 'uvs',
        'normals': 'normals'
    };

    const params = {
        shader: rendererCfg.renderMode
    };

    renderingFolder.add(params, 'shader', shaderOptions).onChange((selectedShader) => {
        console.log('Shader selected:', selectedShader);
    
        switch (selectedShader) {
            case 'volsurfs':
                rendererCfg.renderMode = 'volsurfs';
                break;
            case 'wireframe':
                rendererCfg.renderMode = 'wireframe';
                break;
            default:
                if (selectedShader in shadersMaterials) {
                    rendererCfg.renderMode = 'debug';
                    rendererCfg.debugShader = selectedShader;
                    if (sceneDebug.children.length > 0) {
                        sceneDebug.children[0].material = shadersMaterials[selectedShader];
                    }
                }
                break;
        }
        console.log('renderer configuration updated:', rendererCfg);
    });

    // add a toggle switch between single of multi surfaces rendering
    const renderOptions = {
        'single_mesh': rendererCfg.render_single_mesh,
    };

    renderingFolder.add(renderOptions, "single_mesh").onChange((value) => {
        rendererCfg.render_single_mesh = value;
        console.log('render single mesh:', value);
    });

    // add a slider for selecting the mesh index to render
    renderingFolder.add(rendererCfg, 'mesh_idx', 0, nrMeshes - 1).step(1).onChange((value) => {
        console.log('mesh index:', value);
    });

    // add a toggle to ignore the alpha channel of the textures
    const ignoreAlphaOptions = {
        'ignore_alpha': rendererCfg.ignore_alpha,
    };

    renderingFolder.add(ignoreAlphaOptions, "ignore_alpha").onChange((value) => {
        rendererCfg.ignore_alpha = value;
        console.log('ignore alpha:', value);
        // iterate over all scenesMeshes and set ignore_alpha to value
        scenesMeshes.forEach((sceneMesh) => {
            sceneMesh.children[0].material.uniforms.ignore_alpha.value = value;
        });
    });

    // add a toggle to use alpha decay for rendering
    const useAlphaDecayOptions = {
        'use_alpha_decay': rendererCfg.use_alpha_decay,
    };

    renderingFolder.add(useAlphaDecayOptions, "use_alpha_decay").onChange((value) => {
        rendererCfg.use_alpha_decay = value;
        console.log('use alpha decay:', value);
        // iterate over all scenesMeshes and set use_alpha_decay to value
        scenesMeshes.forEach((sceneMesh) => {
            sceneMesh.children[0].material.uniforms.use_alpha_decay.value = value;
        });
    });

    // // Dropdown menu for texture interpolation selection
    // const textureInterpOptions = {
    //     'linear': 'linear',
    //     'nearest': 'nearest',
    // };

    // const textureInterpParams = {
    //     texture_interp: rendererCfg.texture_interp
    // };

    // renderingFolder.add(textureInterpParams, 'texture_interp', textureInterpOptions).onChange((selectedInterp) => {
    //     console.log('Texture interpolation selected:', selectedInterp);
    //     rendererCfg.texture_interp = selectedInterp;
    //     // iterate over all scenesMeshes and set texture interpolation to value
    //     let filter = THREE.LinearFilter;
    //     if (selectedInterp == 'nearest') {
    //         filter = THREE.NearestFilter;
    //     }
    //     console.log('filter:', filter);
    //     scenesMeshes.forEach((sceneMesh) => {
    //         // sh_0_coeffs_texture_3D
    //         sceneMesh.children[0].material.uniforms.sh_0_coeffs_texture_3D.value.minFilter = filter;
    //         sceneMesh.children[0].material.uniforms.sh_0_coeffs_texture_3D.value.magFilter = filter;
    //         // sh_1_coeffs_texture_3D
    //         sceneMesh.children[0].material.uniforms.sh_1_coeffs_texture_3D.value.minFilter = filter;
    //         sceneMesh.children[0].material.uniforms.sh_1_coeffs_texture_3D.value.magFilter = filter;
    //         // sh_2_coeffs_texture_3D
    //         sceneMesh.children[0].material.uniforms.sh_2_coeffs_texture_3D.value.minFilter = filter;
    //         sceneMesh.children[0].material.uniforms.sh_2_coeffs_texture_3D.value.magFilter = filter;
    //         // sh_3_coeffs_texture_3D
    //         sceneMesh.children[0].material.uniforms.sh_3_coeffs_texture_3D.value.minFilter = filter;
    //         sceneMesh.children[0].material.uniforms.sh_3_coeffs_texture_3D.value.magFilter = filter;
    //     });
    // });

    renderingFolder.open();

    const cameraFolder = gui.addFolder('cameras');

    // add datasetCamera selection dropdown

    let cameraOptions = {
        'orbit' : 'orbit',
    };

    Object.keys(datasetCameras).forEach((cameraSet) => {
        Object.keys(datasetCameras[cameraSet]).forEach((cameraIdx) => {
            cameraOptions[cameraSet + ' ' + cameraIdx] = cameraSet + ' ' + cameraIdx;
        });
    });

    const cameraParams = {
        'camera': 'orbit'
    };

    cameraFolder.add(cameraParams, 'camera', cameraOptions).onChange((value) => {
        let newActiveCamera;
        if (value == 'orbit') {
            newActiveCamera = orbitCamera;
        } else {
            let [cameraSet, cameraIdx] = value.split(' ');
            newActiveCamera = datasetCameras[cameraSet][cameraIdx];
        }
        activateCamera(newActiveCamera);
    });

    cameraFolder.open();

    guiElement.appendChild(gui.domElement);
}

function activateCamera(camera) {
    activeCamera = camera;
    console.log('active camera:', activeCamera);
}

function get_bg_color(color_str) {
    // return RGB color based on string
    console.log('bg_color:', color_str);
    switch (color_str) {
        case "black":
            return new THREE.Vector3(0, 0, 0);
        case "white":
            return new THREE.Vector3(1, 1, 1);
        case "gray":
            return new THREE.Vector3(0.5, 0.5, 0.5);
        default:
            return new THREE.Vector3(0, 0, 0);
    }
}  

function initPost() {
    return new Promise((resolve, reject) => {
        // Setup post processing stage

        // init post camera
        postCamera = new THREE.OrthographicCamera(- 1, 1, 1, - 1, 0, 1);
        
        // load shaders
        const vertexShaderPromise = loadShader('shaders/vertexShader.glsl');
        const fragmentShaderPromise = loadShader('shaders/finalFrameFragmentShader.glsl');
        
        // wait for shaders to load first
        const shadersPromise = Promise.all([vertexShaderPromise, fragmentShaderPromise]);

        shadersPromise.then(([vertexShader, fragmentShader]) => {
            console.log('post shaders loaded');

            let bg_color = get_bg_color(rendererCfg.bg_color);

            postMaterial = new THREE.ShaderMaterial({
                vertexShader: vertexShader,
                fragmentShader: fragmentShader,
                uniforms: {
                    bgColor: { value: bg_color },
                    nrMeshes: { value: rendererCfg.nrMeshes },
                    screenSize: { value: new THREE.Vector2(rendererCfg.width, rendererCfg.height) },
                    meshesRGBATextures: { value: [] },
                }
            });

            const plane = new THREE.PlaneGeometry(2, 2);
            const quad = new THREE.Mesh(plane, postMaterial);
            scenePost = new THREE.Scene();
            scenePost.add(quad);
            resolve();

        }).catch((error) => {
            console.error('Error loading shaders:', error);
            reject('Error loading shaders');
        });
    }).catch((error) => {
        console.error('Error initializing post:', error);
        reject('Error initializing post');
    });
}

function parseSceneCfgJson(jsonPath) {
    console.log('loading', jsonPath);
    return new Promise((resolve, reject) => {
        $.getJSON(jsonPath, function(json) {
            try {
                console.log(json);
                // load meshes and textures infos
                const sceneCfg = {
                    nrMeshes: json['meshes'].length,
                    meshes: json['meshes'].map(mesh => ({
                        meshPath: mesh["mesh_path"],
                        texturesCfg: mesh['textures'].map(texture => ({
                            texturePath: texture['texture_path'],
                            textureScale: texture['texture_scale'],
                            textureResolution: texture['texture_resolution']
                        })),
                        ignore_alpha: mesh['ignore_alpha'],
                    })),
                };
                // (optional) load cameras info
                if (json['cameras']) {
                    sceneCfg.cameras = json['cameras'];
                } else {
                    sceneCfg.cameras = {
                        train: {},
                        test: {}
                    };
                }
                // (optional) load background color
                if (json['bg_color']) {
                    rendererCfg.bg_color = json['bg_color'];
                } else {
                    rendererCfg.bg_color = "black";
                }
                // (optional) load resolution
                if (json['resolution']) {
                    console.log('resolution:', json['resolution']);
                    let width = json['resolution'][0][0];
                    let height = json['resolution'][0][1];
                    rendererCfg.dataset_height = height;
                    rendererCfg.dataset_width = width;
                    sceneCfg.resolution = [width, height];
                } else {
                    rendererCfg.dataset_height = 800;
                    rendererCfg.dataset_width = 800;
                    sceneCfg.resolution = [800, 800];
                }
                resolve(sceneCfg);
            } catch (error) {
                reject('error parsing scene json:', error);
            }
        }).fail((jqxhr, textStatus, error) => {
            const err = textStatus + ", " + error;
            console.error("request failed: " + err);
            reject('error loading JSON: ' + err);
        });
    });
}

async function initSolidScene() {

    solidScene = new THREE.Scene();
    solidScene.background = new THREE.Color(0xffffff);
    solidScene.add(new THREE.AxesHelper(1));

    // box to test occlusion
    const geometry = new THREE.IcosahedronGeometry(0.1, 6);
    const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial());
    mesh.position.set(0.5, 0.3, 0.2);
    solidScene.add(mesh);

    // add light
    const light = new THREE.HemisphereLight(0xffffff, 0x444444, 3);
    light.position.set(-2, 2, 2);
    solidScene.add(light);

    // // create render targets for G-buffer (depth, color)
    // bgRenderTarget = new THREE.WebGLRenderTarget(rendererCfg.width, rendererCfg.height);
    // bgRenderTarget.stencilBuffer = false;
    // bgRenderTarget.texture.format = THREE.RGBAFormat;
    // bgRenderTarget.texture.type = THREE.FloatType;
    // // bgRenderTarget.texture.minFilter = THREE.NearestFilter;
    // // bgRenderTarget.texture.magFilter = THREE.NearestFilter;
    // bgRenderTarget.texture.colorSpace = THREE.NoColorSpace;
    // bgRenderTarget.depthTexture = new THREE.DepthTexture();
    // bgRenderTarget.depthTexture.format = THREE.DepthFormat;
    // bgRenderTarget.depthTexture.type = THREE.UnsignedIntType;
}

async function initWireframeScene() {
    sceneWireframe = new THREE.Scene();
    sceneWireframe.background = new THREE.Color(0xDEDEDE);

    var arrowPos = new THREE.Vector3(0, 0, 0);
    sceneWireframe.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), arrowPos, 1, 0xff0000, 0.1, 0.05));
    sceneWireframe.add(new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), arrowPos, 1, 0x00ff00, 0.1, 0.05));
    sceneWireframe.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), arrowPos, 1, 0x0000ff, 0.1, 0.05));

    const sphereR = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 8), new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.05, wireframe: true }));
    sceneWireframe.add(sphereR);

    const size = 2;
    const divisions = 10;
    const gridHelper = new THREE.GridHelper(size, divisions);
    sceneWireframe.add(gridHelper);
}

async function initDebugScene() {
    sceneDebug = new THREE.Scene();
    sceneDebug.background = new THREE.Color(0x1C2B33);

    const vertexShader = await loadShader('shaders/vertexShader.glsl');
    const rgbTextureFragmentShader = await loadShader('shaders/rgbTextureFragmentShader.glsl');
    const uvsFragmentShader = await loadShader('shaders/UVsFragmentShader.glsl');

    shadersMaterials['uvs'] = new THREE.ShaderMaterial({
        uniforms: {
            rgbaTexture: { value: null },
            inverse_alpha: { value: false },
        },
        vertexShader: vertexShader,
        // fragmentShader: rgbTextureFragmentShader,
        fragmentShader: uvsFragmentShader,
    });

    // uvs shader texture
    const textureLoader = new THREE.TextureLoader();
    const uvsTexturePath = 'assets/debug/textures/rgb.png';
    // const uvsTexturePath = 'assets/debug/textures/blue.png';
    textureLoader.load(uvsTexturePath,
            function (texture) {
                console.log('texture ' + uvsTexturePath + ' loaded');
                shadersMaterials['uvs'].uniforms.rgbaTexture.value = texture;
            }
    );

    const normalsFragmentShader = await loadShader('shaders/normalsFragmentShader.glsl');
    shadersMaterials['normals'] = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: normalsFragmentShader,
    });

    const viewDirsFragmentShader = await loadShader('shaders/viewDirsFragmentShader.glsl');
    shadersMaterials['view_dirs'] = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: viewDirsFragmentShader,
    });

    const grazingAnglesFragmentShader = await loadShader('shaders/grazingAnglesFragmentShader.glsl');
    shadersMaterials['grazing_angles'] = new THREE.ShaderMaterial({
        vertexShader: vertexShader,
        fragmentShader: grazingAnglesFragmentShader,
    });
}

function renderSingleMesh(meshIdx) {
    // render scene into target
    try {
        renderer.setRenderTarget(null); // unbind the render target
        renderer.render(scenesMeshes[meshIdx], activeCamera);
    } catch (error) {
        console.error('Error rendering mesh scene:', error);
    }
}
        

function renderVolsurfs() {

    // if frames_buffering > 1, render the scene multiple times (for benchmarking)
    for (let i = 0; i < rendererCfg.frames_buffering; i++) {

        if (rendererCfg.render_single_mesh) {
            renderSingleMesh(rendererCfg.mesh_idx);;
        }
        else {

            // // render scene into target
            // try {
            //     renderer.setRenderTarget(bgRenderTarget);
            //     renderer.render(solidScene, activeCamera);
            //     renderer.setRenderTarget(null); // unbind the render target
            // } catch (error) {
            //     console.error('Error rendering solid scene:', error);
            // }
            
            // iterate over meshesRenderTargets and scenesMeshes
            for (let i = 0; i < rendererCfg.nrMeshes; i++) {
                // render each mesh independently
                try { 
                    renderer.setRenderTarget(meshesRenderTargets[i]);
                    renderer.render(scenesMeshes[i], activeCamera);
                    renderer.setRenderTarget(null); // unbind the render target
                } catch (error) {
                    console.error('Error rendering mesh scene:', error);
                }
            };
        
            // render post FX
            try {
                // update post material with meshes textures
                for (let i = 0; i < rendererCfg.nrMeshes; i++) {
                    postMaterial.uniforms.meshesRGBATextures.value[i] = meshesRenderTargets[i].texture;
                }
                // render post scene
                renderer.render(scenePost, postCamera);
                renderer.setRenderTarget(null); // unbind the render target
            } catch (error) {
                console.error('Error rendering post scene:', error);
            }
        }
    }
}

function renderWireframe() {
    renderer.render(sceneWireframe, activeCamera);
    renderer.setRenderTarget(null); // unbind the render target
}

function renderDebug() {
    renderer.render(sceneDebug, activeCamera);
    renderer.setRenderTarget(null); // unbind the render target
}

function render() {

    switch (rendererCfg.renderMode) {
        case "volsurfs":
            renderVolsurfs();
            break;
        case "wireframe":
            renderWireframe();
            break;
        case "debug":
            renderDebug();
            break;
        default:
            console.error('unknown render mode');
            break;
    }
}

// Set up the rendering loop
function draw(now) {

    // console.log('now:', now);

    // Update the controls
    controls.update();

    // Render the scene
    if (canRender) {

        stats.begin();

        // if orbit camera is rotating
        if (orbitCameraCfg.isRotating) {
            orbitCameraCfg.theta += orbitCameraCfg.speed;
            // orbitCameraCfg.phi += orbitCameraCfg.speed;
            // make sure phi is between 0 and pi
            // orbitCameraCfg.theta = orbitCameraCfg.theta % (2 * Math.PI);
            // orbitCameraCfg.phi = orbitCameraCfg.phi % (Math.PI/4);
            orbitCamera.position.x = orbitCameraCfg.radius * Math.sin(orbitCameraCfg.theta) * Math.cos(orbitCameraCfg.phi);
            orbitCamera.position.y = orbitCameraCfg.radius * Math.sin(orbitCameraCfg.theta) * Math.sin(orbitCameraCfg.phi);
            orbitCamera.position.z = orbitCameraCfg.radius * Math.cos(orbitCameraCfg.theta);
            orbitCamera.lookAt(0, 0, 0);
        }
    
        render();
        // canRender = false;

        stats.end();

        // elapsed time
        const renderTime = (now - prec) / rendererCfg.frames_buffering;
        prec = now;

        // Add the render time to the frame times array
        rendererCfg.frameTimes.push(renderTime);

        // Keep the array size within the limit
        if (rendererCfg.frameTimes.length > rendererCfg.maxFrameTimes) {
            rendererCfg.frameTimes.shift();
        }
    }

    // Update frametime
    if (rendererCfg.nrFrames % 100 == 0) {
        // Calculate the average render time
        const averageRenderTime = rendererCfg.frameTimes.reduce((sum, time) => sum + time, 0) / rendererCfg.frameTimes.length;
        // Calculate the average FPS
        const averageFPS = 1000 / averageRenderTime;
        document.getElementById('frametime').innerHTML = averageRenderTime.toFixed(2);
        document.getElementById('fps').innerHTML = (averageFPS).toFixed(2);
        
        // Call the function to log memory usage
        if (performance.memory) {
            const memory = performance.memory;
            // console.log(`JS Heap Size Limit: ${memory.jsHeapSizeLimit / (1024 * 1024)} MB`);
            // console.log(`Total JS Heap Size: ${memory.totalJSHeapSize / (1024 * 1024)} MB`);
            // console.log(`Used JS Heap Size: ${memory.usedJSHeapSize / (1024 * 1024)} MB`);
            document.getElementById('heap_limit').innerHTML = (memory.jsHeapSizeLimit / (1024 * 1024)).toFixed(2);
            document.getElementById('total_heap').innerHTML = (memory.totalJSHeapSize / (1024 * 1024)).toFixed(2);
            document.getElementById('used_heap').innerHTML = (memory.usedJSHeapSize / (1024 * 1024)).toFixed(2);
        }
    }
}

function setCanvasSize(width, height) {
    renderer.setSize(width, height, true);
    orbitCamera.aspect = width / height;
    orbitCamera.updateProjectionMatrix();
    rendererCfg.width = width;
    rendererCfg.height = height;
    canvasContainer.style.width = width + 'px';
    canvasContainer.style.height = height + 'px';
    console.log('width: ' + rendererCfg.width + ', height: ' + rendererCfg.height);
    // update targets
    // if (bgRenderTarget)
    //     bgRenderTarget.setSize(width, height);
    meshesRenderTargets.forEach((target) => {
        target.setSize(width, height);
    });
    // if (bgRenderTarget) {
    //     bgRenderTarget.setSize(width, height);
    // }
    if (postMaterial) {
        postMaterial.uniforms.screenSize.value.set(width, height);
    }
}

function defaultCanvasSize() {
    setCanvasSize(rendererCfg.benchmark_width, rendererCfg.benchmark_height);
}

async function captureScreenshotsSequentially(split, cameraDict) {
    for (const cameraIdx of Object.keys(cameraDict[split])) {
        await new Promise((resolve) => {
            setTimeout(() => {
                activeCamera = cameraDict[split][cameraIdx];
                setTimeout(() => {
                    // format cameraIndex such that it is always 3 digits long
                    let cameraIdxStr = cameraIdx.padStart(3, '0');
                    captureScreenshot(renderer, split + '_' + cameraIdxStr);
                    resolve();
                }, 500);
            }, 500);
        });
    }
}

function createRenderer() {
    // Renderer
    let renderer_ = new THREE.WebGLRenderer({
        antialias: rendererCfg.antialias,
        depth: true,
        preserveDrawingBuffer: true,
        // colorspace
        outputColorSpace: THREE.NoColorSpace,
    });
    renderer_.autoClear = true;
    renderer_.setClearColor(new THREE.Color( 0x000000 ), 0.0);
    // renderer_.gammaOutput = true;
    // renderer_.gammaFactor = 2.2;
    // renderer_.setPixelRatio(window.devicePixelRatio);
    canvasElement.appendChild(renderer_.domElement);
    // const pixelRatio = window.devicePixelRatio;
    // rendererCfg.width = canvasElement.clientWidth * pixelRatio | 0;
    // rendererCfg.height = canvasElement.clientHeight * pixelRatio | 0;
    // renderer.setSize(rendererCfg.width, rendererCfg.height, false);
    return renderer_;
}

async function init() {

    canvasContainer = document.getElementById('canvas_container');
    canvasElement = document.getElementById('canvas');
    statsElement = document.getElementById('stats');
    guiElement = document.getElementById('gui');
    loadingScreen = document.getElementById('loading_screen');

    // Renderer
    renderer = createRenderer();

    // Orbit Camera
    orbitCamera = new THREE.PerspectiveCamera(
        45,
        rendererCfg.width / rendererCfg.height,
        rendererCfg.near,
        rendererCfg.far
    );
    orbitCamera.position.set(-0.8, 0.2, 0.8);
    orbitCamera.lookAt(0, 0, 0);
    activeCamera = orbitCamera;

    defaultCanvasSize();

    // Controls
    controls = new OrbitControls(orbitCamera, canvasElement);
    controls.enableDamping = true;
    // controls.update = function(){
    //     canRender = true;
    //     console.log('controls updated')
    // }

    // Stats
    stats = new Stats();
    stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom
    stats.dom.style.position = "absolute";
    statsElement.appendChild(stats.dom);

    // Buttons
    const screenshotButton = document.querySelector('#screenshot');
    screenshotButton.addEventListener('click', () => {
        captureScreenshot(renderer, "screenshot");
    });
    const screenshotTestButton = document.querySelector('#screenshot_test');
    screenshotTestButton.addEventListener('click', () => {
        // disable all buttons
        const buttons = document.querySelectorAll('button');
        buttons.forEach((button) => {
            button.disabled = true;
        });
        let split = "test";
        // set dataset resolution
        setCanvasSize(rendererCfg.dataset_width, rendererCfg.dataset_height);
        captureScreenshotsSequentially(split, datasetCameras).then(() => {
            // set back to orbit camera
            activateCamera(orbitCamera);
            // enable all buttons
            buttons.forEach((button) => {
                button.disabled = false;
            });
            // restore viewer resolution
            setCanvasSize(rendererCfg.benchmark_width, rendererCfg.benchmark_height);
        });
    });
    const screenshotTrainButton = document.querySelector('#screenshot_train');
    screenshotTrainButton.addEventListener('click', () => {
        // disable all buttons
        const buttons = document.querySelectorAll('button');
        buttons.forEach((button) => {
            button.disabled = true;
        });
        // set dataset resolution
        setCanvasSize(rendererCfg.dataset_width, rendererCfg.dataset_height);
        let split = "train";
        captureScreenshotsSequentially(split, datasetCameras).then(() => {
            // set back to orbit camera
            activateCamera(orbitCamera);
            // enable all buttons
            buttons.forEach((button) => {
                button.disabled = false;
            });
            // restore viewer resolution
            setCanvasSize(rendererCfg.benchmark_width, rendererCfg.benchmark_height);
        });
    });

    // initSolidScene();
    initWireframeScene();
    initDebugScene();
}

function loadScene(sceneDirPath) {
    
    return new Promise((resolve, reject) => {

        // Load scene configuration
        const sceneJsonPath = sceneDirPath + 'scene.json';

        // check if scene.json exists
        $.ajax({
            url: sceneJsonPath,
            type: 'HEAD',
            error: function() {
                // if scene.json not found, reject promise
                console.error('scene.json not found');
                reject('scene.json not found');
            },
            success: function() {
                // if scene.json found, parse it
                console.log('scene.json found');
                let parsePromise = parseSceneCfgJson(sceneJsonPath);

                parsePromise.then(sceneCfg => {

                    console.log('loaded scene config', sceneCfg);
        
                    // Load textured meshes and cameras in parallel and wait for both to complete
                    let loadTexturedMeshesPromise = loadTexturedMeshes(
                        sceneCfg,
                        scenesMeshes,
                        sceneWireframe,
                        sceneDebug,
                        shadersMaterials,
                        rendererCfg,
                        sceneDirPath
                    );
                    loadTexturedMeshesPromise.then(() => {
                        console.log('all textured meshes loaded');
                    });
                    
                    let loadCamerasPromise = loadCameras(sceneCfg);
                    // console.log(loadCamerasPromise);

                    loadCamerasPromise.then(cameras => {
                        console.log('all cameras loaded');
                        datasetCameras = cameras;
                        // iterate over cameras and add them to wireframe scene
                        Object.keys(datasetCameras).forEach((cameraSet) => {
                            if (datasetCameras[cameraSet]) {
                                Object.keys(datasetCameras[cameraSet]).forEach((cameraIdx) => {
                                    if (datasetCameras[cameraSet][cameraIdx]) {
                                        const camera = datasetCameras[cameraSet][cameraIdx];
                                        // add camera helper to wireframe scene
                                        const cameraHelper = new THREE.CameraHelper(camera);
                                        sceneWireframe.add(cameraHelper);
                                    }
                                });
                            }
                        });
                    });
                    
                    // when both promises are resolved, resolve the main promise
                    Promise.all([loadTexturedMeshesPromise, loadCamerasPromise]).then(() => {
                        console.log('all textured meshes and cameras loaded');
                        console.log('loaded cameras', datasetCameras);
                        resolve();
                    }).catch((error) => {
                        console.error('Error loading scene:', error);
                        reject('Error loading scene');
                    });
                
                }).catch((error) => {
                    console.error('Error parsing scene:', error);
                    reject('Error parsing scene');
                });
            }
        });
    });    
}

// MAIN ------------------------------------------------------------------------

init().then(() => {
    console.log('init done');
    let scenePromise = loadScene(sceneDirPath)
    
    scenePromise.then(() => {
        console.log('scene loaded');

        // init mesheses render targets
        for (let i = 0; i < rendererCfg.nrMeshes; i++) {
            const meshRenderTarget = new THREE.WebGLRenderTarget(rendererCfg.width, rendererCfg.height);
            // meshRenderTarget.stencilBuffer = false;
            meshRenderTarget.texture.format = THREE.RGBAFormat;
            meshRenderTarget.texture.type = THREE.HalfFloatType
            // meshRenderTarget.texture.minFilter = THREE.NearestFilter;
            // meshRenderTarget.texture.magFilter = THREE.NearestFilter;
            meshRenderTarget.texture.colorSpace = THREE.NoColorSpace;
            // meshRenderTarget.depthTexture = new THREE.DepthTexture();
            // meshRenderTarget.depthTexture.format = THREE.DepthFormat;
            // meshRenderTarget.depthTexture.type = THREE.UnsignedIntType;
            meshesRenderTargets.push(meshRenderTarget);
        }
        console.log("meshesRenderTargets:", meshesRenderTargets);

        initGUI();

        initPost().then(() => {
            loadingScreen.style.display = "none";
            document.getElementById("progress_container").style.display = "none";
            document.getElementById("buttons_container").style.display = "block";
            renderer.setAnimationLoop((now) => draw(now));
        });
    });

}).catch((error) => {
    console.error('Error initializing:', error);
});