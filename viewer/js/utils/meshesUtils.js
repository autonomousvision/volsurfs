import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { loadShader } from './shaderUtils.js';

let nrMeshesToLoad = 0;
let meshesLoadingProgress = [];
let nrTexturesToLoad = 0;
let texturesLoadingProgress = [];
let dataProcessingProgress = 0;

/**
 * Returns a promise that fires within a specified amount of time. Can be used
 * in an asynchronous function for sleeping.
 * @param {number} milliseconds Amount of time to sleep
 * @return {!Promise}
 */
function sleep(milliseconds) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

/**
   * Safe fetching. Some servers restrict the number of requests and
   * respond with status code 429 ("Too Many Requests") when a threshold
   * is exceeded. When we encounter a 429 we retry after a short waiting period.
   * @param {!object} fetchFn Function that fetches the file.
   * @return {!Promise} Returns fetchFn's response.
   */
async function fetchAndRetryIfNecessary(fetchFn) {
    const response = await fetchFn();
    if (response.status === 429) {
        await sleep(500);
        return fetchAndRetryIfNecessary(fetchFn);
    }
    return response;
}

/**
 * Loads PNG image from rgbaURL and decodes it to an Uint8Array.
 * @param {string} rgbaUrl The URL of the PNG image.
 * @return {!Promise<!Uint8Array>}
 */
function loadPNG(rgbaUrl) {
    let fetchFn = () => fetch(rgbaUrl, {method: 'GET', mode: 'cors'});
    const rgbaPromise = fetchAndRetryIfNecessary(fetchFn)
                            .then(response => {
                                return response.arrayBuffer();
                            })
                            .then(buffer => {
                                let data = new Uint8Array(buffer);
                                let pngDecoder = new PNG(data);
                                let pixels = pngDecoder.decodePixels();
                                return pixels;
                            });
    rgbaPromise.catch(error => {
        console.error('Could not PNG image from: ' + rgbaUrl + ', error: ' + error);
        return;
    });
    return rgbaPromise;
}

// PNG loading code from Christian Reiser: thank you!

function loadTexture(texturePath, nrMesh, nrTexture, nrTexturesPerMesh) {
    let textureIdx = nrMesh * nrTexturesPerMesh + nrTexture;
    nrTexturesToLoad += 1;
    texturesLoadingProgress[textureIdx] = 0;

    let pngPromise = loadPNG(texturePath)
    
    pngPromise.then((pixels) => {
        texturesLoadingProgress[textureIdx] = 100;
        let sum = 0;
        texturesLoadingProgress.forEach(num => { sum += num });
        document.getElementById("progress_textures").style.width = ((sum / nrTexturesToLoad) + '%');
        console.log('mesh ' + nrMesh + ' loaded texture ' + nrTexture);
    });

    return pngPromise;
}

function loadTextures(nrMesh, texturesCfg, sceneDirPath) {
    
    // array to hold promises for each async operation
    let texturesPromises = [];

    texturesCfg.map((textureCfg, nrChunck) => {
        const texturePath = sceneDirPath + textureCfg.texturePath;
        let texturePromise = loadTexture(texturePath, nrMesh, nrChunck, texturesCfg.length)
        texturesPromises.push(texturePromise);
    });

    return texturesPromises;
}

function loadMeshGeometry(meshPath, nrMesh) {

    console.log('loading mesh ' + nrMesh + ' (' + meshPath + ')');

    nrMeshesToLoad += 1;
    meshesLoadingProgress[nrMesh] = 0;
    
    const objLoader = new OBJLoader();
    return new Promise((resolve, reject) => {
        objLoader.load(meshPath,
            function (obj) {
                console.log('loaded mesh geometry ' + nrMesh);
                // rotate by 90 degrees clockwise on X axis
                obj.children[0].geometry.rotateX(-Math.PI / 2);
                resolve(obj.children[0].geometry);
            },
            function (xhr) {
                meshesLoadingProgress[nrMesh] = ((xhr.loaded / xhr.total) * 100);
                let sum = 0;
                meshesLoadingProgress.forEach(num => {sum += num}) 
                document.getElementById("progress_meshes").style.width = ((sum / nrMeshesToLoad) + '%');
            },
            function (error) {
                reject(error); // Reject the promise if an error occurs
            }
        );
    });
}

function loadMeshesGeometries(sceneCfg, sceneDirPath) {

    const meshesGeometriesPromises = sceneCfg.meshes.map((meshCfg, nrMesh) => {
        const meshPath = sceneDirPath + meshCfg.meshPath;
        return loadMeshGeometry(meshPath, nrMesh).catch(error => {
            console.error('Error loading mesh:', error);
            throw error;  // Re-throw to be caught by Promise.all
        });
    });

    return Promise.all(meshesGeometriesPromises)
        .then(geometries => {
            console.log('all meshes geometries loaded');
            return geometries;
        })
        .catch(error => {
            console.error('error loading one or more meshes geometries:', error);
            throw error;
        });
}

function loadMeshes(sceneCfg, scenesMeshes, sceneWireframe, sceneDebug, shadersMaterials, rendererCfg, sceneDirPath) {

    return new Promise((resolve, reject) => {

        // load shaders
        const vertexShaderPromise = loadShader('./shaders/vertexShader.glsl');
        let fragmentShaderPromise;
        switch (rendererCfg.shDeg) {
            case 0:
                fragmentShaderPromise = loadShader('./shaders/sh0FragmentShader.glsl');
                console.log('sh0FragmentShader');
                break;
            case 1:
                fragmentShaderPromise = loadShader('./shaders/sh1FragmentShader.glsl');
                console.log('sh1FragmentShader');
                break;
            case 2: 
                fragmentShaderPromise = loadShader('./shaders/sh2FragmentShader.glsl');
                console.log('sh2FragmentShader');
                break;
            case 3:
                fragmentShaderPromise = loadShader('./shaders/sh3FragmentShader.glsl');
                console.log('sh3FragmentShader');
                break;
        }

        // load meshes geometries
        const meshesGeometriesPromise = loadMeshesGeometries(sceneCfg, sceneDirPath);

        // when all shaders are loaded
        vertexShaderPromise.then((vertexShader) => {
        
            fragmentShaderPromise.then((fragmentShader) => {
                
                meshesGeometriesPromise.then((geometries) => {
                    
                    // iterate over meshes geometries
                    geometries.forEach((geometry, nrMesh) => {
                        
                        // assumption: texture scales are all equal
                        let values_range = sceneCfg.meshes[nrMesh].texturesCfg[0].textureScale;

                        const material = new THREE.ShaderMaterial({
                            uniforms: {
                                sh_0_coeffs_texture_3D: { value: null },
                                sh_1_coeffs_texture_3D: { value: null },
                                sh_2_coeffs_texture_3D: { value: null },
                                sh_3_coeffs_texture_3D: { value: null },
                                inverse_alpha: { value: false },
                                ignore_alpha: { value: sceneCfg.meshes[nrMesh].ignore_alpha },
                                use_alpha_decay: { value: true },
                                visualize_sh_coeffs: { value: rendererCfg.visualize_sh_coeffs },
                                values_range: { value: new THREE.Vector2(values_range[0], values_range[1]) },
                            },
                            transparent: false,
                            vertexShader: vertexShader,
                            fragmentShader: fragmentShader
                        });
        
                        // create mesh
                        const mesh = new THREE.Mesh(geometry, material);
                        mesh.renderOrder = nrMesh;
                        // create mesh scene
                        scenesMeshes[nrMesh] = new THREE.Scene();
                        // add mesh to mesh scene
                        scenesMeshes[nrMesh].add(mesh);

                        // get random color for mesh
                        const color = new THREE.Color(Math.random() * 0xffffff);
                        // add mesh for wireframe scene
                        const meshWireframe = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ 
                            wireframe: true, 
                            color: color,
                            transparent: true, 
                            opacity: 0.1, 
                            depthWrite: false, 
                            depthTest: false
                        }));
                        meshWireframe.renderOrder = nrMesh;
                        
                        // add mesh to wireframe scene
                        sceneWireframe.add(meshWireframe);
                        
                        // add last (outer) mesh for debug scene
                        if (nrMesh == sceneCfg.nrMeshes - 1) {
                            const meshDebug = new THREE.Mesh(geometry, shadersMaterials[rendererCfg.debugShader]);
                            sceneDebug.add(meshDebug);
                        }
                        
                        // update counter of loaded meshes
                        rendererCfg.nrMeshes += 1;
                    
                    });
                    
                    console.log('all meshes loaded');

                    resolve();

                }).catch(error => {
                    console.error('Error loading meshes geometries:', error);
                    reject(error);
                });
            
            }).catch(error => {
                console.error('Error loading fragment shader:', error);
                reject(error);
            });

        }).catch(error => {
            console.error('Error loading vertex shader:', error);
            reject(error);
        });

    });
}

function loadTexture3D(nrMesh, texturesCfg, sceneDirPath) {
    
    return new Promise((resolve, reject) => {
        
        let meshTexturesPromise = loadTextures(nrMesh, texturesCfg, sceneDirPath);

        // when all mesh's textures are loaded
        Promise.all(meshTexturesPromise).then((textures) => {

            let sh_data = {};
            let textures_dims = {};

            // SH deg coeffs per degree
            // assumption: all textures of the same degree have the same resolution

            textures_dims['0'] = [
                texturesCfg[0].textureResolution[0], 
                texturesCfg[0].textureResolution[1], 
                1
            ];
            sh_data['0'] = new Uint8Array(textures_dims['0'][0] * textures_dims['0'][1] * 4 * textures_dims['0'][2]);
            console.log('deg 0 textures res', textures_dims['0']);

            if (textures.length >= 4) {
                textures_dims['1'] = [
                    texturesCfg[3].textureResolution[0],
                    texturesCfg[3].textureResolution[1],
                    3
                ];
                sh_data['1'] = new Uint8Array(textures_dims['1'][0] * textures_dims['1'][1] * 4 * textures_dims['1'][2]);
                console.log('deg 1 textures res', textures_dims['1']);
            
                if (textures.length >= 9) {
                    textures_dims['2'] = [
                        texturesCfg[8].textureResolution[0],
                        texturesCfg[8].textureResolution[1],
                        5
                    ];
                    sh_data['2'] = new Uint8Array(textures_dims['2'][0] * textures_dims['2'][1] * 4 * textures_dims['2'][2]);
                    console.log('deg 2 textures res', textures_dims['2']);
                }

                if (textures.length == 16) {
                    textures_dims['3'] = [
                        texturesCfg[15].textureResolution[0],
                        texturesCfg[15].textureResolution[1],
                        7
                    ];
                    sh_data['3'] = new Uint8Array(textures_dims['3'][0] * textures_dims['3'][1] * 4 * textures_dims['3'][2]);
                    console.log('deg 3 textures res', textures_dims['3']);
                }

                if (textures.length > 16)
                    throw 'too many textures, max 16 supported';
            }

            for (let nrChunck = 0; nrChunck < textures.length; nrChunck++) {
                
                if (nrChunck == 0) {
                    sh_data['0'].set(textures[nrChunck], 0);
                }
                
                if (nrChunck >= 1 && nrChunck < 4) {
                    let offset = (nrChunck - 1) * textures_dims['1'][0] * textures_dims['1'][1] * 4;
                    sh_data['1'].set(textures[nrChunck], offset);
                }

                if (nrChunck >= 4 && nrChunck < 9) {
                    let offset = (nrChunck - 4) * textures_dims['2'][0] * textures_dims['2'][1] * 4;
                    sh_data['2'].set(textures[nrChunck], offset);
                }

                if (nrChunck >= 9 && nrChunck < 16) {
                    let offset = (nrChunck - 9) * textures_dims['3'][0] * textures_dims['3'][1] * 4;
                    sh_data['3'].set(textures[nrChunck], offset);
                }

                dataProcessingProgress += 100;
                console.log('mesh ' + nrMesh + ' processed image data for texture ' + nrChunck);
                document.getElementById("progress_data").style.width = ((dataProcessingProgress / nrTexturesToLoad) + '%');

            }

            try {

                let sh_textures3D = {};
                
                // iterate over degrees, create a 3D texture for each degree
                for (const [key, data] of Object.entries(sh_data)) {    

                    const width = textures_dims[key][0];
                    const height = textures_dims[key][1];
                    const depth = textures_dims[key][2];
                    
                    const texture3D = new THREE.DataArrayTexture(
                        data, width, height, depth,
                        THREE.RGBAFormat,  // format
                        THREE.UnsignedByteType,  // type
                        // THREE.Texture.DEFAULT_MAPPING,  // mapping
                        // THREE.ClampToEdgeWrapping,  // wrapS
                        // THREE.ClampToEdgeWrapping,  // wrapT
                        // THREE.ClampToEdgeWrapping,  // wrapR
                        // THREE.LinearMipMapLinearFilter,  // mipMap
                    );
                    //
                    texture3D.minFilter = THREE.LinearFilter;
                    texture3D.magFilter = THREE.LinearFilter;
                    //
                    // texture3D.minFilter = THREE.NearestFilter;
                    // texture3D.magFilter = THREE.NearestFilter;
                    // 
                    texture3D.wrapS = THREE.ClampToEdgeWrapping;
                    texture3D.wrapT = THREE.ClampToEdgeWrapping;
                    // colorspace
                    texture3D.colorSpace = THREE.NoColorSpace;
                    texture3D.unpackAlignment = 4;
                    texture3D.needsUpdate = true;

                    sh_textures3D['texture_' + key] = texture3D;
                }

                resolve(sh_textures3D);

            } catch (error) {
                console.error('error creating 3D texture:', error);
                reject(error);
            }

        });
    });
}

            //         // // iterate over mesh's textures

            //         // for (let nrChunck = 0; nrChunck < textures.length; nrChunck++) {

            //         //     // const textureCfg = texturesCfg[nrChunck];
            //         //     // const v_min = textureCfg.textureScale[0];
            //         //     // const v_max = textureCfg.textureScale[1];

            //         //     // promise to extract data of texture
            //         //     // let imageDataPromise = extractImageData(textures[nrChunck]);
            //         //     // imageDataPromises.push(imageDataPromise);
                        
            //         //     let imageData = textures[nrChunck]

            //         //     // // when texture data is extracted
            //         //     // imageDataPromise.then((imageData) => {
                            
            //         //     console.log('mesh ' + nrMesh + ' extracted image data for texture ' + nrChunck);

            //         //     // // convert image data to float32 and scale it
            //         //     // // const imageDataUint16 = new Uint16Array(imageData.data.length);
            //         //     // const imageDataUint8 = new Uint8Array(imageData.data.length);
            //         //     // for (let i = 0; i < imageData.data.length / 4; i++) {
            //         //     //     const stride = i * 4;
            //         //     //     for (let j = 0; j < 4; j++) {
            //         //     //         // const value = imageData.data[stride + j] / 255.0;
            //         //     //         // const scaled_value = value * (v_max - v_min) + v_min;
            //         //     //         // const half_precision_value = toHalf(scaled_value);
            //         //     //         // imageDataUint16[stride + j] = half_precision_value;
            //         //     //         const value = imageData.data[stride + j];
            //         //     //         imageDataUint8[stride + j] = value;
            //         //     //     }
            //         //     // }

            //         //     console.log(imageData);

            //         //     // copy texture data to the data array

            //         //     if (nrChunck == 0) {
            //         //         console.log("sh_data['0']", sh_data['0']);
            //         //         sh_data['0'].set(imageDataUint8, 0);
            //         //     }

            //         //     dataProcessingProgress += 100;
            //         //     console.log('mesh ' + nrMesh + ' processed image data for texture ' + nrChunck);
            //         //     document.getElementById("progress_data").style.width = ((dataProcessingProgress / nrTexturesToLoad) + '%');

            //         //     }).catch(error => {
            //         //         console.error('error extracting texture data:', error);
            //         //         reject(error);
            //         //     });
            //         // }
                
            //         Promise.all(imageDataPromises).then(() => {
            //             console.log('all textures data extracted');
            //             resolve([sh_data, textures_dims]);
            //         });

            //     // } catch (error) {
            //     //     console.error('error extracting textures data:', error);
            //     //     reject(error);
            //     // }
            
            // // });

            // dataPromise.then(([sh_data, textures_dims]) => {
                
                // create Texture3D
                
                // // iterate over textures data and save them to file as png
                // for (const [key, data] of Object.entries(sh_data)) {
                //     console.log('key', key);
                //     console.log('data', data);
                //     const width = textures_dims[key][0];
                //     const height = textures_dims[key][1];
                
                //     const canvas = document.createElement('canvas');
                //     canvas.width = width;
                //     canvas.height = height;
                //     const context = canvas.getContext('2d');
                //     const imageData = context.createImageData(width, height);
                //     for (let i = 0; i < data.length; i++) {
                //         imageData.data[i] = data[i];
                //     }
                //     context.putImageData(imageData, 0, 0);
                //     const png = canvas.toDataURL('image/png');
                //     const a = document.createElement('a');
                //     a.href = png;
                //     // download the png
                //     a.download = 'texture_' + key + '.png';
                //     a.click();
                // }

export function loadTexturedMeshes(sceneCfg, scenesMeshes, sceneWireframe, sceneDebug, shadersMaterials, rendererCfg, sceneDirPath) {
    
    return new Promise((resolve, reject) => {
        
        // Load meshes
        let meshesPromise = loadMeshes(sceneCfg, scenesMeshes, sceneWireframe, sceneDebug, shadersMaterials, rendererCfg, sceneDirPath)

        // when all meshes are loaded
        let meshes3DTexturesPromises = [];
        meshesPromise.then(() => {
            
            // iterate over meshes (configs)
            sceneCfg.meshes.map((meshCfg, nrMesh) => {

                // load all mesh's textures
                const texturesCfg = meshCfg.texturesCfg;

                // promise loading of a 3D texture
                let meshTexture3DPromise = loadTexture3D(nrMesh, texturesCfg, sceneDirPath)
                meshes3DTexturesPromises.push(meshTexture3DPromise);

                meshTexture3DPromise.then((sh_textures3D) => {

                    // add 3D textures to materials
                    console.log('mesh ' + nrMesh + ' loaded 3D texture');

                    console.log('sh_textures3D.texture_0', sh_textures3D.texture_0);
                    scenesMeshes[nrMesh].children[0].material.uniforms.sh_0_coeffs_texture_3D.value = sh_textures3D.texture_0;

                    // if sh_textures3D.texture_1 is defined
                    if (sh_textures3D.texture_1) {
                        console.log('sh_textures3D.texture_1', sh_textures3D.texture_1);
                        scenesMeshes[nrMesh].children[0].material.uniforms.sh_1_coeffs_texture_3D.value = sh_textures3D.texture_1;
                    }

                    // if sh_textures3D.texture_2 is defined
                    if (sh_textures3D.texture_2) {
                        console.log('sh_textures3D.texture_2', sh_textures3D.texture_2);
                        scenesMeshes[nrMesh].children[0].material.uniforms.sh_2_coeffs_texture_3D.value = sh_textures3D.texture_2;
                    }

                    if (sh_textures3D.texture_3) {
                        console.log('sh_textures3D.texture_3', sh_textures3D.texture_3);
                        scenesMeshes[nrMesh].children[0].material.uniforms.sh_3_coeffs_texture_3D.value = sh_textures3D.texture_3;
                    }
                
                }).catch(error => {
                    console.error('Error loading 3D texture:', error);
                    reject(error);
                });
            
            });

            // Wait for all 3D textures to be loaded
            Promise.all(meshes3DTexturesPromises)
            .then(() => {
                resolve();
            })
            .catch(error => {
                console.error('Error loading one or more 3D textures:', error);
                reject(error);
            });

        }).catch(error => {
            console.error('Error loading meshes:', error);
            reject(error);
        });
    });   
}