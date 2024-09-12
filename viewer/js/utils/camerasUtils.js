import * as THREE from 'three';

function loadCamera(cameraProjMatrix, cameraMatrixWorld, width, height) {
    
    const projectionMatrix = new THREE.Matrix4().set(
        cameraProjMatrix[0][0], cameraProjMatrix[0][1], cameraProjMatrix[0][2], cameraProjMatrix[0][3],
        cameraProjMatrix[1][0], cameraProjMatrix[1][1], cameraProjMatrix[1][2], cameraProjMatrix[1][3],
        cameraProjMatrix[2][0], cameraProjMatrix[2][1], cameraProjMatrix[2][2], cameraProjMatrix[2][3],
        cameraProjMatrix[3][0], cameraProjMatrix[3][1], cameraProjMatrix[3][2], cameraProjMatrix[3][3]
    );

    const near = projectionMatrix.elements[14] / (projectionMatrix.elements[10] - 1);
    const far = projectionMatrix.elements[14] / (projectionMatrix.elements[10] + 1);
    const fov = 2 * Math.atan(1 / projectionMatrix.elements[5]) * (180 / Math.PI);

    // Create the camera
    let camera = new THREE.PerspectiveCamera(fov, width / height, near, far); 

    // camera.projectionMatrixAutoUpdate = false;
    camera.matrixAutoUpdate = false;

    const matrixWorld = new THREE.Matrix4().set(
        cameraMatrixWorld[0][0], cameraMatrixWorld[0][1], cameraMatrixWorld[0][2], cameraMatrixWorld[0][3],
        cameraMatrixWorld[1][0], cameraMatrixWorld[1][1], cameraMatrixWorld[1][2], cameraMatrixWorld[1][3],
        cameraMatrixWorld[2][0], cameraMatrixWorld[2][1], cameraMatrixWorld[2][2], cameraMatrixWorld[2][3],
        cameraMatrixWorld[3][0], cameraMatrixWorld[3][1], cameraMatrixWorld[3][2], cameraMatrixWorld[3][3]
    );

    // rotate camera pose in world space by 90 degrees clockwise on X axis
    const rotationMatrix = new THREE.Matrix4().makeRotationX(-Math.PI / 2);

    // Apply the rotation to the inverse of the matrixWorld
    matrixWorld.premultiply(rotationMatrix);

    // Set the cameraMatrix matrix
    camera.matrixWorld.copy(matrixWorld);

    return camera;
}

export async function loadCameras(sceneCfg) {
    
    let datasetCameras = {
        train: {},
        test: {}
    };

    const width = sceneCfg["resolution"][0];
    const height = sceneCfg["resolution"][1];
    const testCameras = sceneCfg.cameras["test"];
    const trainCameras = sceneCfg.cameras["train"];

    // Load test cameras
    if (!testCameras) {
        console.error('No test cameras found in the scene configuration!');
    } else {
        Object.keys(testCameras).map((cameraIdx) => {
            // console.log('Loading camera:', cameraIdx);
            let cameraProjMatrix = testCameras[cameraIdx]["projectionMatrix"];
            let cameraMatrixWorld = testCameras[cameraIdx]["matrixWorld"];
            let camera = loadCamera(cameraProjMatrix, cameraMatrixWorld, width, height);
            datasetCameras.test[cameraIdx] = camera;
        });
    }

    // Load train cameras
    if (!trainCameras) {
        console.error('No train cameras found in the scene configuration!');
    } else {
        Object.keys(trainCameras).map((cameraIdx) => {
            // console.log('Loading camera:', cameraIdx);
            let cameraProjMatrix = trainCameras[cameraIdx]["projectionMatrix"];
            let cameraMatrixWorld = trainCameras[cameraIdx]["matrixWorld"];
            let camera = loadCamera(cameraProjMatrix, cameraMatrixWorld, width, height);
            datasetCameras.train[cameraIdx] = camera;
        });
    }

    return datasetCameras;
}

// export function loadCameras(sceneCfg) {

//     let datasetCameras = {
//         train: {},
//         test: {}
//     };

//     const testCameras = sceneCfg.cameras["test"];
//     Object.keys(testCameras).forEach((cameraIdx) => {
//         const cameraProj = testCameras[cameraIdx];
//         const camera = loadCamera(cameraProj);
//         datasetCameras.test[cameraIdx] = camera;
//     });

//     return datasetCameras;
// }