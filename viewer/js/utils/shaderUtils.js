// Utility function to load shaders
export async function loadShader(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`error loading shader from ${url}: ${response.statusText}`);
    }
    //else {
    //    console.log(`loaded shader from ${url}`);
    //}
    return await response.text();
}