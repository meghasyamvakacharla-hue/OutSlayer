const colorMap = {
    0: [0, 0, 0],         // Background -> Black
    1: [34, 139, 34],     // Vegetation/Shadows -> Green
    2: [205, 133, 63],    // Dirt/Terrain -> Brown/Orange
    3: [244, 164, 96],    // Sandy Dirt -> SandyBrown
    4: [128, 128, 128],   // Road/Path -> Gray
    5: [135, 206, 235]    // Sky -> Light Blue
};

let session = null;
const statusEl = document.getElementById('status');

async function initModel() {
    try {
        statusEl.innerHTML = '<div class="loader"></div> Loading AI Engine... (Downloading 13MB ONNX Model)';

        // Configure WebAssembly paths to use the CDN
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

        // Initialize ONNX Runtime session
        session = await ort.InferenceSession.create('./best_model_20_epochs.onnx', {
            executionProviders: ['wasm']
        });
        statusEl.innerHTML = '✅ Engine Ready. Waiting for image.';
        statusEl.style.color = '#4ade80';
    } catch (e) {
        statusEl.innerHTML = '❌ Failed to load AI Engine: ' + e.message;
        statusEl.style.color = '#ef4444';
        console.error(e);
    }
}

// Ensure execution starts
initModel();

// UI Elements
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');

const inputCanvas = document.getElementById('inputCanvas');
const displayCanvas = document.getElementById('displayCanvas');
const outputCanvas = document.getElementById('outputCanvas');

const inCtx = inputCanvas.getContext('2d');
const dispCtx = displayCanvas.getContext('2d');
const outCtx = outputCanvas.getContext('2d');

fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--accent-color)';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '';
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

function handleFile(file) {
    if (!file || !session) return;
    statusEl.innerHTML = '<div class="loader"></div> Processing...';

    const img = new Image();
    img.src = URL.createObjectURL(file);
    img.onload = () => {
        // Draw to input canvas (256x256) for inference preprocessing
        inCtx.drawImage(img, 0, 0, 256, 256);

        // Draw to display canvas (scaled) for visual UI
        dispCtx.drawImage(img, 0, 0, 256, 256);

        runInference();
    }
}

async function runInference() {
    const imgData = inCtx.getImageData(0, 0, 256, 256).data;

    // Convert to Float32Array [1, 3, 256, 256] planar format
    const float32Data = new Float32Array(3 * 256 * 256);

    for (let i = 0; i < 256 * 256; i++) {
        // imgData is RGBA interleaved
        float32Data[i] = imgData[i * 4] / 255.0;                   // R
        float32Data[256 * 256 + i] = imgData[i * 4 + 1] / 255.0;   // G
        float32Data[2 * 256 * 256 + i] = imgData[i * 4 + 2] / 255.0; // B
    }

    try {
        const tensor = new ort.Tensor('float32', float32Data, [1, 3, 256, 256]);
        const results = await session.run({ 'input': tensor });
        const output = results.output.data; // Float32Array of size 6 * 256 * 256

        // Output from model is [1, 6, 256, 256] logits.
        // We need argmax over dim 1.
        renderOutput(output);

        statusEl.innerHTML = '✅ Segmentation Complete.';
    } catch (e) {
        console.error(e);
        statusEl.innerHTML = '❌ Inference Failed.';
        statusEl.style.color = '#ef4444';
    }
}

function renderOutput(outputData) {
    const outImgData = outCtx.createImageData(256, 256);

    for (let i = 0; i < 256 * 256; i++) {
        // Find argmax for pixel i among 6 classes
        let maxVal = -Infinity;
        let maxClass = 0;

        for (let c = 0; c < 6; c++) {
            // outputData layout is [class_index][pixel_index]
            const val = outputData[c * 256 * 256 + i];
            if (val > maxVal) {
                maxVal = val;
                maxClass = c;
            }
        }

        const color = colorMap[maxClass];
        outImgData.data[i * 4] = color[0];     // R
        outImgData.data[i * 4 + 1] = color[1]; // G
        outImgData.data[i * 4 + 2] = color[2]; // B
        outImgData.data[i * 4 + 3] = maxClass === 0 ? 0 : 200; // Alpha (transparent for background)
    }

    outCtx.putImageData(outImgData, 0, 0);
}
