import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as bodyPix from '@tensorflow-models/body-pix';
import './BodyPixSegmentation.css'; // Import the styles

const BodyPixSegmentation = () => {
    // State for TFJS model, webcam status, and UI messages
    const [model, setModel] = useState(null);
    const [webcamRunning, setWebcamRunning] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('Loading TensorFlow.js and BodyPix model...');
    const [errorMessage, setErrorMessage] = useState('');
    const [bestCamera, setBestCamera] = useState(null);

    // State for segmentation controls
    const [foregroundThreshold, setForegroundThreshold] = useState(0.5);
    const [maskOpacity, setMaskOpacity] = useState(0.7);
    const [maskBlur, setMaskBlur] = useState(9);

    // Refs for DOM elements and animation loop
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const tempCanvasRef = useRef(null);
    const animationFrameIdRef = useRef(null);

    // Effect to load the BodyPix model
    useEffect(() => {
        const loadModel = async () => {
            try {
                await tf.setBackend('webgl');
                await tf.ready();
                const net = await bodyPix.load({
                    architecture: 'ResNet50',
                    outputStride: 32,
                    quantBytes: 4
                });
                setModel(net);
                setLoadingMessage('');
            } catch (error) {
                console.error("Failed to load model:", error);
                setErrorMessage("Could not load the BodyPix model. Please refresh the page.");
            }
        };
        loadModel();
    }, []);

    // Effect to find the best camera (highest resolution)
    useEffect(() => {
        const findHighestResolutionCamera = async () => {
            if (!navigator.mediaDevices?.enumerateDevices) {
                setErrorMessage("This browser does not support listing media devices.");
                return;
            }
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const videoDevices = devices.filter(d => d.kind === 'videoinput');
                if (videoDevices.length > 0) {
                    setBestCamera(videoDevices[0]); // Just pick the first camera available
                } else {
                    setErrorMessage("No video input devices found.");
                }
            } catch (err) {
                console.error("Error enumerating devices:", err);
                setErrorMessage("Could not list video devices.");
            }
        };
        findHighestResolutionCamera();
    }, []);

    // Main prediction loop
    const predictWebcam = useCallback(async () => {
        if (!webcamRunning || !model || !videoRef.current || !canvasRef.current || videoRef.current.readyState < 3) {
            if (webcamRunning) {
                animationFrameIdRef.current = requestAnimationFrame(predictWebcam);
            }
            return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const tempCanvas = tempCanvasRef.current;
        const canvasCtx = canvas.getContext('2d');
        const tempCtx = tempCanvas.getContext('2d');

        const segmentation = await model.segmentPerson(video, {
            segmentationThreshold: foregroundThreshold
        });

        if (segmentation.data) {
            const foregroundColor = { r: 255, g: 255, b: 255, a: 255 };
            const backgroundColor = { r: 0, g: 0, b: 0, a: 0 };
            const personMask = bodyPix.toMask(segmentation, foregroundColor, backgroundColor);
            const maskBitmap = await createImageBitmap(personMask);

            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
            tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);

            if (maskBlur > 0) {
                tempCtx.filter = `blur(${maskBlur}px)`;
            }
            tempCtx.drawImage(maskBitmap, 0, 0);
            tempCtx.filter = 'none';

            canvasCtx.globalAlpha = maskOpacity;
            canvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvasCtx.globalCompositeOperation = 'destination-in';
            canvasCtx.drawImage(tempCanvas, 0, 0);

            canvasCtx.globalCompositeOperation = 'source-over';
            canvasCtx.globalAlpha = 1.0;
        } else {
            canvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }

        if (webcamRunning) {
            animationFrameIdRef.current = requestAnimationFrame(predictWebcam);
        }
    }, [webcamRunning, model, foregroundThreshold, maskOpacity, maskBlur]);

    // Start/stop prediction loop when webcamRunning changes
    useEffect(() => {
        if (webcamRunning) {
            animationFrameIdRef.current = requestAnimationFrame(predictWebcam);
        } else {
            if (animationFrameIdRef.current) {
                cancelAnimationFrame(animationFrameIdRef.current);
            }
        }
        return () => {
            if (animationFrameIdRef.current) {
                cancelAnimationFrame(animationFrameIdRef.current);
            }
        };
    }, [webcamRunning, predictWebcam]);


    const enableCam = async () => {
        setErrorMessage('');
        if (!model || !bestCamera) return;

        if (webcamRunning) {
            setWebcamRunning(false);
            if (videoRef.current?.srcObject) {
                videoRef.current.srcObject.getTracks().forEach(track => track.stop());
                videoRef.current.srcObject = null;
            }
        } else {
            try {
                const constraints = {
                    video: {
                        deviceId: bestCamera.deviceId,
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    }
                };
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                videoRef.current.srcObject = stream;
                videoRef.current.onloadedmetadata = () => {
                    canvasRef.current.width = videoRef.current.videoWidth;
                    canvasRef.current.height = videoRef.current.videoHeight;
                    tempCanvasRef.current.width = videoRef.current.videoWidth;
                    tempCanvasRef.current.height = videoRef.current.videoHeight;
                    setWebcamRunning(true);
                };
            } catch (error) {
                console.error("Error accessing webcam:", error);
                setErrorMessage(`Could not access webcam: ${error.message}. Please grant permission and try again.`);
            }
        }
    };

    const saveImage = () => {
        const canvas = canvasRef.current;
        if (!canvas || !webcamRunning) {
            setErrorMessage("Please enable the webcam to save an image.");
            return;
        }
        const link = document.createElement('a');
        link.download = `segmented-person-${new Date().getTime()}.png`;
        link.href = canvas.toDataURL('image/png');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="app-container">
            <div className="demo-container">
                <h1>Person Segmentation with BodyPix</h1>
                <p>This app uses TensorFlow.js to perform real-time person segmentation. Enable your webcam and adjust the controls to see the effect.</p>

                <div className="controls-container">
                    <div className="control-group">
                        <label htmlFor="threshold">Foreground Threshold</label>
                        <input type="range" id="threshold" min="0" max="1" step="0.05" value={foregroundThreshold} onChange={(e) => setForegroundThreshold(parseFloat(e.target.value))} disabled={!model} />
                        <span>{foregroundThreshold.toFixed(2)}</span>
                    </div>
                    <div className="control-group">
                        <label htmlFor="opacity">Mask Opacity</label>
                        <input type="range" id="opacity" min="0" max="1" step="0.05" value={maskOpacity} onChange={(e) => setMaskOpacity(parseFloat(e.target.value))} disabled={!model} />
                        <span>{maskOpacity.toFixed(2)}</span>
                    </div>
                    <div className="control-group">
                        <label htmlFor="blur">Mask Blur</label>
                        <input type="range" id="blur" min="0" max="20" step="1" value={maskBlur} onChange={(e) => setMaskBlur(parseInt(e.target.value, 10))} disabled={!model} />
                        <span>{maskBlur}</span>
                    </div>
                </div>

                <div className="webcam-container">
                    <video ref={videoRef} autoPlay playsInline></video>
                    <canvas ref={canvasRef}></canvas>
                    {/* Hidden canvas for processing */}
                    <canvas ref={tempCanvasRef} style={{ display: 'none' }}></canvas>

                    <div className="webcam-buttons">
                        <button onClick={enableCam} disabled={!model || !bestCamera}>
                            {webcamRunning ? 'DISABLE WEBCAM' : 'ENABLE WEBCAM'}
                        </button>
                        <button onClick={saveImage} disabled={!webcamRunning}>
                            SAVE IMAGE
                        </button>
                    </div>
                    {loadingMessage && <p className="message loading">{loadingMessage}</p>}
                    {errorMessage && <p className="message error">{errorMessage}</p>}
                </div>
            </div>
        </div>
    );
};

export default BodyPixSegmentation;