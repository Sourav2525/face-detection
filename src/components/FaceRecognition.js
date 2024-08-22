import React, { useEffect, useRef } from 'react';
import * as faceapi from 'face-api.js';

const FaceRecognition = () => {
  const videoRef = useRef();
  const canvasRef = useRef();

  useEffect(() => {
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
      } catch (error) {
        console.error('Error accessing camera: ', error);
      }
    };

    const loadModels = async () => {
      const MODEL_URL = '/models';
      console.log('MODEL_URL: ', MODEL_URL);
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
      await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
      await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL); // Ensure SSD Mobilenet is loaded
      startVideo();
    };

    loadModels();

    const handlePlay = async () => {
      const canvas = faceapi.createCanvasFromMedia(videoRef.current);
      canvasRef.current = canvas;
      document.body.append(canvas);

      const displaySize = { width: videoRef.current.width, height: videoRef.current.height };
      faceapi.matchDimensions(canvas, displaySize);

      const labeledFaceDescriptors = await loadLabeledImages();
      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

      setInterval(async () => {
        const detections = await faceapi
          .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptors();

        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
        results.forEach((result, i) => {
          const box = resizedDetections[i].detection.box;
          const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
          drawBox.draw(canvas);
        });
      }, 100);
    };

    videoRef.current?.addEventListener('play', handlePlay);

    return () => {
      videoRef.current?.removeEventListener('play', handlePlay);
    };
  }, []);

  const loadLabeledImages = () => {
    const labels = ['person-1']; // Replace with actual names
    return Promise.all(
      labels.map(async (label) => {
        const descriptions = [];
        for (let i = 1; i < 2; i++) {
          const imageUrl = `/labeled_images/${label}/${i}.jpg`;
          console.log('imageUrl: ', imageUrl);
          try {
            console.log(`Fetching image from URL: ${imageUrl}`);
            const img = await faceapi.fetchImage(imageUrl);
            const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
            descriptions.push(detections.descriptor);
          } catch (error) {
            console.error(`Error fetching or processing image at ${imageUrl}: `, error);
          }
        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions);
      })
    );
  };

  return (
    <div>
      <video ref={videoRef} width="720" height="560" autoPlay muted />
      <canvas ref={canvasRef} />
    </div>
  );
};

export default FaceRecognition;
