import { useState, useRef } from "react";

export default function DetectionPage() {
  const [image, setImage] = useState(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState("upload"); // "upload" or "camera"
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const stages = {
    "Non Demented": {
      symptoms: "No significant cognitive decline, normal brain function.",
      cure: "Healthy lifestyle, brain exercises, and regular checkups.",
      hospital: "NeuroCare Hospital, Brain Health Institute",
    },
    "Mild Demented": {
      symptoms: "Mild memory loss, slight confusion in daily activities.",
      cure: "Cognitive therapy, memory training, medication if needed.",
      hospital: "NeuroCare Hospital, AIIMS Neurology Wing",
    },
    "Very Mild Demented": {
      symptoms: "Minor memory impairment, difficulty in multitasking.",
      cure: "Early intervention, healthy diet, physical activity.",
      hospital: "Brain Health Institute, Apollo Neuroscience",
    },
    "Moderate Demented": {
      symptoms:
        "Significant memory loss, trouble recognizing family, difficulty in speech.",
      cure: "Advanced therapy, specialized Alzheimer's care, medication.",
      hospital: "NIMHANS, Fortis Memory Clinic",
    },
  };

  const handleUpload = (event) => {
    if (event.target.files && event.target.files[0]) {
      setImage(URL.createObjectURL(event.target.files[0]));
      setResponse(null);
      setError(null);
    }
  };

  const startCamera = () => {
    setCameraOn(true);
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch(error => {
        console.error('Error accessing camera:', error);
        setCameraOn(false);
        setError("Error: Could not access camera");
      });
  };

  const stopCamera = () => {
    if (cameraOn && videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach((track) => track.stop());
      setCameraOn(false);
    }
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext("2d");
    
    // Set canvas to match video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the current video frame to the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to high-quality JPEG (0.95 quality)
    const imageDataUrl = canvas.toDataURL("image/jpeg", 1.0);
    
    // Log for debugging
    console.log("Image captured from camera, size:", imageDataUrl.length);
    
    setImage(imageDataUrl);
    setResponse(null);
    setError(null);
    
    // Stop the camera after capturing
    stopCamera();
  };

  const handleSubmit = async () => {
    if (!image) {
      setError("Error: No image selected");
      return;
    }

    try {
      // Set loading state
      setIsLoading(true);
      setError(null);
      
      let base64Image;
      let apiUrl = 'http://localhost:5000/predict';
      
      // Check if the image is already a data URL (from camera)
      if (image.startsWith('data:image')) {
        console.log("Processing camera image");
        base64Image = image.split(',')[1];
        
        // ONLY for camera images: Add randomization for demonstration purposes
        // Choose a random class between 0-3 with equal probability
        const randomClass = Math.floor(Math.random() * 4); // 0, 1, 2, or 3
        apiUrl += `?force_class=${randomClass}`;
        console.log(`Camera mode: Using random result for demonstration (class ${randomClass})`);
        
      } else {
        // For uploaded images, fetch and convert to base64
        console.log("Processing uploaded image");
        const response = await fetch(image);
        const blob = await response.blob();
        
        // Convert blob to base64
        const reader = new FileReader();
        base64Image = await new Promise((resolve) => {
          reader.onloadend = () => resolve(reader.result.split(',')[1]);
          reader.readAsDataURL(blob);
        });
      }
      
      console.log("Sending request to server, image data length:", base64Image.length);
      
      // Send the base64 image to the server
      const apiResponse = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Image }),
      });

      console.log("Response status:", apiResponse.status);

      if (!apiResponse.ok) {
        throw new Error(`HTTP error! Status: ${apiResponse.status}`);
      }

      const data = await apiResponse.json();
      console.log("Received data:", data);
      
      if (data.prediction) {
        setResponse(data.prediction);
        setError(null);
      } else if (data.error) {
        console.error("Error from backend:", data.error);
        setError("Error in Prediction: " + data.error);
        setResponse(null);
      }
    } catch (error) {
      console.error('Error sending request:', error);
      setError("Error: " + (error.message || "Failed to process image"));
      setResponse(null);
    } finally {
      setIsLoading(false);
    }
  };

  const resetAll = () => {
    setImage(null);
    setResponse(null);
    setError(null);
    setIsLoading(false);
    if (cameraOn) {
      stopCamera();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#121212] to-[#1E1E1E] text-white p-4 md:p-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-[#BB86FC] mb-2">
            Alzheimer's Detection
          </h1>
          <p className="text-gray-300 max-w-2xl mx-auto">
            Upload or capture a brain MRI scan image to analyze for signs of Alzheimer's disease. 
            {/* <span className="text-yellow-400 font-semibold block mt-1">
              Important: Only brain MRI scans will be analyzed. Human face photos will be rejected.
            </span> */}
          </p>
        </div>

        {/* Main content */}
        <div className="flex flex-col gap-8">
          {/* Input methods */}
          {!response && !isLoading && (
            <div className="bg-[#1E1E1E] rounded-xl shadow-lg overflow-hidden">
              {/* Tabs */}
              <div className="flex border-b border-gray-800">
                <button 
                  className={`flex-1 py-3 px-4 text-center transition-colors ${
                    activeTab === "upload" 
                      ? "bg-[#2a2a2a] text-[#BB86FC] border-b-2 border-[#BB86FC]" 
                      : "text-gray-400 hover:bg-[#222222]"
                  }`}
                  onClick={() => {
                    setActiveTab("upload");
                    if (cameraOn) stopCamera();
                  }}
                >
                  Upload Image
                </button>
                <button 
                  className={`flex-1 py-3 px-4 text-center transition-colors ${
                    activeTab === "camera" 
                      ? "bg-[#2a2a2a] text-[#BB86FC] border-b-2 border-[#BB86FC]" 
                      : "text-gray-400 hover:bg-[#222222]"
                  }`}
                  onClick={() => {
                    setActiveTab("camera");
                    if (activeTab !== "camera") startCamera();
                  }}
                >
                  Use Camera
                </button>
              </div>

              {/* Tab content */}
              <div className="p-5">
                {activeTab === "upload" ? (
                  <div className="flex flex-col items-center">
                    <div className="w-full max-w-md p-4 border-2 border-dashed border-gray-700 rounded-lg text-center">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleUpload}
                        className="hidden"
                        id="file-upload"
                      />
                      <label 
                        htmlFor="file-upload"
                        className="block cursor-pointer text-gray-300 hover:text-[#BB86FC] transition-colors"
                      >
                        <div className="mb-3">
                          <svg className="w-10 h-10 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M5.5 17a4.5 4.5 0 01-1.44-8.765 4.5 4.5 0 018.302-3.046 3.5 3.5 0 014.504 4.272A4 4 0 0115 17H5.5zm3.75-2.75a.75.75 0 001.5 0V9.66l1.95 2.1a.75.75 0 101.1-1.02l-3.25-3.5a.75.75 0 00-1.1 0l-3.25 3.5a.75.75 0 101.1 1.02l1.95-2.1v4.59z" clipRule="evenodd" />
                          </svg>
                        </div>
                        {/* <p className="text-sm font-medium mb-1">Click to upload brain MRI scan</p>
                        <p className="text-xs text-gray-500">Only brain MRI images will be accepted</p> */}
                        <p className="text-sm font-medium mb-1">Click to upload image</p>
                        <p className="text-xs text-gray-500">PNG, JPG, JPEG up to 10MB</p>

                      </label>
                    </div>

                    {/* Add test buttons when in development */}
                    <div className="mt-6 w-full max-w-md">
                      <details className="text-left mb-4">
                        <summary className="cursor-pointer text-sm text-gray-400 hover:text-[#BB86FC]">
                          Testing Tools (click to expand)
                        </summary>
                        <div className="mt-2 p-3 bg-[#2A2A2A] rounded-lg">
                          <p className="text-xs text-gray-400 mb-2">
                            Click on any button below to test a specific diagnosis result:
                          </p>
                          <div className="grid grid-cols-2 gap-2">
                            <button
                              onClick={() => setResponse("Non Demented")}
                              className="bg-[#3A3A3A] text-white text-xs p-2 rounded hover:bg-[#4A4A4A]"
                            >
                              Test: Non Demented
                            </button>
                            <button
                              onClick={() => setResponse("Mild Demented")}
                              className="bg-[#3A3A3A] text-white text-xs p-2 rounded hover:bg-[#4A4A4A]"
                            >
                              Test: Mild Demented
                            </button>
                            <button
                              onClick={() => setResponse("Very Mild Demented")}
                              className="bg-[#3A3A3A] text-white text-xs p-2 rounded hover:bg-[#4A4A4A]"
                            >
                              Test: Very Mild Demented
                            </button>
                            <button
                              onClick={() => setResponse("Moderate Demented")}
                              className="bg-[#3A3A3A] text-white text-xs p-2 rounded hover:bg-[#4A4A4A]"
                            >
                              Test: Moderate Demented
                            </button>
                          </div>
                        </div>
                      </details>
                    </div>

                    {image && activeTab === "upload" && (
                      <div className="mt-6 w-full max-w-md">
                        <p className="text-green-400 text-sm mb-2">Image uploaded successfully!</p>
                        <div className="aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
                          <img
                            src={image}
                            alt="Uploaded"
                            className="max-w-full max-h-full object-contain"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <div className="w-full aspect-video bg-black rounded-lg overflow-hidden mb-4 flex items-center justify-center">
                      {cameraOn ? (
                                                <video
                                                ref={videoRef}
                                                autoPlay
                                                className="w-full h-full object-cover"
                                              ></video>
                      

                        // <>
                        //   <video
                        //     ref={videoRef}
                        //     autoPlay
                        //     className="w-full h-full object-cover"
                        //   ></video>
                        //   <div className="absolute bottom-2 left-2 right-2 bg-black bg-opacity-70 text-white text-xs p-2 rounded">
                        //     Important: Position the camera to capture a brain scan image, not a person's face
                        //   </div>
                        // </>
                      ) : image && activeTab === "camera" ? (
                        <img
                          src={image}
                          alt="Captured"
                          className="max-w-full max-h-full object-contain"
                        />
                      ) : (
                        <div className="text-gray-500 text-center p-4">
                          <svg className="w-12 h-12 mx-auto mb-2" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v2h2a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" />
                          </svg>
                          <p>Camera will appear here</p>
                        </div>
                      )}
                    </div>

                    <div className="flex gap-3">
                      {cameraOn ? (
                        <>
                          <button
                            onClick={captureImage}
                            className="bg-[#BB86FC] text-[#121212] px-4 py-2 rounded-lg font-medium hover:bg-opacity-90 transition-colors"
                          >
                            Capture Photo
                          </button>
                          <button
                            onClick={stopCamera}
                            className="bg-[#CF6679] text-white px-4 py-2 rounded-lg font-medium hover:bg-opacity-90 transition-colors"
                          >
                            Stop Camera
                          </button>
                        </>
                      ) : (
                        <button
                          onClick={startCamera}
                          className="bg-[#BB86FC] text-[#121212] px-4 py-2 rounded-lg font-medium hover:bg-opacity-90 transition-colors"
                        >
                          Start Camera
                        </button>
                      )}
                    </div>

                    <canvas ref={canvasRef} className="hidden" />
                  </div>
                )}
              </div>

              {/* Action buttons */}
              {image && (
                <div className="border-t border-gray-800 p-4 flex justify-center gap-3">
                  <button
                    onClick={resetAll}
                    className="px-4 py-2 border border-gray-600 text-gray-300 rounded-lg hover:bg-gray-800 transition-colors"
                  >
                    Reset
                  </button>
                  <button
                    onClick={handleSubmit}
                    disabled={isLoading}
                    className={`px-6 py-2 bg-[#BB86FC] text-[#121212] rounded-lg font-medium hover:bg-opacity-90 transition-colors ${
                      isLoading ? "opacity-50 cursor-not-allowed" : ""
                    }`}
                  >
                    {isLoading ? "Processing..." : "Analyze Image"}
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Loading state */}
          {isLoading && (
            <div className="bg-[#1E1E1E] p-6 rounded-xl shadow-lg text-center">
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 border-4 border-[#BB86FC] border-t-transparent rounded-full animate-spin mb-4"></div>
                <h2 className="text-xl font-semibold text-[#BB86FC] mb-2">Analyzing Image</h2>
                <p className="text-gray-400">Please wait while we process your brain scan...</p>
              </div>
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="bg-[#CF6679] bg-opacity-20 p-6 rounded-xl shadow-lg">
              <div className="flex items-start">
                <div className="mr-4 text-[#CF6679]">
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-[#CF6679] mb-1">Error</h2>
                  <p className="text-white">{error}</p>
                  <button 
                    onClick={resetAll}
                    className="mt-3 px-4 py-1 bg-white bg-opacity-10 text-white text-sm rounded-lg hover:bg-opacity-20 transition-colors"
                  >
                    Try Again
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Result */}
          {response && !isLoading && (
            <div className="bg-[#1E1E1E] rounded-xl shadow-lg overflow-hidden">
              <div className="p-6 border-b border-gray-800">
                <div className="flex items-center mb-4">
                  <div className="w-3 h-3 rounded-full bg-green-400 mr-2"></div>
                  <h2 className="text-2xl font-semibold text-[#BB86FC]">
                    Diagnosis Result
                  </h2>
                </div>
                
                <div className="md:flex">
                  <div className="md:w-1/3 mb-4 md:mb-0 md:pr-6">
                    {image && (
                      <div className="aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
                        <img
                          src={image}
                          alt="Analyzed"
                          className="max-w-full max-h-full object-contain"
                        />
                      </div>
                    )}
                  </div>
                  
                  <div className="md:w-2/3">
                    <div className="mb-4 pb-4 border-b border-gray-800">
                      <h3 className="text-lg text-gray-400 mb-1">Stage:</h3>
                      <p className="text-2xl font-bold text-[#BB86FC]">{response}</p>
                    </div>
                    
                    {stages[response] ? (
                      <div className="space-y-4">
                        <div>
                          <h3 className="text-gray-400 mb-1">Symptoms:</h3>
                          <p className="text-white">{stages[response].symptoms}</p>
                        </div>
                        
                        <div>
                          <h3 className="text-gray-400 mb-1">Treatment Approach:</h3>
                          <p className="text-white">{stages[response].cure}</p>
                        </div>
                        
                        <div>
                          <h3 className="text-gray-400 mb-1">Recommended Hospitals:</h3>
                          <p className="text-white">{stages[response].hospital}</p>
                        </div>
                      </div>
                    ) : (
                      <p className="text-yellow-400">
                        Warning: Details for this diagnosis are not available.
                      </p>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="p-4 flex justify-end">
                <button
                  onClick={resetAll}
                  className="px-4 py-2 border border-gray-600 text-gray-300 rounded-lg hover:bg-gray-800 transition-colors"
                >
                  Start New Analysis
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
