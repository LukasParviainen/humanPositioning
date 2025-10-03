
// Human pose detection for iPhone 12 Pro Max to be used in landscapemode

import UIKit
import CoreML
import ARKit
import SceneKit


class ViewController: UIViewController, ARSessionDelegate, ARSCNViewDelegate {
    
    let overlayView = UIView()
    
    // MARK: - ARKit
    let session = ARSession()
    var arView: ARSCNView!
    var configuration = ARWorldTrackingConfiguration()
    
    
    // MARK: - Pose
    var model: yolov8n_pose!
    let COCO_PAIRS: [(Int, Int)] = [
        (5, 7), (7, 9),    // Left arm
        (6, 8), (8, 10),   // Right arm
        (5, 6),            // Shoulders
        (11, 13), (13, 15), // Left leg
        (12, 14), (14, 16), // Right leg
        (11, 12),          // Hips
        (5, 11), (6, 12)   // Torso
    ]
    
    
    // MARK: - Position
    struct TorsoPositionRecord {
        let personIndex: Int
        let timestamp: Date
        let x: Float
        let y: Float
        let z: Float
        let confidence: UInt8
    }
    var torsoPositionRecords: [TorsoPositionRecord] = []
    private var positionHistory: [PositionData] = [] {
        didSet { updateCountLabel() }
    }
    
    
    // MARK: - 3D re4ndering
    private let sharedSphereGeometry = SCNSphere(radius: 0.01)
    private let greenMaterial = SCNMaterial()
    private let yellowMaterial = SCNMaterial()
    private let redMaterial = SCNMaterial()
    private var isPermanentMode = false
    private var isTrackingMode = false
    private var renderedNodes: [SCNNode] = []
    
    
    // MARK: - FPS
    let frameProcessingInterval = 2
    let personCountLabel = UILabel()
    var isProcessingFrame = false
    var frameCounter = 0
    private var lastFrameTimestamp: TimeInterval = 0
    private var frameRate: Double = 0 {
        didSet { updateFPSLabel() }
    }
    
    // MARK: - Latency
    private var frameCaptureTimestamps: [CFTimeInterval] = []
    private var processingStartTimestamps: [CFTimeInterval] = []
    private var processingEndTimestamps: [CFTimeInterval] = []
    
    struct FramePerformanceMetrics {
        let timestamp: Date
        let frameLatency: Double
        let processingTime: Double
        let drawingTime: Double
        let personCount: Int
    }
    var framePerformanceMetrics: [FramePerformanceMetrics] = []
    
    // MARK: - Buttons
    private let resetButton = UIButton(type: .system)
    private let exportButton = UIButton(type: .system)
    private let toggleViewButton = UIButton(type: .system)
    private let startTrackingButton = UIButton(type: .system)
    private let permanenceButton = UIButton(type: .system)
    private let pauseMarkButton = UIButton(type: .system)
    private let exportMetricsButton = UIButton(type: .system)
    
    // MARK: - Labels and Sliders
    private let fpsLabel = UILabel()
    private let countLabel = UILabel()
    private let latencyLabel = UILabel()

    let confidenceSlider = UISlider()
    let confidenceLabel = UILabel()
    var personConfidenceThreshold: Float = 0.75
    
    
    // MARK: - Map
    private let topDownMapView = TopDownMapView()
    private var isMapVisible = false
    
    
    // MARK: - Date Formatters
    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        formatter.timeZone = TimeZone(identifier: "Europe/Helsinki")!
        return formatter
    }()
    
    private lazy var filenameDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        formatter.timeZone = TimeZone(identifier: "Europe/Helsinki")!
        return formatter
    }()

    
    //MARK: - Where these?
    private let ciContext = CIContext()
    var lastScale: CGFloat = 1.0
    var lastXOffset: CGFloat = 0.0
    var lastYOffset: CGFloat = 0.0

    
    // MARK: - Views
    private func setupARView() {
        arView = ARSCNView(frame: view.bounds)
        view.addSubview(arView)
        arView.session = session
        arView.delegate = self
    }
    
    private func setupOverlay() {
        overlayView.frame = view.bounds
        overlayView.backgroundColor = .clear
        view.addSubview(overlayView)
    }
    
    private func loadModel() {
        session.delegate = self
        let ARconfig = ARWorldTrackingConfiguration()
        ARconfig.frameSemantics = .sceneDepth
        session.run(ARconfig)
        
        let MLconfig = MLModelConfiguration()
        MLconfig.computeUnits = .cpuAndNeuralEngine
        model = try? yolov8n_pose(configuration: MLconfig)
    }
    
    private func setupMapUI() {
        topDownMapView.backgroundColor = UIColor.white.withAlphaComponent(0.9)
        topDownMapView.isHidden = true
        view.addSubview(topDownMapView)
        topDownMapView.frame = view.bounds
    }
    
    func setupConfidenceUI() {
        // Label
        confidenceLabel.frame = CGRect(x: 20, y: view.bounds.height - 100, width: 200, height: 30)
        confidenceLabel.textColor = .white
        confidenceLabel.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        confidenceLabel.text = "Confidence: \(Int(personConfidenceThreshold * 100))%"
        view.addSubview(confidenceLabel)

        // Slider
        confidenceSlider.frame = CGRect(x: 20, y: view.bounds.height - 60, width: view.bounds.width - 40, height: 30)
        confidenceSlider.minimumValue = 0.0
        confidenceSlider.maximumValue = 1.0
        confidenceSlider.value = personConfidenceThreshold
        confidenceSlider.addTarget(self, action: #selector(confidenceSliderChanged(_:)), for: .valueChanged)
        view.addSubview(confidenceSlider)
    }
    


    
    private func setupButtons() {
        // Toggle View Button
        toggleViewButton.setTitle("Show Map", for: .normal)
        toggleViewButton.backgroundColor = .white
        toggleViewButton.layer.cornerRadius = 8
        toggleViewButton.addTarget(self, action: #selector(toggleView), for: .touchUpInside)
        
        // Reset Button
        resetButton.setTitle("Reset", for: .normal)
        resetButton.backgroundColor = .white
        resetButton.layer.cornerRadius = 8
        resetButton.addTarget(self, action: #selector(resetData), for: .touchUpInside)
        
        // Export Button
        exportButton.setTitle("Export CSV", for: .normal)
        exportButton.backgroundColor = .white
        exportButton.layer.cornerRadius = 8
        exportButton.addTarget(self, action: #selector(exportCSV), for: .touchUpInside)
        
        // Start Tracking Button
        startTrackingButton.setTitle("Start Tracking", for: .normal)
        startTrackingButton.backgroundColor = .white
        startTrackingButton.layer.cornerRadius = 8
        startTrackingButton.addTarget(self, action: #selector(startTracking), for: .touchUpInside)
        
        // Permanence Button
        permanenceButton.setTitle("Persist: Off", for: .normal)
        permanenceButton.backgroundColor = .white
        permanenceButton.layer.cornerRadius = 8
        permanenceButton.addTarget(self, action: #selector(togglePermanence), for: .touchUpInside)
        permanenceButton.translatesAutoresizingMaskIntoConstraints = false
        
        // Pausemark Button
        pauseMarkButton.setTitle("Pausemark", for: .normal)
        pauseMarkButton.backgroundColor = .white
        pauseMarkButton.layer.cornerRadius = 8
        pauseMarkButton.addTarget(self, action: #selector(addPauseMark), for: .touchUpInside)
        pauseMarkButton.translatesAutoresizingMaskIntoConstraints = false
        
        
        // Export Metrics Button:
        exportMetricsButton.setTitle("Export Metrics", for: .normal)
        exportMetricsButton.backgroundColor = .white
        exportMetricsButton.layer.cornerRadius = 8
        exportMetricsButton.addTarget(self, action: #selector(exportPerformanceMetrics), for: .touchUpInside)

        
        // Add buttons to view
        [toggleViewButton, resetButton, exportButton, startTrackingButton, permanenceButton, pauseMarkButton, exportMetricsButton].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            view.addSubview($0)
        }


        // Layout constraints
        NSLayoutConstraint.activate([
            // Top row
            resetButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            resetButton.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            resetButton.widthAnchor.constraint(equalToConstant: 80),
            resetButton.heightAnchor.constraint(equalToConstant: 40),

            exportButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            exportButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            exportButton.widthAnchor.constraint(equalToConstant: 120),
            exportButton.heightAnchor.constraint(equalToConstant: 40),

            // Second row
            toggleViewButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            toggleViewButton.leadingAnchor.constraint(equalTo: resetButton.trailingAnchor, constant: 20),
            toggleViewButton.widthAnchor.constraint(equalToConstant: 120),
            toggleViewButton.heightAnchor.constraint(equalToConstant: 40),

            startTrackingButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            startTrackingButton.leadingAnchor.constraint(equalTo: toggleViewButton.trailingAnchor, constant: 20),
            startTrackingButton.widthAnchor.constraint(equalToConstant: 160),
            startTrackingButton.heightAnchor.constraint(equalToConstant: 40),
            
            permanenceButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            permanenceButton.leadingAnchor.constraint(equalTo: startTrackingButton.trailingAnchor, constant: 20),
            permanenceButton.widthAnchor.constraint(equalToConstant: 120),
            permanenceButton.heightAnchor.constraint(equalToConstant: 40),
            
            pauseMarkButton.topAnchor.constraint(equalTo: exportButton.bottomAnchor, constant: 20),
            pauseMarkButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            pauseMarkButton.widthAnchor.constraint(equalToConstant: 120),
            pauseMarkButton.heightAnchor.constraint(equalToConstant: 40),
            
        // Add constraints for the new button (adjust as needed based on your layout)
            exportMetricsButton.topAnchor.constraint(equalTo: pauseMarkButton.bottomAnchor, constant: 20),
            exportMetricsButton.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -20),
            exportMetricsButton.widthAnchor.constraint(equalToConstant: 120),
            exportMetricsButton.heightAnchor.constraint(equalToConstant: 40)
        
            
            
        ])
    }
    
    private func setupLabels() {
        // Count Label
        countLabel.text = "Count: 0"
        countLabel.textColor = .white
        countLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        countLabel.layer.cornerRadius = 6
        countLabel.clipsToBounds = true
        countLabel.textAlignment = .center
        countLabel.font = UIFont.boldSystemFont(ofSize: 14)
        countLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(countLabel)
        
        // FPS Label
        fpsLabel.text = "FPS: 0"
        fpsLabel.textColor = .white
        fpsLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        fpsLabel.layer.cornerRadius = 6
        fpsLabel.clipsToBounds = true
        fpsLabel.textAlignment = .center
        fpsLabel.font = UIFont.boldSystemFont(ofSize: 14)
        fpsLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(fpsLabel)
        
        // People Count Label
        personCountLabel.text = "People: 0"
        personCountLabel.textColor = .white
        personCountLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        personCountLabel.layer.cornerRadius = 6
        personCountLabel.clipsToBounds = true
        personCountLabel.textAlignment = .center
        personCountLabel.font = UIFont.boldSystemFont(ofSize: 14)
        personCountLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(personCountLabel)
        
        // Latency Label
       latencyLabel.text = "Latency: 0ms"
       latencyLabel.textColor = .white
       latencyLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
       latencyLabel.layer.cornerRadius = 6
       latencyLabel.clipsToBounds = true
       latencyLabel.textAlignment = .center
       latencyLabel.font = UIFont.boldSystemFont(ofSize: 14)
       latencyLabel.translatesAutoresizingMaskIntoConstraints = false
       view.addSubview(latencyLabel)
    
        
        NSLayoutConstraint.activate([
            personCountLabel.topAnchor.constraint(equalTo: fpsLabel.bottomAnchor, constant: 10),
            personCountLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            personCountLabel.widthAnchor.constraint(equalToConstant: 100),
            personCountLabel.heightAnchor.constraint(equalToConstant: 30),
    
            countLabel.topAnchor.constraint(equalTo: resetButton.bottomAnchor, constant: 20),
            countLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            countLabel.widthAnchor.constraint(equalToConstant: 100),
            countLabel.heightAnchor.constraint(equalToConstant: 30),
            
            fpsLabel.topAnchor.constraint(equalTo: countLabel.bottomAnchor, constant: 10),
            fpsLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            fpsLabel.widthAnchor.constraint(equalToConstant: 100),
            fpsLabel.heightAnchor.constraint(equalToConstant: 30),
            
            latencyLabel.topAnchor.constraint(equalTo: personCountLabel.bottomAnchor, constant: 10),
            latencyLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 20),
            latencyLabel.widthAnchor.constraint(equalToConstant: 150),
            latencyLabel.heightAnchor.constraint(equalToConstant: 30)
        ])
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        UIApplication.shared.isIdleTimerDisabled = true
        
        greenMaterial.diffuse.contents = UIColor.green
        yellowMaterial.diffuse.contents = UIColor.yellow
        redMaterial.diffuse.contents = UIColor.red
        [greenMaterial, yellowMaterial, redMaterial].forEach {
            $0.lightingModel = .constant
            $0.isDoubleSided = true
        }
        
        setupARView()
        setupOverlay()
        loadModel()
        setupMapUI()
        setupConfidenceUI()
        setupButtons()
        setupLabels()
    }
    
    
    // MARK: - Label Updates
    private func updateCountLabel() {
        countLabel.text = "Count: \(positionHistory.count)"
    }
    
    private func updateFPSLabel() {
        fpsLabel.text = String(format: "FPS: %.1f", frameRate)
    }
    
    
    // MARK: - Button and Slider Actions
    @objc private func resetData() {
        positionHistory.removeAll()
        torsoPositionRecords.removeAll()
        renderedNodes.forEach { $0.removeFromParentNode() }
        renderedNodes.removeAll()
        overlayView.layer.sublayers?.forEach { $0.removeFromSuperlayer() }
        isTrackingMode = false
        startTrackingButton.setTitle("Start Tracking", for: .normal)
        startTrackingButton.isEnabled = true
        print("Data reset.")
    }
    

        
    @objc private func exportCSV() {
        let dateString = filenameDateFormatter.string(from: Date())
        let fileName = "Torso3DPositions_\(dateString).csv"

        let header = "Person,Timestamp,X,Y,Z,confidence\n"
        var csv = header
        for record in torsoPositionRecords {
            // Format timestamp using dateFormatter
            let formattedDate = dateFormatter.string(from: record.timestamp)
            let line = "\(record.personIndex),\(formattedDate),\(record.x),\(record.y),\(record.z), \(record.confidence)\n"
            csv += line
        }
            

        // Save to temporary location
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        do {
            try csv.write(to: path, atomically: true, encoding: .utf8)
            let activityVC = UIActivityViewController(activityItems: [path], applicationActivities: nil)
            activityVC.popoverPresentationController?.sourceView = self.view
            present(activityVC, animated: true)
        } catch {
            print("❌ Failed to write CSV: \(error)")
        }
    }
    
    @objc private func startTracking() {
        isTrackingMode = true
        startTrackingButton.setTitle("Tracking All", for: .normal)
        startTrackingButton.isEnabled = false
        print("Tracking mode activated.")
    }
    
    @objc private func toggleView() {
        isMapVisible.toggle()
        
        arView.isHidden = isMapVisible
        overlayView.isHidden = isMapVisible
        topDownMapView.isHidden = !isMapVisible
        
        let buttonTitle = isMapVisible ? "Show Camera" : "Show Map"
        toggleViewButton.setTitle(buttonTitle, for: .normal)
        
        if isMapVisible {
            topDownMapView.positions = positionHistory
            view.bringSubviewToFront(toggleViewButton)
            view.bringSubviewToFront(resetButton)
            view.bringSubviewToFront(exportButton)
            view.bringSubviewToFront(countLabel)
            view.bringSubviewToFront(fpsLabel)
        }
    }
    
    @objc private func togglePermanence() {
        isPermanentMode.toggle()
        let title = isPermanentMode ? "Permanent: On" : "Permanent: Off"
        permanenceButton.setTitle(title, for: .normal)
        
        if !isPermanentMode {
            // Clear existing spheres when turning off persistence
            renderedNodes.forEach { $0.removeFromParentNode() }
            renderedNodes.removeAll()
        }
    }
    
    @objc func confidenceSliderChanged(_ sender: UISlider) {
        personConfidenceThreshold = sender.value
        confidenceLabel.text = String(format: "Confidence: %.0f%%", personConfidenceThreshold * 100)
    }
    
    @objc func addPauseMark() {
        let currentDate = Date()
        torsoPositionRecords.append(TorsoPositionRecord(
            personIndex: 0,
            timestamp: currentDate,
            x: 0,
            y: 0,
            z: 0,
            confidence: 0
        ))
    }
    
    @objc private func exportPerformanceMetrics() {
        let dateString = filenameDateFormatter.string(from: Date())
        let fileName = "PerformanceMetrics_\(dateString).csv"

        let header = "Timestamp,FrameLatency(ms),ProcessingTime(ms),DrawingTime(ms),PersonCount\n"
        var csv = header
        for metric in framePerformanceMetrics {
            let formattedDate = dateFormatter.string(from: metric.timestamp)
            let line = "\(formattedDate),\(String(format: "%.2f", metric.frameLatency)),\(String(format: "%.2f", metric.processingTime)),\(String(format: "%.2f", metric.drawingTime)),\(metric.personCount)\n"
            csv += line
        }

        // Save to temporary location
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)

        do {
            try csv.write(to: path, atomically: true, encoding: .utf8)
            let activityVC = UIActivityViewController(activityItems: [path], applicationActivities: nil)
            activityVC.popoverPresentationController?.sourceView = self.view
            present(activityVC, animated: true)
        } catch {
            print("❌ Failed to write performance CSV: \(error)")
        }
    }
    

    
    
    
    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameCounter += 1
        if frameCounter % frameProcessingInterval != 0 { return }
        guard !isProcessingFrame else { return }
        isProcessingFrame = true

        let pixelBuffer = frame.capturedImage
        let captureTime = CACurrentMediaTime()
        frameCaptureTimestamps.append(captureTime)
        
        // FPS Calculation
        let currentTime = frame.timestamp
        if lastFrameTimestamp > 0 {
            let delta = currentTime - lastFrameTimestamp
            frameRate = 1.0 / delta
        }
        lastFrameTimestamp = currentTime

        DispatchQueue.global(qos: .userInitiated).async {
            let processingStartTime = CACurrentMediaTime()
            self.processingStartTimestamps.append(processingStartTime)
            
            self.runPoseEstimation(on: pixelBuffer, frame: frame)
            
            let processingEndTime = CACurrentMediaTime()
                    self.processingEndTimestamps.append(processingEndTime)

            DispatchQueue.main.async {
                self.isProcessingFrame = false
            }
        }
    }

    
    
    
    
    func updateOverlayFrame(for pixelBuffer: CVPixelBuffer) {
        let imageWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let imageHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        let aspect = imageWidth / imageHeight

        let screenWidth = view.bounds.width
        let screenHeight = screenWidth / aspect

        let yOffset = (view.bounds.height - screenHeight) / 2
        overlayView.frame = CGRect(x: 0, y: yOffset, width: screenWidth, height: screenHeight)
    }

    
    
    
    
    
    
    
    
    
    
    
    func runPoseEstimation(on pixelBuffer: CVPixelBuffer, frame: ARFrame) {
        DispatchQueue.main.async {
            if self.overlayView.frame == self.view.bounds {
                self.updateOverlayFrame(for: pixelBuffer)
            }
        }

        guard let resized = resizePixelBuffer(pixelBuffer, width: 640, height: 640) else {
            print("❌ Resize failed")
            return
        }

        let resizedBuffer = resized.buffer
        lastScale = resized.scale
        lastXOffset = resized.xOffset
        lastYOffset = resized.yOffset

        guard let output = try? model.prediction(image: resizedBuffer) else {
            print("❌ Prediction failed")
            return
        }

        let outputArray = output.var_1035
        let shape = outputArray.shape  // [1, 56, 8400]
        guard shape.count == 3,
              let count = shape[2] as? Int else {
            return
        }

        // Step 1: Gather all raw detections above threshold
        var detections: [(index: Int, confidence: Float, box: CGRect)] = []

        for i in 0..<count {
            let idx = { (c: Int) -> [NSNumber] in [0, NSNumber(value: c), NSNumber(value: i)] }
            let confidence = outputArray[idx(4)].floatValue
            if confidence < personConfidenceThreshold { continue }

            let box = getBoundingBox(from: outputArray, index: i)
            detections.append((i, confidence, box))
        }

        // Step 2: Apply Non-Maximum Suppression (NMS)
        let nmsIoUThreshold: CGFloat = 0.5
        var finalDetections: [(index: Int, box: CGRect)] = []

        detections.sort { $0.confidence > $1.confidence }
        var active = [Bool](repeating: true, count: detections.count)

        for i in 0..<detections.count {
            if !active[i] { continue }
            finalDetections.append((detections[i].index, detections[i].box))

            for j in (i + 1)..<detections.count {
                if !active[j] { continue }
                let iou = iouBetween(detections[i].box, detections[j].box)
                if iou > nmsIoUThreshold {
                    active[j] = false
                }
            }
        }

        // Step 3: Decode final selected keypoints
        var peopleKeypoints: [[CGPoint]] = []
        var boundingBoxes: [CGRect] = []
        
        for (i, box) in finalDetections {
            let idx = { (c: Int) -> [NSNumber] in [0, NSNumber(value: c), NSNumber(value: i)] }

            var keypoints: [CGPoint] = []
            for k in 0..<17 {
                let base = 5 + k * 3
                let kpX = outputArray[idx(base)].floatValue
                let kpY = outputArray[idx(base + 1)].floatValue
                keypoints.append(CGPoint(x: CGFloat(kpX), y: CGFloat(kpY)))
            }

            peopleKeypoints.append(keypoints)
            boundingBoxes.append(box)
        }

        // Step 4: Draw
        DispatchQueue.main.async {
            self.drawMultipleKeypoints(peopleKeypoints, boundingBoxes: boundingBoxes, originalPixelBuffer: pixelBuffer, frame: frame)
        }
    }

    
    
   //drawmultiplekeypoint phase
    func drawMultipleKeypoints(_ allKeypoints: [[CGPoint]], boundingBoxes: [CGRect], originalPixelBuffer: CVPixelBuffer, frame: ARFrame) {
        let drawingStartTime = CACurrentMediaTime()
       
        clearPreviousVisualizations()
        
        let inputImageSize = getImageSize(from: originalPixelBuffer)
        let overlaySize = overlayView.bounds.size
        
        for (index, keypoints) in allKeypoints.enumerated() {
            let box = boundingBoxes[index]
            
            // Draw bounding box
            drawBoundingBox(box, inputImageSize: inputImageSize, overlaySize: overlaySize)
            
            // Draw keypoints and skeleton
            let mappedPoints = drawKeypointsAndSkeleton(keypoints, inputImageSize: inputImageSize, overlaySize: overlaySize)
            
            // Process torso if enough keypoints
            if keypoints.count > 12 {
                processTorso(keypoints: keypoints, index: index,
                            inputImageSize: inputImageSize, overlaySize: overlaySize,
                            frame: frame, mappedPoints: mappedPoints)
            }
        }
        
        updateUI(personCount: allKeypoints.count)
        
        // Calculate latency
        let drawingEndTime = CACurrentMediaTime()
        
        // Get the timestamps for this frame
        guard let captureTime = frameCaptureTimestamps.first,
              let processStart = processingStartTimestamps.first,
              let processEnd = processingEndTimestamps.first else {
            return
        }
        
        let totalLatency = (drawingEndTime - captureTime) * 1000
        let processingTime = (processEnd - processStart) * 1000
        let drawingTime = (drawingEndTime - drawingStartTime) * 1000

        // Store frame performance metrics
        let currentDate = Date()
        framePerformanceMetrics.append(FramePerformanceMetrics(
            timestamp: currentDate,
            frameLatency: totalLatency,
            processingTime: processingTime,
            drawingTime: drawingTime,
            personCount: allKeypoints.count
        ))

        // Update UI
        latencyLabel.text = String(format: "Latency: %.1fms", totalLatency)
            
        // Remove the used timestamps
        frameCaptureTimestamps.removeFirst()
        processingStartTimestamps.removeFirst()
        processingEndTimestamps.removeFirst()
    }

    // MARK: - Helper Functions

    private func clearPreviousVisualizations() {
        if isTrackingMode && !isPermanentMode {
            renderedNodes.forEach { $0.removeFromParentNode() }
            renderedNodes.removeAll()
        }
        overlayView.layer.sublayers?.forEach { $0.removeFromSuperlayer() }
    }

    private func getImageSize(from pixelBuffer: CVPixelBuffer) -> CGSize {
        return CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                     height: CVPixelBufferGetHeight(pixelBuffer))
    }

    private func drawBoundingBox(_ box: CGRect, inputImageSize: CGSize, overlaySize: CGSize) {
        let mappedRect = mapRectToOverlay(box, inputImageSize: inputImageSize, overlaySize: overlaySize)
        
        let boxLayer = CALayer()
        boxLayer.frame = mappedRect
        boxLayer.borderColor = UIColor.systemRed.cgColor
        boxLayer.borderWidth = 2.0
        overlayView.layer.addSublayer(boxLayer)
    }

    private func drawKeypointsAndSkeleton(_ keypoints: [CGPoint], inputImageSize: CGSize, overlaySize: CGSize) -> [CGPoint] {
        var mappedPoints: [CGPoint] = []
        
        // Draw keypoints
        for point in keypoints {
            let mappedPoint = mapPointToOverlay(point, inputImageSize: inputImageSize, overlaySize: overlaySize)
            mappedPoints.append(mappedPoint)
            
            let dot = createDotLayer(at: mappedPoint)
            overlayView.layer.addSublayer(dot)
        }
        
        // Draw skeleton
        drawSkeleton(mappedPoints: mappedPoints)
        
        return mappedPoints
    }

    private func drawSkeleton(mappedPoints: [CGPoint]) {
        for (start, end) in COCO_PAIRS {
            guard start < mappedPoints.count && end < mappedPoints.count else { continue }
            
            let line = createLineLayer(from: mappedPoints[start], to: mappedPoints[end])
            overlayView.layer.addSublayer(line)
        }
    }

    private func processTorso(keypoints: [CGPoint], index: Int, inputImageSize: CGSize,
                             overlaySize: CGSize, frame: ARFrame, mappedPoints: [CGPoint]) {
        guard let torsoMid = calculateTorsoMidpoint(from: keypoints) else { return }
        
        let mappedTorsoPoint = mapPointToOverlay(torsoMid, inputImageSize: inputImageSize, overlaySize: overlaySize)
        drawTorsoDot(at: mappedTorsoPoint)
        
        if let sceneDepth = frame.sceneDepth, let confidenceMap = sceneDepth.confidenceMap {
            processDepthInformation(for: torsoMid, personIndex: index,
                                   inputImageSize: inputImageSize, frame: frame,
                                   depthMap: sceneDepth.depthMap, confidenceMap: confidenceMap)
        }
    }

    private func calculateTorsoMidpoint(from keypoints: [CGPoint]) -> CGPoint? {
        guard keypoints.count > 12 else { return nil }
        
        let leftShoulder = keypoints[5]
        let rightShoulder = keypoints[6]
        let leftHip = keypoints[11]
        let rightHip = keypoints[12]
        
        let shoulderMid = CGPoint(
            x: (leftShoulder.x + rightShoulder.x) / 2,
            y: (leftShoulder.y + rightShoulder.y) / 2
        )
        let hipMid = CGPoint(
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2
        )
        
        return CGPoint(
            x: (shoulderMid.x + hipMid.x) / 2,
            y: (shoulderMid.y + hipMid.y) / 2
        )
    }

    private func processDepthInformation(for torsoMid: CGPoint, personIndex: Int,
                                       inputImageSize: CGSize, frame: ARFrame,
                                       depthMap: CVPixelBuffer, confidenceMap: CVPixelBuffer) {
        // Calculate unscaled torso coordinates
        let unpaddedTorsoX = (torsoMid.x - lastXOffset) / lastScale
        let unpaddedTorsoY = (torsoMid.y - lastYOffset) / lastScale

        // Get depth map resolution
        let depthResolution = CGSize(width: CVPixelBufferGetWidth(depthMap),
                                    height: CVPixelBufferGetHeight(depthMap))
        
        // Rescale torso midpoint to match depth map resolution
        let depthX = Int(unpaddedTorsoX * (depthResolution.width / inputImageSize.width))
        let depthY = Int(unpaddedTorsoY * (depthResolution.height / inputImageSize.height))

        // Check bounds
        guard depthX >= 0, depthX < Int(depthResolution.width),
              depthY >= 0, depthY < Int(depthResolution.height) else {
            print("⚠️ Torso point out of bounds for depth map")
            return
        }

        // Read depth value
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        let depthBaseAddr = CVPixelBufferGetBaseAddress(depthMap)!
        let depthRowBytes = CVPixelBufferGetBytesPerRow(depthMap)
        let depthRowData = depthBaseAddr + depthY * depthRowBytes
        let depthValue = depthRowData.assumingMemoryBound(to: Float32.self)[depthX]
        CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)

        // Read confidence value
        CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
        let confidenceBaseAddr = CVPixelBufferGetBaseAddress(confidenceMap)!
        let confidenceRowBytes = CVPixelBufferGetBytesPerRow(confidenceMap)
        let confidenceRowData = confidenceBaseAddr + depthY * confidenceRowBytes
        let confidenceValue = confidenceRowData.assumingMemoryBound(to: UInt8.self)[depthX]
        CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly)

        // Validate depth value
        if depthValue.isNaN || depthValue <= 0 {
            print("⚠️ Invalid depth at torso point")
            return
        }

        // Reproject to 3D using camera intrinsics
        let intrinsics = frame.camera.intrinsics
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        // Flip Y coordinate for correct coordinate system
        let flippedTorsoY = inputImageSize.height - unpaddedTorsoY
        let x = (Float(unpaddedTorsoX) - cx) * depthValue / fx
        let y = (Float(flippedTorsoY) - cy) * depthValue / fy
        let z = depthValue

        // Transform from camera to world space
        let cameraTransform = frame.camera.transform
        let torsoCameraPosition = SIMD4<Float>(x, y, -z, 1.0)
        let torsoWorldPosition = cameraTransform * torsoCameraPosition
        let worldPosition = SIMD3<Float>(torsoWorldPosition.x,
                                         torsoWorldPosition.y,
                                         torsoWorldPosition.z)

        // Create visualization node if in tracking mode
        if isTrackingMode {
            let clonedGeometry = sharedSphereGeometry.copy() as! SCNGeometry
            
            // Set color based on confidence
            switch confidenceValue {
            case 2:
                clonedGeometry.firstMaterial = greenMaterial
            case 1:
                clonedGeometry.firstMaterial = yellowMaterial
            default:
                clonedGeometry.firstMaterial = redMaterial
            }

            let node = SCNNode(geometry: clonedGeometry)
            node.simdPosition = worldPosition
            arView.scene.rootNode.addChildNode(node)
            renderedNodes.append(node)
        }

        
        // Record position data
        let currentDate = Date()
        torsoPositionRecords.append(TorsoPositionRecord(
            personIndex: personIndex + 1,
            timestamp: currentDate,
            x: worldPosition.x,
            y: worldPosition.y,
            z: worldPosition.z,
            confidence: confidenceValue
        ))

        let positionData = PositionData(
            timestamp: currentDate,
            position: worldPosition,
            confidence: confidenceValue
        )

        positionHistory.append(positionData)

        print("✅ Person \(personIndex + 1) torso at \(dateFormatter.string(from: currentDate)) → x: \(worldPosition.x), y: \(worldPosition.y), z: \(worldPosition.z), confidence: \(confidenceValue)")
    }
    

    private func mapPointToOverlay(_ point: CGPoint, inputImageSize: CGSize, overlaySize: CGSize) -> CGPoint {
        let unpaddedX = (point.x - lastXOffset) / lastScale
        let unpaddedY = (point.y - lastYOffset) / lastScale

        return CGPoint(
            x: unpaddedX * (overlaySize.width / inputImageSize.width),
            y: unpaddedY * (overlaySize.height / inputImageSize.height)
        )
    }

    private func mapRectToOverlay(_ rect: CGRect, inputImageSize: CGSize, overlaySize: CGSize) -> CGRect {
        let unpaddedX = (rect.origin.x - lastXOffset) / lastScale
        let unpaddedY = (rect.origin.y - lastYOffset) / lastScale
        let unpaddedWidth = rect.size.width / lastScale
        let unpaddedHeight = rect.size.height / lastScale

        return CGRect(
            x: unpaddedX * (overlaySize.width / inputImageSize.width),
            y: unpaddedY * (overlaySize.height / inputImageSize.height),
            width: unpaddedWidth * (overlaySize.width / inputImageSize.width),
            height: unpaddedHeight * (overlaySize.height / inputImageSize.height)
        )
    }

    private func createDotLayer(at point: CGPoint) -> CALayer {
        let dot = CALayer()
        dot.frame = CGRect(x: point.x - 3, y: point.y - 3, width: 6, height: 6)
        dot.cornerRadius = 3
        dot.backgroundColor = UIColor.systemPink.cgColor
        return dot
    }

    private func createLineLayer(from start: CGPoint, to end: CGPoint) -> CAShapeLayer {
        let line = CAShapeLayer()
        let path = UIBezierPath()
        path.move(to: start)
        path.addLine(to: end)
        line.path = path.cgPath
        line.strokeColor = UIColor.blue.cgColor
        line.lineWidth = 2.0
        return line
    }

    private func drawTorsoDot(at point: CGPoint) {
        let torsoDot = CALayer()
        torsoDot.frame = CGRect(x: point.x - 6, y: point.y - 6, width: 12, height: 12)
        torsoDot.cornerRadius = 6
        torsoDot.backgroundColor = UIColor.yellow.cgColor
        torsoDot.borderColor = UIColor.black.cgColor
        torsoDot.borderWidth = 1.5
        overlayView.layer.addSublayer(torsoDot)
    }

    private func updateUI(personCount: Int) {
        if isMapVisible {
            topDownMapView.positions = positionHistory
        }
        personCountLabel.text = "People: \(personCount)"
    }
    



    
  
    
    // MARK: - Utility Functions
    func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int)
    -> (buffer: CVPixelBuffer, scale: CGFloat, xOffset: CGFloat, yOffset: CGFloat)? {
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let inputSize = ciImage.extent.size
        
        let outputSize = CGSize(width: width, height: height)
        let scale = min(outputSize.width / inputSize.width, outputSize.height / inputSize.height)
        
        let scaledWidth = inputSize.width * scale
        let scaledHeight = inputSize.height * scale
        let dx = (outputSize.width - scaledWidth) / 2.0
        let dy = (outputSize.height - scaledHeight) / 2.0
        
        let transform = CGAffineTransform(scaleX: scale, y: scale)
            .translatedBy(x: dx / scale, y: dy / scale)
        let scaledImage = ciImage.transformed(by: transform)
        
        var resizedBuffer: CVPixelBuffer?
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary

        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                         kCVPixelFormatType_32BGRA, attrs, &resizedBuffer)
        guard status == kCVReturnSuccess, let result = resizedBuffer else {
            return nil
        }

        ciContext.render(scaledImage, to: result)
        return (result, scale, dx, dy)
    }
    
    func iouBetween(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0.0 }
        let union = a.union(b)
        return intersection.area / union.area
    }

    func getBoundingBox(from outputArray: MLMultiArray, index i: Int) -> CGRect {
        let idx = { (c: Int) -> [NSNumber] in [0, NSNumber(value: c), NSNumber(value: i)] }
        let centerX = outputArray[idx(0)].floatValue
        let centerY = outputArray[idx(1)].floatValue
        let width = outputArray[idx(2)].floatValue
        let height = outputArray[idx(3)].floatValue

        return CGRect(x: CGFloat(centerX - width / 2),
                      y: CGFloat(centerY - height / 2),
                      width: CGFloat(width),
                      height: CGFloat(height))
    }
}


// MARK: - Top-Down Map View
class TopDownMapView: UIView {
    var positions: [PositionData] = [] {
        didSet { setNeedsDisplay() }
    }
    
    private let gridSize: CGFloat = 20.0  // 20m x 20m grid
    private let gridStep: CGFloat = 5.0   // 5m grid steps
    private let pointSize: CGFloat = 4.0  // Size of position points
    private let padding: CGFloat = 30.0   // Padding around the grid
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        guard let context = UIGraphicsGetCurrentContext() else { return }
        
        let drawableWidth = bounds.width - (padding * 2)
        let drawableHeight = bounds.height - (padding * 2)
        let scale = min(drawableWidth, drawableHeight) / gridSize
        let gridOriginX = padding
        let gridOriginY = padding
        
        // Draw grid
        drawGrid(in: context, scale: scale, originX: gridOriginX, originY: gridOriginY)
        
        // Draw positions
        context.setFillColor(UIColor.red.cgColor)
        for data in positions {
            let color: UIColor
            switch data.confidence{
            case 2: color = .green      // high measurement confidence
            case 1: color = .yellow     // medium measurement confidence
            default: color = .red        // low measurement confidence
            }
            context.setFillColor(color.cgColor)

            
            let point = CGPoint(
                x: gridOriginX + (drawableWidth / 2) + CGFloat(data.position.x) * scale,
                y: gridOriginY + (drawableHeight / 2) + CGFloat(data.position.z) * scale
            )
            let dotRect = CGRect(x: point.x - pointSize/2, y: point.y - pointSize/2, width: pointSize, height: pointSize)
            context.fillEllipse(in: dotRect)
        }
        
        // Draw device marker
        context.setFillColor(UIColor.blue.cgColor)
        let deviceSize: CGFloat = 10.0
        let centerX = gridOriginX + (drawableWidth / 2)
        let centerY = gridOriginY + (drawableHeight / 2)
        context.fillEllipse(in: CGRect(x: centerX - deviceSize/2, y: centerY - deviceSize/2, width: deviceSize, height: deviceSize))
    }
    
    
    private func drawGrid(in context: CGContext, scale: CGFloat, originX: CGFloat, originY: CGFloat) {
        let halfGrid = gridSize / 2
        let drawableWidth = bounds.width - (padding * 2)
        let drawableHeight = bounds.height - (padding * 2)
        let centerX = originX + (drawableWidth / 2)
        let centerY = originY + (drawableHeight / 2)
        
        context.setStrokeColor(UIColor.lightGray.cgColor)
        context.setLineWidth(1.0)
        
        // Vertical lines
        for x in stride(from: -halfGrid, through: halfGrid, by: gridStep) {
            let viewX = centerX + x * scale
            context.move(to: CGPoint(x: viewX, y: originY))
            context.addLine(to: CGPoint(x: viewX, y: originY + drawableHeight))
        }
        
        // Horizontal lines
        for y in stride(from: -halfGrid, through: halfGrid, by: gridStep) {
            let viewY = centerY - y * scale
            context.move(to: CGPoint(x: originX, y: viewY))
            context.addLine(to: CGPoint(x: originX + drawableWidth, y: viewY))
        }
        
        context.strokePath()
        
        // Axes
        context.setStrokeColor(UIColor.darkGray.cgColor)
        context.setLineWidth(2.0)
        context.move(to: CGPoint(x: originX, y: centerY))
        context.addLine(to: CGPoint(x: originX + drawableWidth, y: centerY))
        context.move(to: CGPoint(x: centerX, y: originY))
        context.addLine(to: CGPoint(x: centerX, y: originY + drawableHeight))
        context.strokePath()
    }
}




extension CGRect {
    var area: CGFloat { width * height }

    func union(_ r: CGRect) -> CGRect {
        return CGRect(x: min(minX, r.minX),
                      y: min(minY, r.minY),
                      width: max(maxX, r.maxX) - min(minX, r.minX),
                      height: max(maxY, r.maxY) - min(minY, r.minY))
    }
}

// MARK: - Position Data Model
struct PositionData {
    let timestamp: Date
    let position: SIMD3<Float>
    let confidence: UInt8
}
