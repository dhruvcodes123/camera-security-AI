# Face Detection, Tracking & Re-Identification Project

This project implements real-time face detection, tracking, and re-identification using **SCRFD** for face detection and **ArcFace** for face recognition, with **FAISS** vector database for fast similarity search.

## 🎯 Features

- **Fast Face Detection**: SCRFD models with different performance/accuracy trade-offs
- **Accurate Face Recognition**: ArcFace models for robust face embeddings
- **Real-time Re-identification**: FAISS-powered fast similarity search
- **Multiple Input Sources**: Webcam, video files, or image sequences
- **Flexible Configuration**: Easy model switching and parameter tuning
- **Pre-built Database**: Includes Friends TV show characters for demo

## 📁 Project Structure

```
Low face detection-tracking/
├── face_reid_env/          # Python virtual environment
├── assets/
│   ├── faces/              # Face images for re-identification
│   │   ├── Chandler.png    # Friends TV show characters
│   │   ├── Joey.png
│   │   ├── Monica.png
│   │   ├── Phoebe.png
│   │   ├── Rachel.png
│   │   └── Ross.png
│   ├── demo.mp4           # Demo video
│   └── in_video.mp4       # Input video
├── weights/               # ONNX model weights
│   ├── det_500m.onnx     # SCRFD 500M (2.41 MB)
│   ├── det_2.5g.onnx     # SCRFD 2.5G (3.14 MB)  
│   ├── det_10g.onnx      # SCRFD 10G (16.1 MB)
│   ├── w600k_mbf.onnx    # ArcFace MobileFace (13 MB)
│   └── w600k_r50.onnx    # ArcFace ResNet-50 (166 MB)
├── database/             # FAISS database storage
├── models/               # Model implementations
├── utils/                # Helper utilities
├── config.py            # Project configuration
├── setup_env.sh         # Setup script
└── main.py              # Main application
```

## 🚀 Quick Start

### 1. Environment Setup
The virtual environment and dependencies are already installed. To activate:

```bash
source face_reid_env/bin/activate
```

### 2. Verify Setup
Run the setup verification script:

```bash
./setup_env.sh
```

### 3. View Model Information
```bash
python config.py
```

### 4. Run the Application

**With webcam:**
```bash
python main.py --source 0
```

**With demo video:**
```bash
python main.py --source assets/demo.mp4
```

**With custom video:**
```bash
python main.py --source path/to/your/video.mp4
```

## ⚙️ Configuration Options

### Model Selection

**Fast Performance (Recommended for real-time):**
```bash
python main.py --det-weight weights/det_500m.onnx --rec-weight weights/w600k_mbf.onnx
```

**Balanced Performance:**
```bash
python main.py --det-weight weights/det_2.5g.onnx --rec-weight weights/w600k_mbf.onnx
```

**Best Accuracy:**
```bash
python main.py --det-weight weights/det_10g.onnx --rec-weight weights/w600k_r50.onnx
```

### Threshold Tuning

**More Sensitive Detection:**
```bash
python main.py --confidence-thresh 0.3 --similarity-thresh 0.6
```

**Less Sensitive (Fewer False Positives):**
```bash
python main.py --confidence-thresh 0.7 --similarity-thresh 0.3
```

### Display Options

**Automatic Video Sizing:**
- Large videos are automatically resized to fit your screen (1200x800 max)
- Original aspect ratio is always maintained
- Output video keeps full original resolution

**Process without display (faster):**
```bash
python main.py --source assets/demo.mp4 --no-display
```

### Complete Parameter List

```bash
python main.py \
  --det-weight weights/det_2.5g.onnx \
  --rec-weight weights/w600k_mbf.onnx \
  --confidence-thresh 0.5 \
  --similarity-thresh 0.4 \
  --faces-dir assets/faces \
  --source assets/demo.mp4 \
  --max-num 10 \
  --db-path database/face_database \
  --output output_video.mp4 \
  --no-display \
  --update-db
```

## 📊 Model Performance Comparison

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **Detection Models** |
| det_500m.onnx | 2.41 MB | Fastest | Good | Real-time, mobile |
| det_2.5g.onnx | 3.14 MB | Fast | Better | Balanced performance |
| det_10g.onnx | 16.1 MB | Slower | Best | High accuracy needs |
| **Recognition Models** |
| w600k_mbf.onnx | 13 MB | Fast | Good | Real-time applications |
| w600k_r50.onnx | 166 MB | Slower | Best | High accuracy needs |

## 🎭 Adding Your Own Faces

1. Add face images to `assets/faces/` directory
2. Use clear, front-facing photos (jpg/png format)
3. Name files as `PersonName.jpg` (filename becomes the identity label)
4. Run with `--update-db` flag to rebuild the database:

```bash
python main.py --source 0 --update-db
```

## 🛠️ Troubleshooting

### Common Issues

**"ModuleNotFoundError":**
```bash
source face_reid_env/bin/activate
pip install -r requirements.txt
```

**"Model file not found":**
- Verify weights are downloaded: `ls -la weights/`
- Re-run download if needed: Check model URLs in original repository

**Poor recognition accuracy:**
- Use higher accuracy models: `det_10g.onnx` + `w600k_r50.onnx`
- Adjust similarity threshold: `--similarity-thresh 0.3`
- Add more reference images per person

**Slow performance:**
- Use lighter models: `det_500m.onnx` + `w600k_mbf.onnx`
- Reduce max faces: `--max-num 5`
- Lower video resolution

## 📈 Performance Tips

1. **For Real-time Applications:** Use det_500m.onnx + w600k_mbf.onnx
2. **For High Accuracy:** Use det_10g.onnx + w600k_r50.onnx  
3. **For Mobile/Edge:** Use smallest models with reduced frame rate
4. **For Crowded Scenes:** Increase --max-num parameter
5. **For Security Applications:** Lower similarity threshold (0.2-0.3)

## 🔧 Technical Details

- **SCRFD**: Sample and Computation Redistribution for Efficient Face Detection
- **ArcFace**: Additive Angular Margin Loss for Deep Face Recognition  
- **FAISS**: Facebook AI Similarity Search for fast vector similarity
- **ONNX Runtime**: Cross-platform ML inference acceleration

## 📝 License

Based on the original [face-reidentification](https://github.com/yakhyo/face-reidentification) repository.

---

**Ready to use!** 🎉 The system will identify faces from the pre-loaded Friends characters or your own added faces. 