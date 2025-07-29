# ⚽ Soccer Player Detection System 🎯

A real-time soccer player detection system powered by **YOLOv11**, capable of detecting players in both **live video feeds** and **static images**.
(u need to train the best.pt by making your own dataset)
---

## 🚀 Features

- 🔍 Real-time player detection using webcam
- 🖼️ Single image detection with result export
- ⚙️ System benchmarking (CPU vs GPU)
- 💻 Automatic hardware (CPU/GPU) utilization
- 📈 Real-time performance metrics display
- 📸 Screenshot capture functionality

---

## 📦 Requirements

Install the following Python packages (automatically handled via `requirements.txt`):

```bash
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
```

---

## 🔧 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/K-B-R-S-W/soccer-player-detection.git
cd soccer-player-detection
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Add the trained model:**
   - Download the `best.pt` YOLOv8 model file
   - Place it in the root directory of the project

---

## 🖥️ Usage

Run the detection script:

```bash
python "Soccer Player Detection - Real-time.py"
```

---

## 🔍 Detection Modes

### Real-time Camera Detection
- Press `q` to quit
- Press `s` to capture a screenshot
- Press `i` to view system info and stats

### Single Image Detection
- Enter the image path when prompted
- Output saved with `_detected` suffix in the filename

### System Benchmark Test
- Evaluates and compares CPU vs GPU performance
- Displays FPS and detection metrics

---

## 🖥️ System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster detection)
- Webcam (for real-time detection)

---

## 📊 Performance

- **GPU Mode:** Real-time detection at ~30+ FPS (with CUDA)
- **CPU Mode:** Performance varies based on system specs
- **Detection Threshold:** Configurable confidence threshold in script

---

## 📁 File Structure

```
soccer-player-detection/
│
├── Soccer Player Detection - Real-time.py
├── requirements.txt
├── README.md
└── best.pt                # YOLOv11 trained model
```

---

## 🐞 Known Issues

- Camera initialization may fail if no webcam is connected
- GPU memory usage may gradually increase during long sessions

---

## 🤝 Contributing

We welcome contributions!

```bash
# Steps to contribute
1. Fork the repository
2. Create a new branch
3. Make your changes and commit
4. Push to the branch
5. Submit a pull request
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

## 📮 Support

**📧 Email:** [k.b.ravindusankalpaac@gmail.com](mailto:k.b.ravindusankalpaac@gmail.com)  
**🐞 Bug Reports:** [GitHub Issues](https://github.com/K-B-R-S-W/Soccer_Player_Detection_Using_YOLO_V11/issues)  
**📚 Documentation:** See the project [Wiki](https://github.com/K-B-R-S-W/Soccer_Player_Detection_Using_YOLO_V11/wiki)  
**💭 Discussions:** Join the [GitHub Discussions](https://github.com/K-B-R-S-W/Soccer_Player_Detection_Using_YOLO_V11/discussions)

---

## ⭐ Support This Project

If you find this project helpful, please consider giving it a **⭐ star** on GitHub!
