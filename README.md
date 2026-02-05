# SuperFast Video Face Recognition (GPU/TensorRT)

A high-performance, industrial-grade video face recognition system utilizing **Python**, **InsightFace**, **TensorRT**, and **Multiprocessing**.

This project is designed for maximum throughput on NVIDIA GPUs. It uses **Shared Memory** to transfer frames between CPU loaders and GPU workers with zero-copy overhead and automatically compiles **TensorRT** engines to accelerate InsightFace inference.

## 🚀 Key Features

- **Multi-GPU / Multi-Process Architecture**: Scales across multiple GPU processes and CPU loaders to maximize hardware utilization.
- **TensorRT Optimization**: Automatically compiles and caches TensorRT engines (`.engine`) for significant inference speedup compared to standard ONNX Runtime.
- **Shared Memory Buffer**: Uses `multiprocessing.Array` as a ring buffer to feed raw frames to the GPU without serialization overhead.
- **Smart Video Slicing (FFmpeg)**: Automatically merges overlapping detection timestamps and cuts video clips containing the target person with configurable padding.
- **Robust Filtering**: Includes configurable confidence thresholds, box padding, and input validation to reduce false positives.

## 🛠 Prerequisites

- **OS**: Windows (tested) or Linux.
- **Hardware**: NVIDIA GPU with CUDA support (Compute Capability 6.0+ recommended).
- **Software**:
  - Python 3.13+
  - CUDA Toolkit (Recommended: 13.0+)
  - CuDNN
  - FFmpeg (Installed globally or placed in a local `ffmpeg/` folder)

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fast-face-rec-gpu.git
cd fast-face-rec-gpu
```

### 2. Install PyTorch (CUDA Version)
**Crucial:** Do not simply run `pip install torch`. You must install the version compatible with your CUDA driver. 
Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to generate the correct command.

*Example for CUDA 13.0:*
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3. Install ONNX Runtime GPU
```bash
pip install onnxruntime-gpu
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Setup FFmpeg
The script requires FFmpeg to extract video metadata and slice clips.
- **Option A (System):** Add `ffmpeg` and `ffprobe` to your system PATH.
- **Option B (Local):** Create a folder named `ffmpeg` inside the project root and place `ffmpeg.exe` and `ffprobe.exe` there.

## 📂 Project Structure

Before running the script, ensure your directory looks like this:

```text
/project-root
│── main.py                  # The main script
│── requirements.txt         # Dependencies
│── ffmpeg/                  # (Optional) Folder containing ffmpeg.exe
│
├── in_video/                # [INPUT] Put video files here (.mp4, .mkv, .avi)
│
├── samples_jpg/             # [INPUT] Put reference images here
│   ├── Elon_Musk.jpg        # Filename becomes the label
│   └── Robert_Downey_Jr.jpg
│
└── found_fragments_colored_/ # [OUTPUT] Results will appear here
```

## ⚙️ Configuration Guide

All settings are located at the top of `main.py`. Here is a detailed explanation of every parameter.

### 1. Hardware & Performance
These settings control how the script utilizes your CPU and GPU.

| Variable | Description | Recommended |
| :--- | :--- | :--- |
| `NUM_GPU_PROCESSES` | Number of independent Python processes accessing the GPU. | `1` or `2` (Depends on VRAM) |
| `GPU_WORKER_THREADS` | Number of concurrent threads inside *each* GPU process. | `2` to `4` |
| `NUM_CPU_LOADERS` | Number of CPU processes decoding video frames. | `4` to `8` |
| `BATCH_SIZE` | How many frames are sent to the GPU inference engine at once. Higher = faster but more VRAM. | `8` to `32` |
| `MAX_BUFFER_SLOTS` | Size of the Shared Memory Ring Buffer. Prevents loader stalling. | `512` to `2048` |
| `REC_CHUNK_SIZE` | Batch size specifically for the Recognition (Embedding) model. | `64` to `128` |

### 2. Detection & Recognition Logic
Settings that affect accuracy and which faces are detected.

```python
MODEL_PACK_NAME = 'buffalo_l'      # InsightFace model pack ('buffalo_l' is accurate, 'buffalo_s' is fast)
DET_SIZE = (640, 640)              # Input resolution for the detector. (640,640) is standard. (1440,1440) for 4K small faces.
CUSTOM_THRESHOLD = 0.49            # Cosine similarity threshold. Lower (e.g. 0.40) is stricter. Higher (0.60) allows more errors.
PRE_UPSCALE_FACTOR = 1.0           # 1.0 = Native resolution. >1.0 upscales image before detection (slow).
BOX_PADDING_PERCENTAGE = 0.3       # Expands the bounding box by 30% to capture full head/hair.
```

### 3. Video Processing & Slicing
Settings controlling how the video is read and how the results are saved.

```python
FRAME_INTERVAL = 1          # 1 = Process every frame. 2 = Process every 2nd frame (2x speedup).
CLIP_DURATION_BEFORE = 1.0  # Seconds of video to save BEFORE the face appears.
CLIP_DURATION_AFTER = 10.0  # Seconds of video to save AFTER the face disappears.
MERGE_GAP_TOLERANCE = 12.0  # If two detections are within 12s of each other, merge them into one long clip.
```

### 4. Advanced / Debugging
These settings are found inside the `settings` dictionary within the `run()` function.

| Key | Description |
| :--- | :--- |
| `det_prob_threshold` | (Default `0.25`) Minimum probability for a generic "face" to be considered a face. |
| `debug_det` | `True` enables saving debug images of raw detection (without recognition). |
| `debug_det_dir` | Folder to save debug artifacts. |
| `debug_rec` | `True` enables logging of raw embedding distances. |

## ▶️ Usage

1. Place your target videos in `in_video/`.
2. Place reference photos of people to find in `samples_jpg/`.
3. Run the script:

```bash
python main.py
```

### ⚠️ First Run Note (Warmup)
The first time you run the script, it may take **1-3 minutes** to start.
The console will show: `[*] WARMUP: Generating TensorRT engine...`
This is normal. The system is compiling the model for your specific GPU. This file is saved in `insightface_trt_cache/` and subsequent runs will be instant.

## 📤 Output

The results are saved in `found_fragments_colored_/`:

1. **Subfolders (by video name)**:
   - Contains `.jpg` snapshots of every detected frame with drawn bounding boxes and confidence scores.
2. **Merged Video Clips**:
   - `.mp4` files containing the specific segments where the person was found.
   - Naming convention: `[StartSec]s_[Name]_d[Distance].mp4`.

## 📄 License

MIT License. See [LICENSE](LICENSE) for more information.

