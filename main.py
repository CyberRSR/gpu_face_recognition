# --- recgn_arcnet_superfast_gpu_ct_up_Avid_v10_PYAV.py ---
# --- VERSION: PyAV DECODER + SINGLE-LOADER + FFMPEG THREADING ---

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import sys
import time
import subprocess
import traceback
import ctypes
import cv2
import gc
import numpy as np
import torch
import threading
from multiprocessing import Process, Queue, Array, set_start_method, Value
from concurrent.futures import ThreadPoolExecutor

# ============ PyAV Import ============
try:
    import av
    av.logging.set_level(av.logging.ERROR)
except ImportError:
    print("[!] ERROR: PyAV not found. Install via: pip install av")
    sys.exit(1)

try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError:
    print("[!] ERROR: insightface not found.")
    sys.exit(1)

# ==========================================
#               SETTINGS
# ==========================================
torch.backends.cudnn.benchmark = True

REFERENCE_FACES_FOLDER = 'samples_jpg'
TARGET_VIDEO_FOLDER = 'in_video'
OUTPUT_FOLDER = 'found_fragments_colored_'
TRT_CACHE_PATH = 'insightface_trt_cache'

MODEL_PACK_NAME = 'buffalo_l'

NUM_GPU_PROCESSES = 5
GPU_WORKER_THREADS = 2

# ===== KEY CHANGE =====
# Instead of 8 loaders with seek — 1-2 sequential decoders
NUM_VIDEO_DECODERS = 2  # 1-2 per video (more is not needed!)
FFMPEG_DECODER_THREADS = 4  # Threads inside FFmpeg

# Set True to use PyAV, False for OpenCV
USE_PYAV = True

BATCH_SIZE = 8
REC_CHUNK_SIZE = 16

MAX_BUFFER_SLOTS = 1024

DET_SIZE = (1440, 1440)

CUSTOM_THRESHOLD = 0.49
SEARCH_MODE = 'original_full_frame'
PRE_UPSCALE_FACTOR = 1.0
BOX_PADDING_PERCENTAGE = 0.3

FRAME_INTERVAL = 2
CLIP_DURATION_BEFORE = 1.0
CLIP_DURATION_AFTER = 10.0
MERGE_GAP_TOLERANCE = 12.0

COLOR_PALETTE = ['red', 'lime', 'cyan', 'magenta', 'orange', 'yellow', 'dodgerblue']

# ==========================================
#           SYSTEM FUNCTIONS
# ==========================================
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    FFMPEG_DIR = os.path.join(script_dir, "ffmpeg")
    FFMPEG_PATH = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
    FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe.exe")
    if not os.path.exists(FFMPEG_PATH):
        FFMPEG_PATH = "ffmpeg"
        FFPROBE_PATH = "ffprobe"
except:
    FFMPEG_PATH = "ffmpeg"
    FFPROBE_PATH = "ffprobe"


def find_recognition_model(app):
    models = getattr(app, 'models', None)
    if models is None:
        return None
    if isinstance(models, dict):
        iterable = models.values()
    else:
        iterable = models
    for m in iterable:
        if getattr(m, 'taskname', None) == 'recognition':
            return m
    return None


def make_providers():
    trt_options = {
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': TRT_CACHE_PATH,
        'trt_fp16_enable': True,
    }
    return [('TensorrtExecutionProvider', trt_options), 'CUDAExecutionProvider']


def warmup_trt_engine(device_id=0):
    print(f"[*] WARMUP: Generating TensorRT engine (Device {device_id})...")
    if not os.path.exists(TRT_CACHE_PATH):
        os.makedirs(TRT_CACHE_PATH)

    app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        allowed_modules=['detection', 'recognition'],
        providers=make_providers()
    )
    app.prepare(ctx_id=device_id, det_size=DET_SIZE)

    rec_model = find_recognition_model(app)
    if rec_model:
        rec_in = rec_model.session.get_inputs()[0].name
        rec_out = rec_model.session.get_outputs()[0].name
        print("[*] WARMUP: Warming up recognition model...")
        fake_blob = np.random.rand(REC_CHUNK_SIZE, 3, 112, 112).astype(np.float32)
        try:
            rec_model.session.run([rec_out], {rec_in: fake_blob})
        except Exception as e:
            print(f"[!] Warn Warmup: {e}")

    print("[*] WARMUP: Done.")
    del app
    gc.collect()
    time.sleep(1)


def get_color_for_name(name):
    if not name:
        return 'red'
    idx = sum(ord(c) for c in name) % len(COLOR_PALETTE)
    return COLOR_PALETTE[idx]


def get_video_duration_and_fps(video_path):
    try:
        cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path]
        fps_str = subprocess.check_output(cmd).decode().strip()
        if '/' in fps_str:
            num, den = map(float, fps_str.split('/'))
            fps = num / den if den != 0 else 25.0
        else:
            fps = float(fps_str)

        cmd_dur = [FFPROBE_PATH, '-v', 'error', '-show_entries',
                   'format=duration', '-of', 'csv=p=0', video_path]
        dur = float(subprocess.check_output(cmd_dur).decode().strip())
        return dur, fps
    except:
        return None, None

def get_video_info(video_path):
    """Get video info (PyAV with fallback to OpenCV)"""
    
    # Try PyAV
    if USE_PYAV:
        try:
            import av
            container = av.open(video_path)
            stream = container.streams.video[0]
            
            w = stream.width
            h = stream.height
            
            # FPS
            if stream.average_rate:
                fps = float(stream.average_rate)
            elif stream.guessed_rate:
                fps = float(stream.guessed_rate)
            else:
                fps = 25.0
            
            # Number of frames
            frames = stream.frames
            if frames == 0 or frames is None:
                # Estimate via duration
                if stream.duration and stream.time_base:
                    duration_sec = float(stream.duration * stream.time_base)
                    frames = int(duration_sec * fps)
                elif container.duration:
                    duration_sec = container.duration / av.time_base
                    frames = int(duration_sec * fps)
                else:
                    # Fallback: count frames (slow)
                    frames = 0
                    for _ in container.decode(stream):
                        frames += 1
                    container.close()
                    container = av.open(video_path)
            
            container.close()
            
            print(f"[PyAV Info] {video_path}: {w}x{h}, {frames} frames, {fps:.2f} fps")
            return {'frames': frames, 'w': w, 'h': h, 'fps': fps, 'path': video_path}
            
        except Exception as e:
            print(f"[PyAV Info] Failed: {e}, trying OpenCV...")
    
    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        cap.release()
        
        print(f"[CV2 Info] {video_path}: {w}x{h}, {frames} frames, {fps:.2f} fps")
        return {'frames': frames, 'w': w, 'h': h, 'fps': fps, 'path': video_path}
        
    except Exception as e:
        print(f"[Video Info] Error: {e}")
        return None
        

def save_merged_clip(input_path, output_path, start_time, end_time, detections_in_clip, fps):
    try:
        duration = end_time - start_time
        start_time_str = time.strftime('%H:%M:%S', time.gmtime(start_time)) + f'.{int((start_time % 1) * 1000):03d}'

        unique_boxes = []
        last_coords_map = {}
        sorted_dets = sorted(detections_in_clip, key=lambda x: x['time'])
        for det in sorted_dets:
            name = det['name']
            x1, y1, x2, y2 = det['coords']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if name in last_coords_map:
                lx, ly = last_coords_map[name]
                dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                if dist < 5:
                    continue
            last_coords_map[name] = (cx, cy)
            unique_boxes.append(det)

        filters = []
        for i, det in enumerate(unique_boxes):
            if i > 250:
                break
            x1, y1, x2, y2 = det['coords']
            w, h = x2 - x1, y2 - y1
            color = get_color_for_name(det['name'])
            filters.append(f"drawbox=x={x1}:y={y1}:w={w}:h={h}:color={color}:t=2")

        cmd = [FFMPEG_PATH, '-y', '-ss', start_time_str, '-i', input_path, '-t', str(duration), '-loglevel', 'error']
        if filters:
            vf = ",".join(filters)
            cmd.extend(['-vf', vf, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23', '-c:a', 'copy'])
        else:
            cmd.extend(['-c', 'copy'])
        cmd.append(output_path)

        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        return True, None
    except subprocess.TimeoutExpired:
        return False, "FFMPEG Timeout"
    except Exception as e:
        return False, str(e)


# ==========================================
#           LOADING REFERENCES
# ==========================================
def prepare_reference_embeddings(folder_path, device_id):
    if not os.path.exists(folder_path):
        return []
    print(f"[*] Loading references (Device: {device_id})...")

    app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        allowed_modules=['detection', 'recognition'],
        providers=make_providers()
    )
    app.prepare(ctx_id=device_id, det_size=(640, 640))

    rec_model = find_recognition_model(app)
    if not rec_model:
        sys.exit("[!] Rec model not found")

    rec_in = rec_model.session.get_inputs()[0].name
    rec_out = rec_model.session.get_outputs()[0].name

    mean = float(getattr(rec_model, 'input_mean', 127.5))
    std = float(getattr(rec_model, 'input_std', 127.5))

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
             if f.lower().endswith(('.jpg', '.png'))]
    refs = []
    loaded = 0
    
    for p in files:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            bboxes, kpss = app.det_model.detect(img, max_num=1)
            if not len(bboxes):
                continue
            aimg = face_align.norm_crop(img, landmark=kpss[0])
            blob = cv2.dnn.blobFromImages([aimg], 1.0 / std, (112, 112), (mean, mean, mean), swapRB=True)
            feat = rec_model.session.run([rec_out], {rec_in: blob})[0][0]
            emb = feat / (np.linalg.norm(feat) + 1e-12)
            refs.append({'name': os.path.basename(p), 'embedding': emb})
            loaded += 1
        except:
            pass

    del app
    gc.collect()
    print(f"[*] References loaded: {loaded}")
    return refs


# ==========================================
#     OPTIMIZED LOADER ON PyAV
# ==========================================
def frame_loader_pyav(video_path, free_q, filled_q, shared_buf, max_frame_size, 
                      shape, start_f, end_f, settings, submitted_counter, loader_id=0):
    """
    Fixed PyAV decoder
    """
    import av
    av.logging.set_level(av.logging.ERROR)
    
    try:
        h, w, c = shape
        raw_arr = np.frombuffer(shared_buf, dtype=np.uint8)
        interval = settings['frame_interval']
        frame_len = h * w * c

        print(f"[Loader-{loader_id}] Starting: frames {start_f}-{end_f}, interval={interval}")

        # Open container
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # Multi-threaded decoding
        stream.thread_type = 'AUTO'  # AUTO is safer than FRAME
        stream.thread_count = 0  # auto-detect
        
        # Get FPS
        if stream.average_rate:
            fps = float(stream.average_rate)
        elif stream.guessed_rate:
            fps = float(stream.guessed_rate)
        else:
            fps = 25.0
            
        print(f"[Loader-{loader_id}] Video FPS: {fps:.2f}, Resolution: {stream.width}x{stream.height}")

        # Seek to start position
        if start_f > 0:
            target_pts = int(start_f / fps / stream.time_base)
            try:
                container.seek(target_pts, stream=stream, backward=True, any_frame=False)
                print(f"[Loader-{loader_id}] Seeked to frame ~{start_f}")
            except Exception as e:
                print(f"[Loader-{loader_id}] Seek failed: {e}, starting from beginning")
                container.close()
                container = av.open(video_path)
                stream = container.streams.video[0]
                stream.thread_type = 'AUTO'

        batch_indices = []
        batch_f_nums = []
        current_frame = 0
        frames_sent = 0
        
        print(f"[Loader-{loader_id}] Starting decode loop...")

        for packet in container.demux(stream):
            for frame in packet.decode():
                # Calculate frame number
                if frame.pts is not None and stream.time_base:
                    time_sec = float(frame.pts * stream.time_base)
                    current_frame = int(time_sec * fps + 0.5)
                else:
                    current_frame = frame.index if hasattr(frame, 'index') else frames_sent

                # Skip to start_f
                if current_frame < start_f:
                    continue
                
                # Exit after end_f
                if current_frame >= end_f:
                    print(f"[Loader-{loader_id}] Reached end_f={end_f}")
                    break

                # Check interval
                if current_frame % interval != 0:
                    continue

                # Get slot
                try:
                    idx = free_q.get(timeout=30)
                except:
                    print(f"[Loader-{loader_id}] Timeout waiting for free slot")
                    break
                    
                if idx is None:
                    print(f"[Loader-{loader_id}] Got None slot, stopping")
                    break

                # Convert to BGR
                try:
                    img = frame.to_ndarray(format='bgr24')
                except Exception as e:
                    print(f"[Loader-{loader_id}] Frame convert error: {e}")
                    free_q.put(idx)
                    continue
                
                # Resize if needed
                if img.shape[0] != h or img.shape[1] != w:
                    img = cv2.resize(img, (w, h))

                # Copy to shared memory
                offset = idx * max_frame_size
                try:
                    raw_arr[offset: offset + frame_len] = img.ravel()
                except Exception as e:
                    print(f"[Loader-{loader_id}] Memory copy error: {e}")
                    free_q.put(idx)
                    continue
                
                batch_indices.append(idx)
                batch_f_nums.append(current_frame)
                frames_sent += 1

                # Send batch
                if len(batch_indices) >= BATCH_SIZE:
                    filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape))
                    with submitted_counter.get_lock():
                        submitted_counter.value += len(batch_indices)
                    batch_indices.clear()
                    batch_f_nums.clear()
                    
            # Check exit from outer loop
            if current_frame >= end_f:
                break

        # Send remainder
        if batch_indices:
            filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape))
            with submitted_counter.get_lock():
                submitted_counter.value += len(batch_indices)

        container.close()
        print(f"[Loader-{loader_id}] Finished. Sent {frames_sent} frames.")

    except Exception as e:
        print(f"[Loader-{loader_id}] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
#     ALTERNATIVE: Optimized OpenCV
# ==========================================
def frame_loader_cv2_optimized(video_path, free_q, filled_q, shared_buf, max_frame_size,
                               shape, start_f, end_f, settings, submitted_counter, loader_id=0):
    """
    Optimized OpenCV loader (fallback if PyAV fails)
    """
    try:
        h, w, c = shape
        raw_arr = np.frombuffer(shared_buf, dtype=np.uint8)
        interval = settings['frame_interval']
        frame_len = h * w * c

        print(f"[CV2-Loader-{loader_id}] Starting: frames {start_f}-{end_f}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[CV2-Loader-{loader_id}] Failed to open video!")
            return
            
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Seek to start position
        if start_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        batch_indices = []
        batch_f_nums = []
        current_frame = start_f
        frames_sent = 0

        print(f"[CV2-Loader-{loader_id}] Starting decode loop...")

        while current_frame < end_f:
            ret, frame = cap.read()
            if not ret:
                print(f"[CV2-Loader-{loader_id}] End of video at frame {current_frame}")
                break

            if current_frame % interval == 0:
                try:
                    idx = free_q.get(timeout=30)
                except:
                    print(f"[CV2-Loader-{loader_id}] Timeout waiting for slot")
                    break
                    
                if idx is None:
                    break

                offset = idx * max_frame_size
                raw_arr[offset: offset + frame_len] = frame.ravel()
                
                batch_indices.append(idx)
                batch_f_nums.append(current_frame)
                frames_sent += 1

                if len(batch_indices) >= BATCH_SIZE:
                    filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape))
                    with submitted_counter.get_lock():
                        submitted_counter.value += len(batch_indices)
                    batch_indices.clear()
                    batch_f_nums.clear()

            current_frame += 1

        if batch_indices:
            filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape))
            with submitted_counter.get_lock():
                submitted_counter.value += len(batch_indices)

        cap.release()
        print(f"[CV2-Loader-{loader_id}] Finished. Sent {frames_sent} frames.")

    except Exception as e:
        print(f"[CV2-Loader-{loader_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
#    ALTERNATIVE: FFmpeg Pipe (faster for some codecs)
# ==========================================
def frame_loader_ffmpeg_pipe(video_path, free_q, filled_q, shared_buf, max_frame_size,
                             shape, start_f, end_f, settings, submitted_counter, loader_id=0):
    """
    Decoder via FFmpeg subprocess with:
    - select filter for frame skipping (FRAME_INTERVAL)
    - Multi-threaded decoding
    - Direct raw BGR output
    
    ADVANTAGE: FFmpeg can optimize B-frame skipping
    """
    try:
        h, w, c = shape
        raw_arr = np.frombuffer(shared_buf, dtype=np.uint8)
        interval = settings['frame_interval']
        frame_len = h * w * c
        fps = settings.get('fps', 25.0)

        # Start time
        start_time = start_f / fps
        frames_to_decode = (end_f - start_f) // interval

        # ===== FFmpeg command with frame selection filter =====
        cmd = [
            FFMPEG_PATH,
            '-threads', str(FFMPEG_DECODER_THREADS),
            '-ss', str(start_time),
            '-i', video_path,
            '-vf', f"select='not(mod(n\\,{interval}))',setpts=N/FRAME_RATE/TB",
            '-frames:v', str(frames_to_decode),
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-vsync', 'vfr',
            '-loglevel', 'error',
            'pipe:1'
        ]

        proc = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL,
            bufsize=frame_len * 4  # Buffer for several frames
        )

        batch_indices = []
        batch_f_nums = []
        frame_count = start_f

        while frame_count < end_f:
            # Read exactly one frame
            raw_frame = proc.stdout.read(frame_len)
            if len(raw_frame) != frame_len:
                break

            idx = free_q.get()
            if idx is None:
                proc.terminate()
                break

            offset = idx * max_frame_size
            raw_arr[offset: offset + frame_len] = np.frombuffer(raw_frame, dtype=np.uint8)

            batch_indices.append(idx)
            batch_f_nums.append(frame_count)

            if len(batch_indices) >= BATCH_SIZE:
                filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape))
                with submitted_counter.get_lock():
                    submitted_counter.value += len(batch_indices)
                batch_indices.clear()
                batch_f_nums.clear()

            frame_count += interval

        if batch_indices:
            filled_q.put((video_path, list(batch_f_nums), list(batch_indices), shape))
            with submitted_counter.get_lock():
                submitted_counter.value += len(batch_indices)

        proc.stdout.close()
        proc.terminate()
        proc.wait(timeout=5)

    except Exception as e:
        print(f"[FFmpeg-Loader-{loader_id} Error] {e}")


# ==========================================
#         DECODER SELECTION
# ==========================================

def frame_loader_optimal(video_path, free_q, filled_q, shared_buf, max_frame_size,
                         shape, start_f, end_f, settings, submitted_counter, loader_id=0):
    """Wrapper with automatic fallback"""
    
    if USE_PYAV:
        try:
            import av
            frame_loader_pyav(
                video_path, free_q, filled_q, shared_buf, max_frame_size,
                shape, start_f, end_f, settings, submitted_counter, loader_id
            )
        except ImportError:
            print(f"[Loader-{loader_id}] PyAV not available, using OpenCV")
            frame_loader_cv2_optimized(
                video_path, free_q, filled_q, shared_buf, max_frame_size,
                shape, start_f, end_f, settings, submitted_counter, loader_id
            )
    else:
        frame_loader_cv2_optimized(
            video_path, free_q, filled_q, shared_buf, max_frame_size,
            shape, start_f, end_f, settings, submitted_counter, loader_id
        )


# ==========================================
#      GPU PREPROCESS (unchanged)
# ==========================================
def gpu_preprocess_scrfd(batch_frames_uint8, target_size,
                         input_mean=127.5, input_std=128.0,
                         swap_rb=True):
    import torch
    import torch.nn.functional as F

    assert batch_frames_uint8.is_cuda
    assert batch_frames_uint8.dtype == torch.uint8

    B, H0, W0, C = batch_frames_uint8.shape
    det_h, det_w = int(target_size[0]), int(target_size[1])

    x = batch_frames_uint8
    if swap_rb:
        x = x[..., [2, 1, 0]]

    x = x.permute(0, 3, 1, 2).contiguous().float()

    scale = min(det_w / float(W0), det_h / float(H0))
    new_w = max(1, int(round(W0 * scale)))
    new_h = max(1, int(round(H0 * scale)))

    resized = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

    canvas = torch.zeros((B, 3, det_h, det_w), device=resized.device, dtype=torch.float32)
    canvas[:, :, :new_h, :new_w] = resized

    canvas = (canvas - float(input_mean)) / float(input_std)

    prep = {
        "det_scale": float(scale),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "orig_w": int(W0),
        "orig_h": int(H0),
        "det_w": int(det_w),
        "det_h": int(det_h),
    }
    return canvas.contiguous(), prep


# ==========================================
#      THREADED WORKER LOGIC (unchanged, condensed)
# ==========================================
def thread_infer_task(det_model, rec_model, ref_matrix, filled_q, free_q, save_q, stats_q,
                      raw_arr, max_frame_size, settings, ref_names):
    # ... (keep as is - this is the GPU part, not the decoder)
    # Code identical to original
    import warnings
    warnings.filterwarnings("ignore")
    cv2.setNumThreads(0)    

    det_sess = det_model.session
    rec_sess = rec_model.session

    det_in_name = det_sess.get_inputs()[0].name
    det_out_names = [o.name for o in det_sess.get_outputs()]

    rec_in_name = rec_sess.get_inputs()[0].name
    rec_out_name = rec_sess.get_outputs()[0].name

    pad_pct = float(settings.get('box_padding_percentage', 0.0))
    threshold = float(settings['threshold'])
    det_shape = settings.get('det_size', (640, 640))

    prob_threshold = float(settings.get('det_prob_threshold', 0.4))
    nms_threshold = float(settings.get('det_nms_threshold', 0.4))
    rec_chunk = int(settings.get('rec_chunk_size', 64))

    det_mean = float(getattr(det_model, 'input_mean', 127.5))
    det_std = float(getattr(det_model, 'input_std', 128.0))
    det_swap = bool(getattr(det_model, 'swapRB', True))

    rec_mean = float(getattr(rec_model, 'input_mean', 127.5))
    rec_std = float(getattr(rec_model, 'input_std', 127.5))

    stride_map = {
        8: {"score": 0, "bbox": 3, "kps": 6},
        16: {"score": 1, "bbox": 4, "kps": 7},
        32: {"score": 2, "bbox": 5, "kps": 8},
    }

    input_height, input_width = int(det_shape[0]), int(det_shape[1])
    _num_anchors = 2

    center_cache = {}
    for stride in (8, 16, 32):
        feat_h = input_height // stride
        feat_w = input_width // stride
        xs = np.arange(feat_w, dtype=np.float32)
        ys = np.arange(feat_h, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        centers = centers * stride
        centers = np.repeat(centers, _num_anchors, axis=0)
        center_cache[stride] = centers

    full_mem_view = memoryview(raw_arr)
    stream = torch.cuda.Stream()
    device_id = torch.cuda.current_device()

    batch_det_supported = True

    while True:
        task = filled_q.get()
        if task is None:
            filled_q.put(None)
            break

        t0 = time.perf_counter()
        video_path, f_nums, indices, shape = task
        h, w, c = shape
        batch_sz = len(indices)
        frame_len = h * w * c

        all_crops = []
        meta = []

        with torch.cuda.stream(stream):
            t_f = time.perf_counter()
            batch_np = np.empty((batch_sz, h, w, c), dtype=np.uint8)
            for k, idx in enumerate(indices):
                off = idx * max_frame_size
                batch_np[k] = np.frombuffer(full_mem_view[off: off + frame_len], dtype=np.uint8).reshape(h, w, c)
            gpu_frames = torch.from_numpy(batch_np).to('cuda', non_blocking=True)
            t_fetch = time.perf_counter() - t_f

            t_i = time.perf_counter()
            det_input_batch, prep = gpu_preprocess_scrfd(
                gpu_frames, det_shape, input_mean=det_mean, input_std=det_std, swap_rb=det_swap
            )
            stream.synchronize()

            det_scale = float(prep["det_scale"])
            valid_w = int(prep["new_w"])
            valid_h = int(prep["new_h"])

            # Detection (simplified - see original for full code)
            for i in range(batch_sz):
                det_binding = det_sess.io_binding()
                det_in = det_input_batch[i:i + 1].contiguous()

                det_binding.bind_input(
                    name=det_in_name,
                    device_type='cuda',
                    device_id=device_id,
                    element_type=np.float32,
                    shape=tuple(det_in.shape),
                    buffer_ptr=det_in.data_ptr()
                )
                for out_name in det_out_names:
                    det_binding.bind_output(out_name, 'cpu')

                det_sess.run_with_iobinding(det_binding)
                det_outs = det_binding.copy_outputs_to_cpu()

                frame_preds = []
                frame_kpss = []

                for stride in (8, 16, 32):
                    feat_h = input_height // stride
                    feat_w = input_width // stride
                    N = feat_h * feat_w * _num_anchors
                    centers = center_cache[stride]

                    scores = det_outs[stride_map[stride]["score"]].reshape(-1).astype(np.float32)
                    bbox_preds = det_outs[stride_map[stride]["bbox"]].reshape(-1, 4).astype(np.float32)
                    kps_preds = det_outs[stride_map[stride]["kps"]].reshape(-1, 10).astype(np.float32)

                    idx_keep = np.where(scores >= prob_threshold)[0]
                    if idx_keep.size == 0:
                        continue
                    if idx_keep.max() >= centers.shape[0]:
                        continue

                    ac = centers[idx_keep]
                    sc = scores[idx_keep]
                    bb = bbox_preds[idx_keep]
                    kp = kps_preds[idx_keep]

                    m = (ac[:, 0] < valid_w) & (ac[:, 1] < valid_h)
                    if not np.any(m):
                        continue
                    ac, sc, bb, kp = ac[m], sc[m], bb[m], kp[m]

                    x1 = (ac[:, 0] - bb[:, 0] * stride) / det_scale
                    y1 = (ac[:, 1] - bb[:, 1] * stride) / det_scale
                    x2 = (ac[:, 0] + bb[:, 2] * stride) / det_scale
                    y2 = (ac[:, 1] + bb[:, 3] * stride) / det_scale
                    boxes = np.stack([x1, y1, x2, y2, sc], axis=-1)

                    kpss = np.zeros((kp.shape[0], 10), dtype=np.float32)
                    for kk in range(5):
                        kpss[:, kk * 2] = (ac[:, 0] + kp[:, kk * 2] * stride) / det_scale
                        kpss[:, kk * 2 + 1] = (ac[:, 1] + kp[:, kk * 2 + 1] * stride) / det_scale

                    frame_preds.append(boxes)
                    frame_kpss.append(kpss)

                if not frame_preds:
                    continue

                frame_preds = np.concatenate(frame_preds, axis=0)
                frame_kpss = np.concatenate(frame_kpss, axis=0)

                nms_boxes = frame_preds[:, :4].copy()
                nms_boxes[:, 2] -= nms_boxes[:, 0]
                nms_boxes[:, 3] -= nms_boxes[:, 1]
                keep = cv2.dnn.NMSBoxes(
                    nms_boxes.tolist(), frame_preds[:, 4].tolist(),
                    prob_threshold, nms_threshold
                )
                if len(keep) == 0:
                    continue
                keep = np.array(keep).flatten()
                det_boxes = frame_preds[keep]
                det_kpss = frame_kpss[keep]

                np.clip(det_boxes[:, 0], 0, w - 1, out=det_boxes[:, 0])
                np.clip(det_boxes[:, 1], 0, h - 1, out=det_boxes[:, 1])
                np.clip(det_boxes[:, 2], 0, w - 1, out=det_boxes[:, 2])
                np.clip(det_boxes[:, 3], 0, h - 1, out=det_boxes[:, 3])

                orig_img = batch_np[i]
                for box, kps in zip(det_boxes, det_kpss):
                    kps = kps.reshape(5, 2)
                    try:
                        aimg = face_align.norm_crop(orig_img, landmark=kps)
                        all_crops.append(aimg)

                        b = box.astype(np.int32)
                        x1i, y1i, x2i, y2i = b[:4]
                        bw, bh = x2i - x1i, y2i - y1i
                        px, py = int(bw * pad_pct), int(bh * pad_pct)
                        nx1 = max(0, x1i - px)
                        ny1 = max(0, y1i - py)
                        nx2 = min(w, x2i + px)
                        ny2 = min(h, y2i + py)
                        meta.append((i, (nx1, ny1, nx2, ny2), ((nx1 + nx2) >> 1, (ny1 + ny2) >> 1), ""))
                    except:
                        pass

            faces_found = 0
            t_match_sum = 0.0
            res_map = [[] for _ in range(batch_sz)]

            if all_crops:
                t_m = time.perf_counter()
                for start in range(0, len(all_crops), rec_chunk):
                    chunk_crops = all_crops[start:start + rec_chunk]
                    chunk_meta = meta[start:start + rec_chunk]

                    blob = cv2.dnn.blobFromImages(
                        chunk_crops, 1.0 / rec_std, (112, 112),
                        (rec_mean, rec_mean, rec_mean), swapRB=True
                    )
                    feats = rec_sess.run([rec_out_name], {rec_in_name: blob})[0]
                    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

                    sims = np.dot(feats, ref_matrix.T)
                    best_indices = np.argmax(sims, axis=1)
                    best_scores = sims[np.arange(sims.shape[0]), best_indices]

                    for k, score in enumerate(best_scores):
                        dist = 1.0 - float(score)
                        if dist <= threshold:
                            fi, co, ce, _ = chunk_meta[k]

                            dup = False
                            for ex in res_map[fi]:
                                if abs(ce[0] - ex['center'][0]) < 20 and abs(ce[1] - ex['center'][1]) < 20:
                                    dup = True
                                    break
                            if not dup:
                                faces_found += 1
                                res_map[fi].append({
                                    'ref_idx': int(best_indices[k]),
                                    'dist': dist,
                                    'coords': co,
                                    'center': ce,
                                    'tag': ""
                                })

                t_match_sum = time.perf_counter() - t_m

            t_infer = time.perf_counter() - t_i

            if faces_found > 0:
                for i in range(batch_sz):
                    if res_map[i]:
                        payload = [{
                            'name': ref_names[x['ref_idx']],
                            'dist': x['dist'],
                            'coords': x['coords'],
                            'tag': ""
                        } for x in res_map[i]]
                        save_q.put(('data', video_path, f_nums[i], batch_np[i].copy(), payload))

        for idx in indices:
            free_q.put(idx)

        stats_q.put(('metrics', (t_fetch, 0.0, t_infer, t_match_sum, time.perf_counter() - t0, faces_found), batch_sz))


def gpu_manager_process(device_str, filled_q, free_q, save_q, stats_q,
                        shared_buf, max_frame_size, ref_embs, settings, num_threads, proc_rank):
    try:
        gpu_id = int(device_str.split(':')[-1])
    except:
        gpu_id = 0

    cv2.setNumThreads(0)
    torch.cuda.set_device(gpu_id)

    print(f"[GPU-MANAGER-{proc_rank}] Init on {device_str} with {num_threads} THREADS...")

    app = FaceAnalysis(
        name=MODEL_PACK_NAME,
        allowed_modules=['detection', 'recognition'],
        providers=make_providers()
    )
    app.prepare(ctx_id=gpu_id, det_size=DET_SIZE)

    det_model = app.det_model
    rec_model = find_recognition_model(app)

    ref_matrix = np.array([r['embedding'] for r in ref_embs], dtype=np.float32)
    ref_names = [r['name'] for r in ref_embs]
    raw_arr = np.frombuffer(shared_buf, dtype=np.uint8)

    print(f"[GPU-MANAGER-{proc_rank}] Ready.")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for _ in range(num_threads):
            futures.append(
                executor.submit(
                    thread_infer_task,
                    det_model, rec_model, ref_matrix,
                    filled_q, free_q, save_q, stats_q,
                    raw_arr, max_frame_size, settings, ref_names
                )
            )
        for f in futures:
            f.result()

    print(f"[GPU-MANAGER-{proc_rank}] Done.")
    del app


# --- SAVER (unchanged) ---
def result_saver(save_q, out_dir, settings):
    print(f"[Saver] Started.")
    os.makedirs(out_dir, exist_ok=True)

    video_meta_cache = {}
    detection_buffer = {}

    while True:
        try:
            task = save_q.get()
            if task is None:
                break

            msg_type = task[0]

            if msg_type == 'data':
                _, video_path, f, img, faces_payload = task

                if video_path not in video_meta_cache:
                    dur, fps = get_video_duration_and_fps(video_path)
                    video_meta_cache[video_path] = {'fps': fps or 25.0, 'dur': dur or 99999}
                    detection_buffer[video_path] = []
                    os.makedirs(os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0], 'img_jpg'),
                                exist_ok=True)

                sec = f / video_meta_cache[video_path]['fps']
                for face in faces_payload:
                    clean_name = os.path.splitext(face['name'])[0]
                    detection_buffer[video_path].append(
                        {'time': sec, 'name': clean_name, 'dist': face['dist'], 'coords': face['coords']}
                    )

                    try:
                        img_copy = img.copy()
                        x1, y1, x2, y2 = face['coords']
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{clean_name} {face['dist']:.2f}"
                        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        fname = f"f{f}_{clean_name}_{int(time.time() * 100) % 1000}.jpg"
                        save_path = os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0], 'img_jpg', fname)
                        cv2.imwrite(save_path, img_copy)
                    except Exception as e:
                        print(f"[Saver Error] ImgSave: {e}")

            elif msg_type == 'end_video':
                _, video_path = task
                if video_path in detection_buffer and detection_buffer[video_path]:
                    dets = sorted(detection_buffer[video_path], key=lambda x: x['time'])
                    meta = video_meta_cache.get(video_path)

                    merged = []
                    if dets:
                        cur_d = [dets[0]]
                        cur_s = max(0, dets[0]['time'] - CLIP_DURATION_BEFORE)
                        cur_e = min(meta['dur'], dets[0]['time'] + CLIP_DURATION_AFTER)

                        for ii in range(1, len(dets)):
                            d = dets[ii]
                            ds = max(0, d['time'] - CLIP_DURATION_BEFORE)
                            de = min(meta['dur'], d['time'] + CLIP_DURATION_AFTER)

                            if ds - cur_e < MERGE_GAP_TOLERANCE:
                                cur_e = max(cur_e, de)
                                cur_d.append(d)
                            else:
                                merged.append((cur_s, cur_e, cur_d))
                                cur_s, cur_e, cur_d = ds, de, [d]

                        merged.append((cur_s, cur_e, cur_d))

                    sub = os.path.join(out_dir, os.path.splitext(os.path.basename(video_path))[0])
                    os.makedirs(sub, exist_ok=True)

                    for s, e, d_list in merged:
                        min_dist = min(x['dist'] for x in d_list)
                        names = sorted(list(set(x['name'] for x in d_list)))
                        names_str = '_'.join(names)[:50]
                        vname = f"{int(s)}s_{names_str}_d{min_dist:.2f}.mp4"
                        save_merged_clip(video_path, os.path.join(sub, vname), s, e, d_list, meta['fps'])

                if video_path in detection_buffer:
                    del detection_buffer[video_path]

        except Exception as e:
            print(f"[!] Saver Error: {e}")
            traceback.print_exc()


# --- MAIN ---
def run():
    print("=" * 60)
    print("  FACE RECOGNITION - FIXED DECODER")
    print("=" * 60)
    t_global_start = time.time()

    if not torch.cuda.is_available():
        print("[!] CUDA is not available.")
        return

    os.makedirs(TRT_CACHE_PATH, exist_ok=True)
    warmup_trt_engine(0)

    refs = prepare_reference_embeddings(REFERENCE_FACES_FOLDER, 0)
    if not refs:
        print("[!] No references found.")
        return

    files = [os.path.join(TARGET_VIDEO_FOLDER, f) for f in os.listdir(TARGET_VIDEO_FOLDER)
             if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    if not files:
        print("[!] No videos found.")
        return

    video_infos = []
    max_h, max_w = 0, 0

    for v in files:
        info = get_video_info(v)  # Fixed function
        if info:
            if info['frames'] == 0:
                print(f"[!] Skip {v}: no frames detected")
                continue
            process_cnt = (info['frames'] + FRAME_INTERVAL - 1) // FRAME_INTERVAL
            info['proc_frames'] = process_cnt
            video_infos.append(info)
            if info['w'] * info['h'] > max_w * max_h:
                max_w, max_h = info['w'], info['h']

    if not video_infos:
        print("[!] No valid videos found")
        return

    max_frame_size = max_h * max_w * 3
    num_buf = MAX_BUFFER_SLOTS

    print(f"[*] Max resolution: {max_w}x{max_h}")
    print(f"[*] Shared Memory: {num_buf} slots × {max_frame_size / 1024 / 1024:.1f} MB")

    try:
        shared = Array(ctypes.c_uint8, num_buf * max_frame_size, lock=False)
    except Exception as e:
        print(f"[!] RAM Error: {e}")
        return

    free_q = Queue()
    filled_q = Queue()
    save_q = Queue()
    stats_q = Queue()

    for i in range(num_buf):
        free_q.put(i)

    settings = {
        'frame_interval': FRAME_INTERVAL,
        'clip_before': CLIP_DURATION_BEFORE,
        'clip_after': CLIP_DURATION_AFTER,
        'threshold': CUSTOM_THRESHOLD,
        'search_mode': SEARCH_MODE,
        'pre_upscale_factor': PRE_UPSCALE_FACTOR,
        'box_padding_percentage': BOX_PADDING_PERCENTAGE,
        'rec_chunk_size': REC_CHUNK_SIZE,
        'det_prob_threshold': 0.25,
        'det_nms_threshold': 0.4,
        'det_size': DET_SIZE,
    }

    saver = Process(target=result_saver, args=(save_q, OUTPUT_FOLDER, settings))
    saver.start()

    print(f"[*] Starting {NUM_GPU_PROCESSES} GPU processes × {GPU_WORKER_THREADS} threads...")
    gpu_processes = []
    for i in range(NUM_GPU_PROCESSES):
        p = Process(
            target=gpu_manager_process,
            args=(f'cuda:0', filled_q, free_q, save_q, stats_q,
                  shared, max_frame_size, refs, settings, GPU_WORKER_THREADS, i)
        )
        p.start()
        gpu_processes.append(p)

    time.sleep(3)
    
    print(f"[*] GPU processes started, checking status...")
    for i, p in enumerate(gpu_processes):
        print(f"    GPU-{i}: alive={p.is_alive()}")

    for info in video_infos:
        vid_path = info['path']
        total_frames_to_proc = info['proc_frames']
        
        print(f"\n{'=' * 60}")
        print(f">> VIDEO: {os.path.basename(vid_path)}")
        print(f">> Total: {info['frames']} frames | Process: {total_frames_to_proc}")
        print(f">> Resolution: {info['w']}x{info['h']} @ {info['fps']:.2f} fps")
        print(f"{'=' * 60}")

        v_start_t = time.time()
        submitted = Value('i', 0)

        # One loader per video (most efficient for sequential reading)
        loaders = []
        
        print(f"[*] Starting {NUM_VIDEO_DECODERS} decoder(s)...")
        
        if NUM_VIDEO_DECODERS == 1:
            l = Process(
                target=frame_loader_optimal,
                args=(vid_path, free_q, filled_q, shared, max_frame_size,
                      (info['h'], info['w'], 3), 0, info['frames'], 
                      {**settings, 'fps': info['fps']}, submitted, 0)
            )
            loaders.append(l)
        else:
            chk = info['frames'] // NUM_VIDEO_DECODERS
            for i in range(NUM_VIDEO_DECODERS):
                s = i * chk
                e = (i + 1) * chk if i != NUM_VIDEO_DECODERS - 1 else info['frames']
                l = Process(
                    target=frame_loader_optimal,
                    args=(vid_path, free_q, filled_q, shared, max_frame_size,
                          (info['h'], info['w'], 3), s, e, 
                          {**settings, 'fps': info['fps']}, submitted, i)
                )
                loaders.append(l)

        for l in loaders:
            l.start()
            
        time.sleep(0.5)
        print(f"[*] Loaders started: {[l.is_alive() for l in loaders]}")

        processed_count = 0
        total_faces_video = 0
        last_log_time = time.time()
        last_submitted = 0
        stall_count = 0

        while True:
            # Gather statistics
            while not stats_q.empty():
                try:
                    msg = stats_q.get_nowait()
                    if msg[0] == 'metrics':
                        batch_sz = msg[2]
                        processed_count += batch_sz
                        total_faces_video += msg[1][5]
                except:
                    break

            with submitted.get_lock():
                submitted_val = submitted.value

            active_loaders = any(l.is_alive() for l in loaders)

            now = time.time()
            if now - last_log_time > 1.0:
                elapsed = now - v_start_t
                fps = processed_count / elapsed if elapsed > 0.1 else 0.0
                pct = (processed_count / total_frames_to_proc) * 100 if total_frames_to_proc > 0 else 0

                try:
                    q_in_sz = filled_q.qsize()
                    q_out_sz = free_q.qsize()
                except:
                    q_in_sz = '?'
                    q_out_sz = '?'

                loader_status = 'ALIVE' if active_loaders else 'DONE'
                
                stat_line = (f"\r[{loader_status}] {pct:5.1f}% | {processed_count}/{total_frames_to_proc} | "
                             f"Submitted: {submitted_val} | FPS: {fps:5.1f} | Faces: {total_faces_video} | "
                             f"Q:{q_in_sz}/{q_out_sz}   ")
                sys.stdout.write(stat_line)
                sys.stdout.flush()
                last_log_time = now
                
                # Detect stalling
                if submitted_val == last_submitted and active_loaders:
                    stall_count += 1
                    if stall_count > 10:
                        print(f"\n[!] WARNING: Loader may be stalled. Submitted={submitted_val}")
                else:
                    stall_count = 0
                last_submitted = submitted_val

            # Exit conditions
            all_loaders_done = not active_loaders
            all_processed = processed_count >= submitted_val
            
            if all_loaders_done and all_processed and submitted_val > 0:
                break
                
            # Timeout if nothing is happening
            if all_loaders_done and submitted_val == 0:
                print("\n[!] ERROR: Loaders finished but submitted 0 frames!")
                break

            time.sleep(0.05)

        print("")

        for l in loaders:
            if l.is_alive():
                l.terminate()
            l.join(timeout=5)

        save_q.put(('end_video', vid_path))

        v_dur = time.time() - v_start_t
        final_fps = processed_count / v_dur if v_dur > 0 else 0

        print(f"✓ DONE: {os.path.basename(vid_path)}")
        print(f"  Time: {v_dur:.2f}s | FPS: {final_fps:.1f} | Faces: {total_faces_video}")

    print("\n[*] Stopping GPU processes...")
    for _ in range(NUM_GPU_PROCESSES):
        filled_q.put(None)
    for p in gpu_processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            
    save_q.put(None)
    saver.join(timeout=10)

    total_time = time.time() - t_global_start
    print(f"\n{'=' * 60}")
    print(f"  ALL DONE. Total: {total_time:.2f}s")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    run()