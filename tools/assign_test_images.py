import sys
import threading
import time
from pathlib import Path

# Add parent directory to path so we can import config
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import json
import numpy as np
from config import Config
from decord import VideoReader, cpu

def main():
    config = Config()
    json_path = Path(config.template_json_path)
    
    if not json_path.exists():
        print(f"Error: {json_path} does not exist. Cannot run annotator.")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        templates_data = json.load(f)
        
    # Initialize test_frames if missing
    for k in templates_data:
        if "test_frames" not in templates_data[k]:
            templates_data[k]["test_frames"] = []
            
    template_keys = list(templates_data.keys())
    
    if not template_keys:
        print("No templates found in JSON.")
        return
        
    # Setup video with decord
    try:
        vr = VideoReader(config.input_video_path, ctx=cpu(0))
    except Exception as e:
        print(f"Error: Could not open video file {config.input_video_path} with decord. ({e})")
        return
        
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    
    testing_dir = Path("assets/testing")
    testing_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Background caching for seamless scrubbing ---
    thumbnail_cache = {}
    cache_stride = 15
    caching_progress = 0.0
    
    def cache_worker():
        nonlocal caching_progress
        try:
            # We load a small VideoReader for extremely fast sequential thumbnail extraction
            # 426x240 is used to keep RAM footprint to a minimum (~2-3 GB for the whole video)
            vr_cache = VideoReader(config.input_video_path, ctx=cpu(0), width=426, height=240)
            indices = list(range(0, total_frames, cache_stride))
            chunk_size = 64
            for i in range(0, len(indices), chunk_size):
                chunk = indices[i:i+chunk_size]
                frames = vr_cache.get_batch(chunk).asnumpy()
                for idx, f in zip(chunk, frames):
                    thumbnail_cache[idx] = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                caching_progress = min(100.0, (i + len(chunk)) / len(indices) * 100.0)
        except Exception as e:
            print(f"Cache worker error: {e}")
            
    threading.Thread(target=cache_worker, daemon=True).start()
    # -----------------------------------------------
    
    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotator", 1280, 720)
    
    state = "LANDING"
    playing = False
    current_template_idx = 0
    frame_idx = 0
    current_frame_float = 0.0
    last_time = time.time()
    
    def save_json():
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(templates_data, f, indent=4)
            
    def set_frame(idx):
        nonlocal frame_idx, current_frame_float
        frame_idx = max(0, min(int(idx), total_frames - 1))
        current_frame_float = float(frame_idx)
        
    def on_trackbar(val):
        if state == "ANNOTATING":
            set_frame(val)
            
    cv2.createTrackbar("Frame", "Annotator", 0, max(1, total_frames - 1), on_trackbar)
    cv2.createTrackbar("Speed %", "Annotator", 100, 400, lambda val: None)
    
    last_scrub_time = 0
    last_trackbar_pos = 0
    
    while True:
        if state == "LANDING":
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "Template Annotation Progress", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            y = 100
            start_i = max(0, current_template_idx - 10)
            end_i = min(len(template_keys), start_i + 20)
            
            for i in range(start_i, end_i):
                k = template_keys[i]
                count = len(templates_data[k]["test_frames"])
                text = f"{k}: {count} frames"
                if i == current_template_idx:
                    text = "-> " + text
                    color = (0, 255, 0)
                else:
                    color = (200, 200, 200)
                cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y += 25
                
            cv2.putText(img, "Press ENTER to start/continue with selected template.", (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, "W / S or Up/Down arrows to select template.", (600, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f"Background Cache Loading: {caching_progress:.1f}%", (600, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "Press Q to quit.", (600, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Annotator", img)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 13: # Enter
                state = "ANNOTATING"
                set_frame(frame_idx)
                last_time = time.time()
                last_trackbar_pos = frame_idx
            elif key == ord('q'):
                break
            elif key == ord('w') or key == 82:
                current_template_idx = max(0, current_template_idx - 1)
            elif key == ord('s') or key == 84:
                current_template_idx = min(len(template_keys) - 1, current_template_idx + 1)
                
        elif state == "ANNOTATING":
            use_cache = False
            
            if playing:
                now = time.time()
                dt = now - last_time
                last_time = now
                
                speed_pct = cv2.getTrackbarPos("Speed %", "Annotator")
                if speed_pct < 10:
                    speed_pct = 10
                speed_mul = speed_pct / 100.0
                
                current_frame_float += fps * dt * speed_mul
                if current_frame_float >= total_frames:
                    current_frame_float = total_frames - 1
                    playing = False
                
                frame_idx = int(current_frame_float)
                cv2.setTrackbarPos("Frame", "Annotator", frame_idx)
                last_trackbar_pos = frame_idx
                
                # At high playback speeds (>1.5x) or when dropping frames, decoding full frames can lag the UI.
                # Use cache if we are skipping multiple frames rapidly.
                if dt * fps * speed_mul > 2.0:
                    use_cache = True
            else:
                # Detect trackbar dragging
                trackbar_pos = cv2.getTrackbarPos("Frame", "Annotator")
                if trackbar_pos != last_trackbar_pos:
                    last_scrub_time = time.time()
                    last_trackbar_pos = trackbar_pos
                    # Scrubbing!
                    use_cache = True
                else:
                    if time.time() - last_scrub_time < 0.15:
                        use_cache = True
            
            ret = True
            try:
                # Decide whether to use fast cache or exact frame
                nearest_cached = (frame_idx // cache_stride) * cache_stride
                if use_cache and nearest_cached in thumbnail_cache:
                    frame = thumbnail_cache[nearest_cached]
                    # Resize it up to full resolution for display
                    frame = cv2.resize(frame, (config.resolution.width, config.resolution.height))
                else:
                    # Full resolution exact frame
                    frame = vr[frame_idx].asnumpy()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                ret = False
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, "End of video / Error reading frame", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            display_img = frame.copy() if ret else frame
            
            k = template_keys[current_template_idx]
            count = len(templates_data[k]["test_frames"])
            
            template_img_path = Path(config.template_img_dir) / templates_data[k]["filenames"][0]
            if template_img_path.exists():
                t_img = cv2.imread(str(template_img_path), cv2.IMREAD_UNCHANGED)
                if t_img is not None:
                    if t_img.shape[2] == 4:
                        alpha = t_img[:,:,3] / 255.0
                        for c in range(3):
                            t_img[:,:,c] = (t_img[:,:,c] * alpha).astype(np.uint8)
                        t_img = t_img[:,:,:3]
                    
                    th, tw = t_img.shape[:2]
                    h, w = display_img.shape[:2]
                    y_off = 50
                    x_off = w - tw - 50
                    if x_off > 0 and y_off + th < h:
                        # Add a background rect for visibility
                        cv2.rectangle(display_img, (x_off-5, y_off-5), (x_off+tw+5, y_off+th+5), (0, 0, 0), -1)
                        display_img[y_off:y_off+th, x_off:x_off+tw] = t_img
            
            # Info overlay with background
            info_bg = display_img[20:175, 30:800]
            if info_bg.shape == (155, 770, 3):
                display_img[20:175, 30:800] = cv2.addWeighted(info_bg, 0.3, np.zeros_like(info_bg), 0.7, 0)
            
            cv2.putText(display_img, f"Template: {k} ({count} saved)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_img, "SPACE: Play/Pause | A: -1 frame | D: +1 frame", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_img, "T: Tag True | F: Tag False | N: Skip/Next | M: Menu", (50, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_img, ", / . : -/+ 0.5s | J / L : -/+ 5.0s skips", (50, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if use_cache:
                cv2.putText(display_img, "SCRUBBING", (50, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                
            cv2.imshow("Annotator", display_img)
            
            wait_time = 1 if playing else 30
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord(' '):
                playing = not playing
                if playing:
                    last_time = time.time()
            elif key == ord('a'):
                playing = False
                set_frame(frame_idx - 1)
                cv2.setTrackbarPos("Frame", "Annotator", frame_idx)
                last_trackbar_pos = frame_idx
                last_scrub_time = 0 # Force high-res immediately
            elif key == ord('d'):
                playing = False
                set_frame(frame_idx + 1)
                cv2.setTrackbarPos("Frame", "Annotator", frame_idx)
                last_trackbar_pos = frame_idx
                last_scrub_time = 0
            elif key == ord(','):
                set_frame(frame_idx - int(fps * 0.5))
                cv2.setTrackbarPos("Frame", "Annotator", frame_idx)
                last_trackbar_pos = frame_idx
                last_scrub_time = time.time()
            elif key == ord('.'):
                set_frame(frame_idx + int(fps * 0.5))
                cv2.setTrackbarPos("Frame", "Annotator", frame_idx)
                last_trackbar_pos = frame_idx
                last_scrub_time = time.time()
            elif key == ord('j'):
                set_frame(frame_idx - int(fps * 5.0))
                cv2.setTrackbarPos("Frame", "Annotator", frame_idx)
                last_trackbar_pos = frame_idx
                last_scrub_time = time.time()
            elif key == ord('l'):
                set_frame(frame_idx + int(fps * 5.0))
                cv2.setTrackbarPos("Frame", "Annotator", frame_idx)
                last_trackbar_pos = frame_idx
                last_scrub_time = time.time()
            elif key == ord('t') or key == ord('f'):
                # Ensure we are not saving a low-res cache frame
                if use_cache:
                    try:
                        save_frame = vr[frame_idx].asnumpy()
                        save_frame = cv2.cvtColor(save_frame, cv2.COLOR_RGB2BGR)
                    except:
                        save_frame = frame
                else:
                    save_frame = frame
                    
                if ret:
                    pressed = (key == ord('t'))
                    safe_name = k.replace(" ", "_").lower()
                    template_dir = testing_dir / safe_name
                    template_dir.mkdir(exist_ok=True)
                    
                    out_name = f"frame_{frame_idx}_pressed_{str(pressed).lower()}.jpg"
                    out_path = template_dir / out_name
                    cv2.imwrite(str(out_path), save_frame)
                    
                    rel_path = out_path.as_posix()
                    templates_data[k]["test_frames"].append({
                        "path": rel_path,
                        "pressed": pressed,
                        "frame_idx": frame_idx
                    })
                    save_json()
                    
                    # Flash feedback
                    hh, ww = display_img.shape[:2]
                    cv2.putText(display_img, "SAVED!", (ww//2 - 100, hh//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    cv2.imshow("Annotator", display_img)
                    cv2.waitKey(100)
                    
            elif key == ord('n'):
                current_template_idx = min(len(template_keys) - 1, current_template_idx + 1)
            elif key == ord('m') or key == 27: # Esc
                state = "LANDING"
                playing = False
            elif key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
