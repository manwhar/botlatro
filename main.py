import itertools
import json
import logging
import time
import concurrent.futures

from config import get_config, configure_logging
from debug_plotter import plot
from eval_template import eval, format_timestamp
from frame import iter_video_frames
from template import template_loader

if __name__ == "__main__":
    configure_logging()

    logger = logging.getLogger(__name__)

    config = get_config()
    templates = template_loader(
        config.template_json_path,
        config.template_img_dir,
        config.resolution,
    )

    def process_frame(arg):
        idx, video_frame = arg
        ts_str = format_timestamp(video_frame.timestamp_seconds, video_frame.index)
        print(f"Processing frame {idx}...", flush=True)
        
        matches = []
        for template in templates:
            result = eval(template, video_frame.image, config, video_timestamp_str=ts_str)
            if result.found and result.plot_kwargs:
                matches.append((template, result.plot_kwargs))
            if result.found and result.clicked:
                break
        return idx, matches

    def chunked_iterable(iterable, size):
        it = iter(iterable)
        while True:
            chunk = tuple(itertools.islice(it, size))
            if not chunk:
                break
            yield chunk

    frames = iter_video_frames(
        config.input_video_path,
        config.resolution,
        config.skip_frames,
        start_timestamp_seconds=config.start_timestamp,
        end_timestamp_seconds=config.end_timestamp,
    )

    start = time.perf_counter()
    results = []

    if config.debug_plot or config.setup_mode:
        # Sequential processing to ensure safe UI/Plotting on the main thread
        for arg in enumerate(frames):
            idx, matches = process_frame(arg)
            for template, plot_kwargs in matches:
                should_plot = config.debug_plot or (config.setup_mode and not template.adjusted)
                if should_plot:
                    plot(**plot_kwargs)
                    try:
                        with open(config.template_json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if template.name in data:
                            template.adjusted = data[template.name].get("adjusted", template.adjusted)
                            template.shadow_dy = data[template.name].get("shadow_dy", template.shadow_dy)
                            template.shadow_dw = data[template.name].get("shadow_dw", template.shadow_dw)
                            template.shadow_dh = data[template.name].get("shadow_dh", template.shadow_dh)
                    except Exception as e:
                        logger.error(f"Failed to reload template data for {template.name}: {e}")
            results.append(idx)
    else:
        # Chunked multithreaded processing for CPU maximum speed (avoids OOM on large videos)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for chunk in chunked_iterable(enumerate(frames), 64):
                chunk_results = list(executor.map(process_frame, chunk))
                results.extend([r[0] for r in chunk_results])

    num_frames = len(results)
    if num_frames > 0:
        elapsed = time.perf_counter() - start
        print(f"Finished in {elapsed:.2f} seconds; {(elapsed / num_frames):.4f} per frame.")
