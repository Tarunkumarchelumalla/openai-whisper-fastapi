import subprocess
from pathlib import Path
from typing import List, Dict


class VideoClippingService:
    def __init__(self, preset: str = "fast", crf: int = 18):
        self.preset = preset
        self.crf = crf

    # -----------------------------------------
    # Utility: Check if file is readable
    # -----------------------------------------
    def is_video_readable(self, path: str) -> bool:
        try:
            subprocess.run(
                ["ffprobe", "-v", "error", "-i", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    # -----------------------------------------
    # Utility: Get stream info (for debugging)
    # -----------------------------------------
    def get_stream_info(self, path: str) -> str:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_name,codec_type",
            "-of", "json",
            path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout

    # -----------------------------------------
    # Step 1: Remux (rebuild container, no decode)
    # -----------------------------------------
    def remux_video(self, input_video: str, output_video: str):
        print(f"[Remux] {input_video} -> {output_video}")

        cmd = [
            "ffmpeg", "-y",
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+igndts",
            "-i", input_video,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c", "copy",
            output_video
        ]

        subprocess.run(cmd, check=True)

    # -----------------------------------------
    # Step 2: Nuclear Normalize (AV1 safe)
    # -----------------------------------------
    def normalize_video(self, input_video: str, output_video: str):
        print(f"[Normalize] {input_video} -> {output_video}")

        cmd = [
            "ffmpeg", "-y",

            # ignore broken data
            "-err_detect", "ignore_err",
            "-fflags", "+genpts+igndts+discardcorrupt",
            "-ignore_unknown",

            # massive probe for broken headers
            "-analyzeduration", "300M",
            "-probesize", "300M",

            "-i", input_video,

            # force first streams
            "-map", "0:v:0",
            "-map", "0:a?",

            # force fps rebuild (critical for broken AV1)
            "-vf", "fps=30",

            # safe re-encode
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.2",

            # stable keyframes
            "-g", "60",
            "-keyint_min", "60",
            "-sc_threshold", "0",

            # audio
            "-c:a", "aac",
            "-ar", "44100",

            # web safe
            "-movflags", "+faststart",

            output_video
        ]

        subprocess.run(cmd, check=True)

    # -----------------------------------------
    # Step 3: Cut clips (FRAME ACCURATE)
    # -----------------------------------------
    def process_clips(
        self,
        input_video: str,
        captions: List[Dict],
        clips: List[Dict],
        output_dir: str = "outputs",
    ) -> List[Dict]:

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if not self.is_video_readable(input_video):
            raise ValueError("❌ Input video is not readable or is corrupted")

        remuxed_video = f"{output_dir}/remuxed.mkv"
        normalized_video = f"{output_dir}/normalized.mp4"

        # ---- Pipeline ----
        self.remux_video(input_video, remuxed_video)
        self.normalize_video(remuxed_video, normalized_video)

        results = []

        for i, clip in enumerate(clips):
            start = float(10)
            end = float(40)

            if end <= start:
                print(f"[Skip] Invalid clip range {start} - {end}")
                continue

            output_path = f"{output_dir}/clip_{i+1}.mp4"

            print(f"[Clip {i+1}] {start}s -> {end}s")

            # IMPORTANT: accurate seek (ss AFTER -i)
            cmd = [
                "ffmpeg", "-y",
                "-i", normalized_video,

                "-ss", str(start),
                "-to", str(end),

                "-c:v", "libx264",
                "-preset", self.preset,
                "-crf", str(self.crf),

                "-c:a", "aac",
                "-movflags", "+faststart",

                output_path
            ]

            subprocess.run(cmd, check=True)

            results.append({
                "index": i + 1,
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "path": output_path
            })

        return results
