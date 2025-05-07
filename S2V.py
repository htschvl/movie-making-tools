import argparse
import numpy as np
import wave
import random
import os
import subprocess
import scipy.signal
from typing import Set
import time


def apply_color_mapping(data: np.ndarray, mode: str) -> np.ndarray:
    if mode == "warp":
        warped = data.astype(np.float32)
        warped = (np.sin(warped / 10.0) * 127 + 128).clip(0, 255)
        return warped.astype(np.uint8)
    elif mode == "rainbow":
        reshaped = data.reshape(-1, 3).astype(np.float32)
        reshaped[:, 0] = (np.sin(reshaped[:, 0] / 15.0) * 127 + 128)
        reshaped[:, 1] = (np.cos(reshaped[:, 1] / 20.0) * 127 + 128)
        reshaped[:, 2] = (np.sin(reshaped[:, 2] / 25.0 + np.pi / 4) * 127 + 128)
        return reshaped.clip(0, 255).astype(np.uint8).flatten()
    elif mode == "drift":
        reshaped = data.reshape(-1, 3)
        drift = np.roll(reshaped, shift=random.randint(-reshaped.shape[0] // 8, reshaped.shape[0] // 8), axis=0)
        return drift.flatten()
    elif mode == "combo" or mode == "yes":
        blend_modes = {
            'warp': 1.0,
            'drift': 1.0,
            'rainbow': 1.0,
            'random': 0.5
        }
        outputs = [apply_color_mapping(data.copy(), m).astype(np.float32) * w for m, w in blend_modes.items() if m != 'random']
        avg = sum(outputs) / sum([w for m, w in blend_modes.items() if m != 'random'])
        return np.clip(avg, 0, 255).astype(np.uint8)
    else:
        # Default mode - create color based on wavelength/frequency mapping
        # Reshape to 3-channel data (assuming it's not already)
        data_flat = data.flatten()
        length = len(data_flat)
        remainder = length % 3
        if remainder > 0:
            # Pad data to make it divisible by 3
            data_flat = np.pad(data_flat, (0, 3 - remainder), 'wrap')
            length = len(data_flat)
            
        # Reshape into RGB
        rgb_data = data_flat.reshape(-1, 3)
        
        # Apply color transformations based on audio frequency characteristics
        transformed = np.zeros_like(rgb_data, dtype=np.float32)
        
        # Red channel - emphasize low frequencies
        transformed[:, 0] = np.clip(rgb_data[:, 0] * 1.2, 0, 255)
        
        # Green channel - emphasize mid frequencies with phase shift
        transformed[:, 1] = np.clip(rgb_data[:, 1] * 0.9 + rgb_data[:, 0] * 0.3, 0, 255)
        
        # Blue channel - emphasize high frequencies with another phase shift
        transformed[:, 2] = np.clip(rgb_data[:, 2] * 1.1 + (rgb_data[:, 0] + rgb_data[:, 1]) * 0.15, 0, 255)
        
        # Add some subtle oscillation for visual interest
        idx = np.arange(rgb_data.shape[0])
        transformed[:, 0] += np.sin(idx / 120.0) * 15
        transformed[:, 1] += np.cos(idx / 100.0) * 12
        transformed[:, 2] += np.sin(idx / 80.0 + np.pi/3) * 18
        
        # Normalize and return
        return np.clip(transformed, 0, 255).astype(np.uint8).flatten()


def detect_beats(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    envelope = np.abs(audio)
    peaks, _ = scipy.signal.find_peaks(envelope, height=np.max(envelope) * 0.5, distance=sample_rate // 10)
    return peaks


def wav_to_video_stream_safe(
    wav_file: str,
    output_video: str,
    fps: float = 30.0,
    color_map: str = 'default',
    fx: Set[str] = set()
) -> None:
    print(f"Opening WAV file: {wav_file}")
    with wave.open(wav_file, 'rb') as wf:
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw_audio = wf.readframes(nframes)
        
    print(f"Audio info: {framerate}Hz, {nchannels} channels, {sampwidth} bytes per sample")
    print(f"Total audio frames: {nframes}, Duration: {nframes/framerate:.2f}s")

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth, np.uint8)
    audio_array = np.frombuffer(raw_audio, dtype=dtype)

    if nchannels > 1:
        audio_array = audio_array.reshape(-1, nchannels).mean(axis=1).astype(dtype)

    duration_seconds = nframes / framerate
    total_frames = int(duration_seconds * fps)
    print(f"Will generate {total_frames} video frames at {fps} fps")

    frame_size = (640, 480)
    bytes_per_frame = frame_size[0] * frame_size[1] * 3
    print(f"Frame size: {frame_size}, {bytes_per_frame} bytes per frame")

    beat_frames = set()
    if 'beat' in fx:
        print("Detecting beats...")
        peaks = detect_beats(raw_audio, framerate)
        beat_frames = set((peaks / framerate * fps).astype(int))
        print(f"Detected {len(beat_frames)} beats")
    
    # Create a full RGB video data buffer
    print(f"Allocating memory for video data: {bytes_per_frame * total_frames / (1024*1024):.1f} MB")
    video_data = np.zeros(bytes_per_frame * total_frames, dtype=np.uint8)
    
    temp_filename = "temp_video.rgb"
    print(f"Will write RGB data to temporary file: {temp_filename}")
    
    start_time = time.time()
    try:
        # Process frames
        for frame_index in range(total_frames):
            # Calculate offset for this frame in the video data buffer
            offset = frame_index * bytes_per_frame
            
            # Map audio data to this frame
            audio_start = int((frame_index / fps) * framerate)
            audio_end = int(((frame_index + 1) / fps) * framerate)
            
            # Ensure we don't go out of bounds
            audio_start = min(audio_start, len(audio_array)-1)
            audio_end = min(audio_end, len(audio_array))
            
            # Extract audio chunk for this frame
            if audio_end > audio_start:
                chunk = audio_array[audio_start:audio_end]
            else:
                # If we're at the end, just use the last sample
                chunk = audio_array[-1:]
            
            # If chunk is too small, repeat it to fill the frame
            if len(chunk) < bytes_per_frame // 3:
                repeats = (bytes_per_frame // 3) // max(1, len(chunk)) + 1
                chunk = np.tile(chunk, repeats)[:bytes_per_frame // 3]
            
            # Normalize chunk to 0-255 range
            chunk_min = int(np.min(chunk))
            chunk_max = int(np.max(chunk))
            if chunk_max > chunk_min:
                norm_chunk = ((chunk.astype(np.int32) - chunk_min) * 255 // max(1, chunk_max - chunk_min))
                norm_chunk = norm_chunk.clip(0, 255).astype(np.uint8)
            else:
                norm_chunk = np.zeros(len(chunk), dtype=np.uint8)
            
            # Ensure norm_chunk is long enough
            if len(norm_chunk) < bytes_per_frame:
                norm_chunk = np.resize(norm_chunk, bytes_per_frame)
            
            # Apply color mapping
            mode = random.choice(['warp', 'rainbow', 'drift']) if color_map == 'random' else color_map
            frame_data = apply_color_mapping(norm_chunk, mode)
            
            # Apply effects
            fx_outputs = []
            base = frame_data.astype(np.float32)

            if 'glitch' in fx:
                temp = base.copy()
                glitch_density = 0.02
                glitch_indices = np.random.choice(len(temp), size=int(len(temp) * glitch_density), replace=False)
                temp[glitch_indices] = np.random.randint(0, 256, size=len(glitch_indices))
                fx_outputs.append(temp)

            if 'pulse' in fx and frame_index % 30 < 15:
                temp = (base * 0.6)
                fx_outputs.append(temp)

            if 'beat' in fx and frame_index in beat_frames:
                temp = np.roll(base, shift=frame_index % 250)
                fx_outputs.append(temp)

            if fx_outputs:
                fx_outputs.append(base) 
                blended = sum(fx_outputs) / len(fx_outputs)
                frame_data = blended.clip(0, 255).astype(np.uint8)
            else:
                frame_data = base.astype(np.uint8)
            
            # Store the frame in our video data buffer
            video_data[offset:offset+len(frame_data)] = frame_data[:bytes_per_frame]
            
            # Calculate frames per second and ETA
            elapsed_time = time.time() - start_time
            frames_per_sec = (frame_index + 1) / elapsed_time if elapsed_time > 0 else 0
            eta = (total_frames - (frame_index + 1)) / frames_per_sec if frames_per_sec > 0 else 0
            print(f"\r{frame_index+1}/{total_frames} {(frame_index+1)/total_frames*100:.1f}% - {frames_per_sec:.1f} fps - ETA: {eta:.1f}s", end="", flush=True)
        
        print("\nWriting video data to file...")
        
        # Write the video data to the temporary file
        with open(temp_filename, "wb") as f:
            f.write(video_data.tobytes())
        
        # Check the file was written correctly
        file_size = os.path.getsize(temp_filename)
        expected_size = bytes_per_frame * total_frames
        print(f"Wrote {file_size} bytes (expected: {expected_size} bytes)")
        
        if file_size != expected_size:
            print("WARNING: File size doesn't match expected size!")
        
        print(f"Running FFmpeg to create {output_video}...")
        
        # Run FFmpeg to convert the raw RGB data to a video file
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{frame_size[0]}x{frame_size[1]}",
            "-r", str(fps),
            "-i", temp_filename,
            "-i", wav_file,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-pix_fmt", "yuv420p",
            output_video
        ]
        
        print(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True)
        
        print(f"Video created successfully: {output_video}")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            print(f"Removing temporary file: {temp_filename}")
            os.remove(temp_filename)
        
        print("Done!")

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WAV to artistic video")
    parser.add_argument('--input', required=True, help='Input WAV file')
    parser.add_argument('--output', required=True, help='Output video file (e.g. MP4)')
    parser.add_argument('--fps', type=float, default=None, help='Frames per second')
    parser.add_argument('--color', choices=['default', 'warp', 'rainbow', 'drift', 'random', 'yes'], default='default')
    parser.add_argument('--fx', type=str, help='Combined effects: comma-separated values like glitch,beat,pulse')
    parser.add_argument('--preset', choices=['art'], help="Shortcut for setting recommended artistic config")

    args = parser.parse_args()

    # Apply preset
    if args.preset == 'art':
        args.color = 'yes'
        args.fx = 'yes'

    # Expand --color yes
    if args.color == 'yes':
        args.color = 'combo'  # internal fallback handled in apply_color_mapping

    # Expand --fx yes
    fx_set = {'beat', 'pulse', 'glitch'} if args.fx == 'yes' else set(args.fx.split(",")) if args.fx else set()

    print(f"Starting vid2song with: color={args.color}, fx={fx_set}")
    wav_to_video_stream_safe(args.input, args.output, args.fps or 30.0, args.color, fx_set)
    print("Process completed successfully!")

if __name__ == "__main__":
    main()