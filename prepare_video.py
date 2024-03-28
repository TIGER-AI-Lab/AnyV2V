import argparse
from moviepy.editor import VideoFileClip
import os
import glob
import random
import numpy as np
from PIL import Image

def extract_frames(video_path, frame_count=16):
    clip = VideoFileClip(video_path)
    duration = clip.duration
    frames = []
    
    # Calculate the time interval at which to extract frames
    times = np.linspace(0, duration, frame_count, endpoint=False)
    
    for t in times:
        # Extract the frame at the specific timestamp
        frame = clip.get_frame(t)
        # Convert the frame (numpy array) to a PIL Image
        pil_img = Image.fromarray(frame)
        frames.append(pil_img)
    
    return frames

def crop_and_resize_video(input_video_path, output_folder, clip_duration, width=None, height=None, start_time=None, end_time=None, n_frames=16, center_crop=False, x_offset=0, y_offset=0, longest_to_width=False):    # Load the video file
    video = VideoFileClip(input_video_path)
    
    # Calculate start and end times for cropping
    if start_time is not None:
        start_time = float(start_time)
        end_time = start_time + clip_duration
    elif end_time is not None:
        end_time = float(end_time)
        start_time = end_time - clip_duration
    else:
        # Default to random cropping if neither start nor end time is specified
        video_duration = video.duration
        if video_duration <= clip_duration:
            print(f"Skipping {input_video_path}: duration is less than or equal to the clip duration.")
            return
        max_start_time = video_duration - clip_duration
        start_time = random.uniform(0, max_start_time)
        end_time = start_time + clip_duration
    
    # Crop the video
    cropped_video = video.subclip(start_time, end_time)

    if center_crop:
        # Calculate scale to ensure the desired crop size fits within the video
        video_width, video_height = cropped_video.size
        scale_width = video_width / width
        scale_height = video_height / height
        if longest_to_width:
            scale = max(scale_width, scale_height)
        else:
            scale = min(scale_width, scale_height)
        
        # Resize video to ensure the crop area fits within the frame
        # This step ensures that the smallest dimension matches or exceeds 512 pixels
        new_width = int(video_width / scale)
        new_height = int(video_height / scale)
        resized_video = cropped_video.resize(newsize=(new_width, new_height))
        print(f"Resized video to ({new_width}, {new_height})")
        
        # Calculate crop position with offset, ensuring the crop does not go out of bounds
        # The offset calculation needs to ensure that the cropping area remains within the video frame
        offset_x = int(((x_offset + 1) / 2) * (new_width - width))  # Adjusted for [-1, 1] scale
        offset_y = int(((y_offset + 1) / 2) * (new_height - height))  # Adjusted for [-1, 1] scale
        
        # Ensure offsets do not push the crop area out of the video frame
        offset_x = max(0, min(new_width - width, offset_x))
        offset_y = max(0, min(new_height - height, offset_y))
        
        # Apply center crop with offsets
        cropped_video = resized_video.crop(x1=offset_x, y1=offset_y, width=width, height=height)
    elif width and height:
        # Directly resize the video to specified width and height if no center crop is specified
        cropped_video = cropped_video.resize(newsize=(width, height))
    

    # After resizing and cropping, set the frame rate to fps
    fps = n_frames // clip_duration
    final_video = cropped_video.set_fps(fps)
    
    # Prepare the output video path
    filename = os.path.basename(input_video_path)
    output_video_path = os.path.join(output_folder, filename)
    
    # Write the result to the output file
    final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac', fps=fps)
    print(f"Processed {input_video_path}, saved to {output_video_path}")
    return output_video_path

def process_videos(input_folder, output_base_folder, clip_duration, width=None, height=None, start_time=None, end_time=None, n_frames=16, center_crop=False, x_offset=0, y_offset=0, longest_to_width=False):
    video_files = glob.glob(os.path.join(input_folder, '*.mp4'))  # Adjust the pattern if needed
    if video_files == []:
        print(f"No video files found in {input_folder}")
        return
    
    for video_file in video_files:
        crop_and_resize_video(video_file, output_base_folder, clip_duration, width, height, start_time, end_time, n_frames, center_crop, x_offset, y_offset, longest_to_width)
    return 

def main():
    parser = argparse.ArgumentParser(description='Crop and resize video segments.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing video files')
    parser.add_argument('--video_path', type=str, default=None, required=False, help='Path to the input video file')
    parser.add_argument('--output_folder', type=str, default="processed_video_data", help='Path to the folder for the output videos')
    parser.add_argument('--clip_duration', type=int, default=2, required=False, help='Duration of the video clips in seconds')
    parser.add_argument('--width', type=int, default=512, help='Width of the output video (optional)')
    parser.add_argument('--height', type=int, default=512, help='Height of the output video (optional)')
    parser.add_argument('--start_time', type=float, help='Start time for cropping (optional)')
    parser.add_argument('--end_time', type=float, help='End time for cropping (optional)')
    parser.add_argument('--n_frames', type=int, default=16, help='Number of frames to extract from each video')
    parser.add_argument('--center_crop', action='store_true', help='Center crop the video')
    parser.add_argument('--x_offset', type=float, default=0, required=False, help='Horizontal offset for center cropping, range -1 to 1 (optional)')
    parser.add_argument('--y_offset', type=float, default=0, required=False, help='Vertical offset for center cropping, range -1 to 1 (optional)')
    parser.add_argument('--longest_to_width', action='store_true', help='Resize the longest dimension to the specified width')

    args = parser.parse_args()
    
    if args.start_time and args.end_time:
        print("Please specify only one of start_time or end_time, not both.")
        return
    
    if args.video_path:
        crop_and_resize_video(args.video_path, 
                              args.output_folder, 
                              args.clip_duration, 
                              args.width, args.height, 
                              args.start_time, args.end_time, 
                              args.n_frames, 
                              args.center_crop, args.x_offset, args.y_offset, args.longest_to_width)
    else:
        process_videos(args.input_folder, 
                       args.output_folder, 
                       args.clip_duration, 
                       args.width, args.height, 
                       args.start_time, args.end_time, 
                       args.n_frames, 
                       args.center_crop, args.x_offset, args.y_offset, args.longest_to_width)

if __name__ == "__main__":
    main()
