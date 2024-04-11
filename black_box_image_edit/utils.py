import os
from moviepy.editor import VideoFileClip
import random
from PIL import Image
import numpy as np

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
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = os.path.basename(input_video_path)
    output_video_path = os.path.join(output_folder, filename)
    
    # Write the result to the output file
    final_video.write_videofile(output_video_path, codec='libx264', audio_codec='aac', fps=fps)
    print(f"Processed {input_video_path}, saved to {output_video_path}")
    return output_video_path


def infer_video_prompt(model, video_path, output_dir, prompt, prompt_type="instruct", force_512=False, seed=42, negative_prompt="", overwrite=False):
    """
    Processes videos from the input directory, resizes them to 512x512 before feeding into the model by first frame,
    and saves the processed video back to its original size in the output directory.

    Args:
        model: The video editing model.
        input_dir (str): Path to the directory containing input videos.
        output_dir (str): Path to the directory where processed videos will be saved.
        prompt (str): Instruction prompt for video editing.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_clip = VideoFileClip(video_path)
    video_filename = os.path.basename(video_path)
    # filename_noext = os.path.splitext(video_filename)[0]
    
    # Create the output directory if it does not exist
    # final_output_dir = os.path.join(output_dir, filename_noext)
    final_output_dir = output_dir
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    result_path = os.path.join(final_output_dir, prompt + ".png")

    # Check if result already exists
    if os.path.exists(result_path) and overwrite is False:
        print(f"Result already exists: {result_path}")
        return

    def process_frame(image):
        pil_image = Image.fromarray(image)
        if force_512:
            pil_image = pil_image.resize((512, 512), Image.LANCZOS)
        if prompt_type == "instruct":
            result = model.infer_one_image(pil_image, instruct_prompt=prompt, seed=seed, negative_prompt=negative_prompt)
        else:
            result = model.infer_one_image(pil_image, target_prompt=prompt, seed=seed, negative_prompt=negative_prompt)
        if force_512:
            result = result.resize(video_clip.size, Image.LANCZOS)
        return np.array(result)
    
    # Process only the first frame
    first_frame = video_clip.get_frame(0)  # Get the first frame
    processed_frame = process_frame(first_frame)  # Process the first frame


    #Image.fromarray(first_frame).save(os.path.join(final_output_dir, "00000.png"))
    Image.fromarray(processed_frame).save(result_path)
    print(f"Processed and saved the first frame: {result_path}")
    return result_path

def infer_video_style(model, video_path, output_dir, style_image, prompt, force_512=False, seed=42, negative_prompt="", overwrite=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_clip = VideoFileClip(video_path)
    video_filename = os.path.basename(video_path)
    final_output_dir = output_dir
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    result_path = os.path.join(final_output_dir, "style" + ".png")
    if os.path.exists(result_path) and overwrite is False:
        print(f"Result already exists: {result_path}")
        return
    def process_frame(image):
        pil_image = Image.fromarray(image)
        if force_512:
            pil_image = pil_image.resize((512, 512), Image.LANCZOS)
        result = model.infer_one_image(pil_image, 
                                        style_image=style_image,
                                        prompt=prompt, 
                                        seed=seed, 
                                        negative_prompt=negative_prompt)
        if force_512:
            result = result.resize(video_clip.size, Image.LANCZOS)
        return np.array(result)
    # Process only the first frame
    first_frame = video_clip.get_frame(0)  # Get the first frame
    processed_frame = process_frame(first_frame)  # Process the first frame
    Image.fromarray(processed_frame).save(result_path)
    print(f"Processed and saved the first frame: {result_path}")
    return result_path