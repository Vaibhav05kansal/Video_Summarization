import moviepy.editor as mp

def extract_audio_segments(input_video_path, selected_frames, fps, output_audio_path):
    video = mp.VideoFileClip(input_video_path)
    segment_duration = 1 / fps
    audio_segments = []
    for frame_num in selected_frames:
        start_time = frame_num * segment_duration
        end_time = start_time + segment_duration
        audio_segments.append(video.audio.subclip(start_time, end_time))
    summarized_audio = mp.concatenate_audioclips(audio_segments)
    summarized_audio.write_audiofile(output_audio_path)
