import moviepy.editor as mp

def add_audio_to_video(summarized_video_path, final_output_path, audio_path):
    summarized_video = mp.VideoFileClip(summarized_video_path)
    audio = mp.AudioFileClip(audio_path)
    video_with_audio = summarized_video.set_audio(audio)
    video_with_audio.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
