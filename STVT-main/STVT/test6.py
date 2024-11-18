import argparse
import torch
import os
from STVT.build_dataloader import build_dataloader
from STVT.build_model import build_model
from generate_visual_scores import calculate_visual_scores
from generate_audio_scores import calculate_audio_scores
from fuse_scores import fuse_scores
from generate_video import generate_summarized_video
from extract_audio_segments import extract_audio_segments
from add_audio_to_video import add_audio_to_video

def parse_args():
    parser = argparse.ArgumentParser(description='Test trained model with audio-visual fusion')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--dataset', type=str, default='TVSum', help='Dataset name')
    parser.add_argument('--test_dataset', type=str, default="1,2,11,16,18,20,31,32,35,46",
                        help='Comma-separated list of test video indices')
    parser.add_argument('--sequence', type=int, default=16, help='The number of sequence')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training data')
    parser.add_argument('--val_batch_size', type=int, default=40, help='Batch size for validation data')  # Added this line
    parser.add_argument('--alpha', type=float, default=0.75, help='Weight for audio-visual fusion (0-1)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the model
    checkpoint = torch.load(args.model_path)
    model_state_dict = checkpoint.get('state_dict', checkpoint)
    model = build_model(args)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Build test data loader
    _, test_loader, _ = build_dataloader(args)

    # Generate visual scores
    visual_scores = calculate_visual_scores(model, test_loader, args.sequence)

    # Calculate audio scores
    video_path = "./Incredible Final Over of England's Innings! _ Stokes Forces Super Over _ ICC Cricket World Cup 2019 - ICC (360p, h264, youtube).mp4"  # Input video path
    audio_scores = calculate_audio_scores(video_path, frame_count=len(visual_scores), fps=30)  # Adjust `fps` as needed

    # Fuse the visual and audio scores
    fused_scores = fuse_scores(visual_scores, audio_scores, alpha=args.alpha)

    # Define paths for intermediate and final outputs
    temp_video_path = "./temp_summarized.avi"  # Temporary video without audio
    final_output_path = "./summarized_output_with_audio_CRICKE.mp4"  # Final video with audio
    temp_audio_path = "./temp_audio.mp3"  # Temporary audio file for extracted segments

    # Generate summarized video based on fused scores
    selected_frames, fps = generate_summarized_video(fused_scores, video_path, temp_video_path)
    if selected_frames is None:
        print("No summary generated due to insufficient selected frames.")
        return

    # Extract audio segments matching the selected frames
    extract_audio_segments(video_path, selected_frames, fps, temp_audio_path)

    # Add trimmed audio to the summarized video
    add_audio_to_video(temp_video_path, final_output_path, temp_audio_path)

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(temp_audio_path)

if __name__ == "__main__":
    main()
