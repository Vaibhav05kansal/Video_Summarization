import argparse
import torch
import cv2
from STVT.build_dataloader import build_dataloader
from STVT.build_model import build_model
from STVT.eval import select_keyshots

def parse_args():
    parser = argparse.ArgumentParser(description='Test trained model')
    parser.add_argument('--model_path', type=str, default='your_model.pth', help='Path to the trained model')
    parser.add_argument('--dataset', default='TVSum', help='Dataset names.')
    parser.add_argument('--test_dataset', type=str, default="1,2,11,16,18,20,31,32,35,46",
                        help='The number of test video in the dataset.')
    parser.add_argument('--sequence', type=int, default=16, help='The number of sequence.')
    parser.add_argument('--val_batch_size', type=int, default=40, help='input batch size for val')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')  # Highlighted change
    args = parser.parse_args()
    return args

def generate_video(predicted_multi_list, video_number_list, image_number_list, video_path, output_path):
    print("Generating video...")
    # Open the original video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Initialize video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate through predictions and write frames to output video
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame_num to integer
        frame_num = int(frame_num)

        # Check if the predicted score for this frame is above a threshold
        # if predicted_multi_list[frame_num] > 0.15:  # Adjusted threshold
        #     out.write(frame)
        #     print(f"Frame {frame_num}, Score: {predicted_multi_list[frame_num]}, Considered")
        # else:
        #     print(f"Frame {frame_num}, Score: {predicted_multi_list[frame_num]}, Not considered")

        if frame_num < len(predicted_multi_list):
            # If the predicted score for this frame is above a threshold, write it to the output video
            if predicted_multi_list[frame_num] > 0.15:  # Adjusted threshold
                out.write(frame)
                print(f"Frame {frame_num}, Score: {predicted_multi_list[frame_num]}, Considered")
            else:
                print(f"Frame {frame_num}, Score: {predicted_multi_list[frame_num]}, Not considered")
        else:
            print(f"Frame {frame_num}, Index out of range")


        # Increment frame number
        frame_num += 1

    # Release resources
    cap.release()
    out.release()


    # Initialize video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate through predictions and write frames to output video
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame_num to integer
        frame_num = int(frame_num)

        # If the predicted score for this frame is above a threshold, write it to the output video
        if predicted_multi_list[frame_num] > 0.15:  # Adjusted threshold
            out.write(frame)
        

        # Increment frame number
        frame_num += 1

    # Release resources
    cap.release()
    out.release()

def main():
    args = parse_args()

    # Load the model
    checkpoint = torch.load(args.model_path)
    
    # Find the correct key containing the model's state dictionary
    model_state_dict = checkpoint.get('state_dict', checkpoint)

    # Build your model architecture
    model = build_model(args)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Build test data loader
    _, test_loader, _ = build_dataloader(args)

    # Perform inference and get predicted_multi_list
    with torch.no_grad():
        predicted_multi_list = []
        video_number_list = []
        image_number_list = []
        for data, _, video_number, image_number in test_loader:
            output = model(data)
            for sequence in range(args.sequence):
                output_sequence = output[sequence]
                predicted_ver2 = output_sequence[:, 1]  # Assuming you're interested in the probability of the positive class
                predicted_multi_list += predicted_ver2.tolist()
                video_number_list += video_number.tolist()
                image_number_list += image_number.tolist()

    # Generate video using predicted_multi_list
    video_path = "./test_video1.mp4"
    output_path = "./output_summ1.avi"
    generate_video(predicted_multi_list, video_number_list, image_number_list, video_path, output_path)

if __name__ == "__main__":
    main()