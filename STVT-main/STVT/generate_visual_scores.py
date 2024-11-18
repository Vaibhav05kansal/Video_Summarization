import torch

def calculate_visual_scores(model, test_loader, sequence_length):
    visual_scores = []
    with torch.no_grad():
        for data, _, video_number, image_number in test_loader:
            output = model(data)
            for sequence in range(sequence_length):
                output_sequence = output[sequence]
                visual_scores += output_sequence[:, 1].tolist()
    return visual_scores

