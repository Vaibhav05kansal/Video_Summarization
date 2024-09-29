import argparse
import torch
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
    args = parser.parse_args()
    return args

def test(args):
    # Load the model
    model = torch.load(args.model_path)
    model.eval()

    # Build test data loader
    _, test_loader, _ = build_dataloader(args)

    # Perform inference and evaluation
    with torch.no_grad():
        predicted_multi_list = []
        target_multi_list = []
        video_number_list = []
        image_number_list = []
        for data, target, video_number, image_number in test_loader:
            predicted_list = []
            target_list = []
            output = model(data)
            multi_target = target.permute(1, 0)
            multi_output = output
            for sequence in range(args.sequence):
                target = multi_target[sequence]
                output = multi_output[sequence]
                predicted_ver2 = output[:, 1]  # Assuming you're interested in the probability of the positive class
                predicted_list.append(predicted_ver2.tolist())
                target_list.append(target.tolist())

            predicted_list = torch.Tensor(predicted_list).permute(1, 0)
            target_list = torch.Tensor(target_list).permute(1, 0)
            predicted_multi_list += predicted_list.tolist()
            target_multi_list += target_list.tolist()
            video_number_list += video_number.tolist()
            image_number_list += image_number.tolist()

        # Flatten the lists
        predicted_multi_list = [item for sublist in predicted_multi_list for item in sublist]
        target_multi_list = [item for sublist in target_multi_list for item in sublist]

        # Perform evaluation
        eval_res = select_keyshots(predicted_multi_list, video_number_list, image_number_list, target_multi_list, args)

        # Print or log evaluation results
        fscore_k = sum(res[2] for res in eval_res) / len(eval_res)
        print("F-score k:", fscore_k)

if __name__ == "__main__":
    args = parse_args()
    test(args)
