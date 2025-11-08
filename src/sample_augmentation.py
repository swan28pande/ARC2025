import os
import json
from matplotlib.pyplot import grid
import numpy as np
from src.utils.augmentation import rotate_grid, flip_horizontal, flip_vertical, change_color
from src.utils.helpers import convert_ndarray
import argparse
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def augment_grid_rotation(training_samples, max_augments=5):
    """Select original grids randomly to apply rotations to generate up to max_augments. Apply maximum of 3 rotations"""
    if len(training_samples) >= max_augments:
        return [training_sample for training_sample in training_samples[:max_augments]]
    augmented_training_samples = [training_sample for training_sample in training_samples]
    angles = [90, 180, 270]
    for angle in angles:
        training_sample = training_samples[np.random.randint(len(training_samples))]
        input_rotated_grid = rotate_grid(training_sample['input'], angle)
        output_rotated_grid = rotate_grid(training_sample['output'], angle)
        augmented_training_samples.append({'input': input_rotated_grid, 'output': output_rotated_grid})
        if len(augmented_training_samples) == max_augments:
            break

    return augmented_training_samples


def augment_grid_color(training_samples, max_augments=7):
    """Apply color change to generate additional augmented grids up to max_augments."""
    augmented_training_samples = [training_sample for training_sample in training_samples]
    for training_sample in training_samples:
        existing_count = len(augmented_training_samples)
        if existing_count < max_augments:
            augmented_input_grid, color_map = change_color(training_sample['input'])
            augmented_output_grid, _ = change_color(training_sample['output'], color_map=color_map)
            augmented_training_samples.append({'input': augmented_input_grid, 'output': augmented_output_grid})
    return augmented_training_samples

def augment_grid_flips(training_samples, max_augments=10):
    """Apply horizontal and vertical flips to generate additional augmented grids up to max_augments."""
    augmented_training_samples = [training_sample for training_sample in training_samples]
    # Randomly sample training samples to apply flips and ensure we do not exceed max_augments. Ensure that we do not apply flips to the same sample more than once. Also randomly choose between horizontal and vertical flip.
    np.random.shuffle(augmented_training_samples)
    for training_sample in augmented_training_samples:
        existing_count = len(augmented_training_samples)
        if existing_count < max_augments:
            flip_type = np.random.choice(['horizontal', 'vertical'])
            if flip_type == 'horizontal':
                augmented_input_grid = flip_horizontal(training_sample['input'])
                augmented_output_grid = flip_horizontal(training_sample['output'])
            else:
                augmented_input_grid = flip_vertical(training_sample['input'])
                augmented_output_grid = flip_vertical(training_sample['output'])
            augmented_training_samples.append({'input': augmented_input_grid, 'output': augmented_output_grid})
    return augmented_training_samples


def process_and_augment_dataset(base_dir, challenges_filename, solutions_filename, save_base_dir):
    """
    Load dataset JSON, apply controlled augmentation to train set for all challenges to get exactly 10 samples, and save augmented dataset.
    """
    if save_base_dir is None or challenges_filename is None or solutions_filename is None:
        return

    with open(os.path.join(base_dir, challenges_filename), 'r') as f:
        dataset = json.load(f)

    with open(os.path.join(base_dir, solutions_filename), 'r') as f:
        solutions = json.load(f)

    for challenge_id, challenge in dataset.items():
        for sample in challenge.get('train', []):
            sample['input'] = np.array(sample['input'])
            if 'output' in sample:
                sample['output'] = np.array(sample['output'])

    augmented_dataset = {}
    counter = 0
    for challenge_id, challenge in dataset.items():
        augmented_challenge = {'train': [], 'test': challenge.get('test', [])}
        # check if the number of train samples is not zero or else dont change the challenge
        if len(challenge.get('train', [])) == 0:
            print(f"Skipping challenge {challenge_id} as it has zero training samples.")
            augmented_dataset[challenge_id] = challenge
            continue
        # Check if the number of train samples is less than 5
        if len(challenge['train']) < 5:
            # Apply rotation augmentation until we reach 5 samples or exhaust rotation options
            augmented_challenge['train'] = augment_grid_rotation(challenge['train'], max_augments=5)
        else:
            # Copy existing training samples
            augmented_challenge['train'] = [training_sample for training_sample in challenge['train']]
            
        if len(augmented_challenge['train']) < 7:
            # Apply color change augmentation until we reach 7 samples or exhaust color change options
            augmented_challenge['train'] = augment_grid_color(augmented_challenge['train'], max_augments=7)

        if len(augmented_challenge['train']) < 10:
            # Apply flips augmentation until we reach 10 samples or exhaust flip options
            augmented_challenge['train'] = augment_grid_flips(augmented_challenge['train'], max_augments=10)

        counter += 1
        if counter % 50 == 0:
            print(f"Processed {counter} challenges...")
        print(f"Challenge {challenge_id} augmented from {len(challenge['train'])} to {len(augmented_challenge['train'])} training samples.")
        augmented_dataset[challenge_id] = augmented_challenge
    # Ensure save directory exists
    os.makedirs(save_base_dir, exist_ok=True)


    challenges_save_path = os.path.join(save_base_dir, challenges_filename)
    solutions_save_path = os.path.join(save_base_dir, solutions_filename)

    with open(challenges_save_path, 'w') as f:
        json.dump(convert_ndarray(augmented_dataset), f, indent=4)

    with open(solutions_save_path, 'w') as f:
        json.dump(convert_ndarray(solutions), f, indent=4)

    print(f"Augmented dataset saved at: {challenges_save_path}")
    return challenges_save_path, solutions_save_path


# --------------------------
# Command line usage
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment ARC dataset JSON with multiple challenges to 10 samples per challenge')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing the dataset JSON files')
    parser.add_argument('--challenges_filename', type=str, required=True, help='Filename of the challenges JSON file')
    parser.add_argument('--solutions_filename', type=str, required=True, help='Filename of the solutions JSON file')
    parser.add_argument('--save_base_dir', type=str, required=True, help='Directory to save the augmented dataset JSON files')
    args = parser.parse_args()
    process_and_augment_dataset(args.base_dir, args.challenges_filename, args.solutions_filename, args.save_base_dir)
    
