import os
import json
from matplotlib.pyplot import grid
import numpy as np
from src.utils.augmentation import rotate_grid, flip_horizontal, flip_vertical, change_color
from src.utils.helpers import convert_ndarray
import argparse
import sys
import random

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


def augment_grid_color(challenge, challenge_solution):
    """Apply color change to generate additional augmented grid according to type_to_augment."""
    augmented_challenge = {'train': [], 'test': []}
    augmented_solution = None
    color_map = None
    for i in range(len(challenge['train'])):
        training_sample = challenge['train'][i]
        training_sample_input = training_sample['input']
        training_sample_output = training_sample['output']
        color_changed_input, color_map = change_color(training_sample_input, color_map=color_map)
        color_changed_output, color_map = change_color(training_sample_output, color_map=color_map)
        augmented_challenge['train'].append({'input': color_changed_input, 'output': color_changed_output})

    augmented_test_input_grid = change_color(challenge['test'][0]['input'], color_map=color_map)[0]
    augmented_challenge['test'].append({'input': augmented_test_input_grid})
    augmented_solution = change_color(challenge_solution, color_map=color_map)[0]
    return augmented_challenge, augmented_solution

def augment_grid_flips(challenge, challenge_solution, type_to_flip=None, type_of_flipping=None):
    """Apply horizontal and vertical flips to generate additional augmented grids up to max_augments."""
    augmented_challenge = {'train': [], 'test': []}
    augmented_solution = None
    for i in range(len(challenge['train'])):
        training_sample = challenge['train'][i]
        training_sample_input = training_sample['input']
        training_sample_output = training_sample['output']
        if type_of_flipping == 'horizontal':
            if type_to_flip == 'input':
                augmented_input_grid = flip_horizontal(training_sample_input)
                augmented_output_grid = training_sample_output
            else:
                augmented_input_grid = training_sample_input
                augmented_output_grid = flip_horizontal(training_sample_output) 
        else:
            if type_to_flip == 'input':
                augmented_input_grid = flip_vertical(training_sample_input)
                augmented_output_grid = training_sample_output
            else:
                augmented_input_grid = training_sample_input
                augmented_output_grid = flip_vertical(training_sample_output)
        augmented_challenge['train'].append({'input': augmented_input_grid, 'output': augmented_output_grid})

    test_input_grid = challenge['test'][0]['input']
    test_output_grid = challenge_solution
    if type_of_flipping == 'horizontal':
        if type_to_flip == 'input':
            augmented_test_input_grid = flip_horizontal(test_input_grid)
            augmented_test_output_grid = test_output_grid
        else:
            augmented_test_input_grid = test_input_grid
            augmented_test_output_grid = flip_horizontal(test_output_grid)
    else:
        if type_to_flip == 'input':
            augmented_test_input_grid = flip_vertical(test_input_grid)
            augmented_test_output_grid = test_output_grid
        else:
            augmented_test_input_grid = test_input_grid
            augmented_test_output_grid = flip_vertical(test_output_grid)

    augmented_challenge['test'].append({'input': augmented_test_input_grid})

    augmented_solution = augmented_test_output_grid
    return augmented_challenge, augmented_solution

def augment_grid_rotation(challenge, challenge_solution, type_to_rotate=None, degrees=90):
    """Apply rotation to generate additional augmented grids."""
    augmented_challenge = {'train': [], 'test': []}
    augmented_solution = None
    for i in range(len(challenge['train'])):
        training_sample = challenge['train'][i]
        training_sample_input = training_sample['input']
        training_sample_output = training_sample['output']
        if type_to_rotate == 'input':
            augmented_input_grid = rotate_grid(training_sample_input, degrees)
            augmented_output_grid = training_sample_output
        else:
            augmented_input_grid = training_sample_input
            augmented_output_grid = rotate_grid(training_sample_output, degrees)
        augmented_challenge['train'].append({'input': augmented_input_grid, 'output': augmented_output_grid})

    test_input_grid = challenge['test'][0]['input']
    test_output_grid = challenge_solution
    if type_to_rotate == 'input':
        augmented_test_input_grid = rotate_grid(test_input_grid, degrees)
        augmented_test_output_grid = test_output_grid
    else:
        augmented_test_input_grid = test_input_grid
        augmented_test_output_grid = rotate_grid(test_output_grid, degrees)

    augmented_challenge['test'].append({'input': augmented_test_input_grid})
    augmented_solution = augmented_test_output_grid

    return augmented_challenge, augmented_solution

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
    augmented_solutions = {}
    counter = 0
    for challenge_id, challenge in dataset.items():
        if len(challenge.get('train', [])) == 0 or len(challenge.get('test', [])) == 0 or len(solutions.get(challenge_id, [])) == 0:
            print(f"Skipping challenge {challenge_id} due to insufficient data.")
            continue

        count = 0
        new_challenge_ids = []
        original_challenge_id = challenge_id
        for i in range(len(challenge['test'])):
            new_challenge_id = f"{original_challenge_id}_{count}"
            new_challenge = {
                'train': challenge['train'],
                'test': [challenge['test'][i]]
            }
            new_challenge_ids.append(new_challenge_id)
            augmented_dataset[new_challenge_id] = new_challenge
            augmented_solutions[new_challenge_id] = solutions[challenge_id][i]
            count += 1
        temp_counter = 1
        color_augmented_ids = new_challenge_ids.copy()
        for i in range(len(new_challenge_ids)):
            for j in range(3):
                challenge = augmented_dataset[new_challenge_ids[i]]
                challenge_solution = augmented_solutions[new_challenge_ids[i]]
                new_challenge, new_challenge_solution = augment_grid_color(challenge, challenge_solution)
                new_challenge_id = f"{original_challenge_id}_{count}"
                count += 1
                augmented_dataset[new_challenge_id] = new_challenge
                #testing
                challenge_solution = np.array(challenge_solution)
                is_equal = np.array_equal(new_challenge_solution, challenge_solution)
                augmented_solutions[new_challenge_id] = new_challenge_solution
                color_augmented_ids.append(new_challenge_id)
                temp_counter += 1

        for i in range(len(color_augmented_ids)):
            challenge_id = color_augmented_ids[i]
            challenge = augmented_dataset[challenge_id]
            challenge_solution = augmented_solutions[challenge_id]
            #randomly select whether to apply augmentation on input or output with random library
            type_to_augment = random.choice(['input', 'output'])
            for flip_type in ['horizontal', 'vertical']:
                new_challenge, new_challenge_solution = augment_grid_flips(challenge, challenge_solution, type_to_flip=type_to_augment, type_of_flipping=flip_type)
                new_challenge_id = f"{original_challenge_id}_{count}"
                count += 1
                augmented_dataset[new_challenge_id] = new_challenge
                augmented_solutions[new_challenge_id] = new_challenge_solution
                temp_counter += 1

        for i in range(len(color_augmented_ids)):
            challenge_id = color_augmented_ids[i]
            challenge = augmented_dataset[challenge_id]
            challenge_solution = augmented_solutions[challenge_id]
            #randomly select whether to apply augmentation on input or output with random library
            type_to_augment = random.choice(['input', 'output'])
            for degrees in [90, 180, 270]:
                new_challenge, new_challenge_solution = augment_grid_rotation(challenge, challenge_solution, type_to_rotate=type_to_augment, degrees=degrees)
                new_challenge_id = f"{original_challenge_id}_{count}"
                count += 1
                augmented_dataset[new_challenge_id] = new_challenge
                augmented_solutions[new_challenge_id] = new_challenge_solution  
                temp_counter += 1
        counter += 1
        print(f'Challenge {challenge_id} augmented to {temp_counter} training samples.')
        print(f"Processed challenge {challenge_id}, processed {counter} original challenges so far...")

    challenges_save_path = os.path.join(save_base_dir, challenges_filename)
    solutions_save_path = os.path.join(save_base_dir, solutions_filename)

    with open(challenges_save_path, 'w') as f:
        json.dump(convert_ndarray(augmented_dataset), f, indent=4)

    with open(solutions_save_path, 'w') as f:
        json.dump(convert_ndarray(augmented_solutions), f, indent=4)

    print(f"Augmented dataset saved at: {challenges_save_path}")
    return challenges_save_path, solutions_save_path

def process_dataset_for_eval(base_dir, challenges_filename, solutions_filename, save_base_dir):
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
    augmented_solutions = {}
    counter = 0
    for challenge_id, challenge in dataset.items():
        if len(challenge.get('train', [])) == 0 or len(challenge.get('test', [])) == 0 or len(solutions.get(challenge_id, [])) == 0:
            print(f"Skipping challenge {challenge_id} due to insufficient data.")
            continue

        count = 0
        new_challenge_ids = []
        original_challenge_id = challenge_id
        for i in range(len(challenge['test'])):
            new_challenge_id = f"{original_challenge_id}_{count}"
            new_challenge = {
                'train': challenge['train'],
                'test': [challenge['test'][i]]
            }
            new_challenge_ids.append(new_challenge_id)
            augmented_dataset[new_challenge_id] = new_challenge
            augmented_solutions[new_challenge_id] = solutions[challenge_id][i]
            count += 1
        counter += 1
        print(f"Processed challenge {challenge_id}, processed {counter} original challenges so far...")

    challenges_save_path = os.path.join(save_base_dir, challenges_filename)
    solutions_save_path = os.path.join(save_base_dir, solutions_filename)

    with open(challenges_save_path, 'w') as f:
        json.dump(convert_ndarray(augmented_dataset), f, indent=4)

    with open(solutions_save_path, 'w') as f:
        json.dump(convert_ndarray(augmented_solutions), f, indent=4)

    print(f"Augmented dataset saved at: {challenges_save_path}")
    return challenges_save_path, solutions_save_path

def process_dataset_for_test(base_dir, challenges_filename, save_base_dir):
    if save_base_dir is None or challenges_filename is None:
        return

    with open(os.path.join(base_dir, challenges_filename), 'r') as f:
        dataset = json.load(f)

    for challenge_id, challenge in dataset.items():
        for sample in challenge.get('train', []):
            sample['input'] = np.array(sample['input'])
            if 'output' in sample:
                sample['output'] = np.array(sample['output'])

    augmented_dataset = {}
    counter = 0
    for challenge_id, challenge in dataset.items():
        if len(challenge.get('train', [])) == 0 or len(challenge.get('test', [])) == 0:
            print(f"Skipping challenge {challenge_id} due to insufficient data.")
            continue

        count = 0
        new_challenge_ids = []
        original_challenge_id = challenge_id
        for i in range(len(challenge['test'])):
            new_challenge_id = f"{original_challenge_id}_{count}"
            new_challenge = {
                'train': challenge['train'],
                'test': [challenge['test'][i]]
            }
            new_challenge_ids.append(new_challenge_id)
            augmented_dataset[new_challenge_id] = new_challenge
            count += 1
        counter += 1
        print(f"Processed challenge {challenge_id}, processed {counter} original challenges so far...")

    challenges_save_path = os.path.join(save_base_dir, challenges_filename)

    with open(challenges_save_path, 'w') as f:
        json.dump(convert_ndarray(augmented_dataset), f, indent=4)

    print(f"Augmented dataset saved at: {challenges_save_path}")
    return challenges_save_path

# --------------------------
# Command line usage
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment challenges by generating more problems')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing the dataset JSON files')
    parser.add_argument('--challenges_filename', type=str, required=True, help='Filename of the challenges JSON file')
    parser.add_argument('--solutions_filename', type=str, required=False, help='Filename of the solutions JSON file')
    parser.add_argument('--save_base_dir', type=str, required=True, help='Directory to save the augmented dataset JSON files')
    parser.add_argument('--eval_dataset', action='store_true', help='Flag to indicate evaluation mode')
    parser.add_argument('--test_dataset', action='store_true', help='Flag to indicate test dataset mode')
    args = parser.parse_args()
    if args.eval_dataset:
        process_dataset_for_eval(args.base_dir, args.challenges_filename, args.solutions_filename, args.save_base_dir)
    elif args.test_dataset:
        process_dataset_for_test(args.base_dir, args.challenges_filename, args.save_base_dir)
    else:
        process_and_augment_dataset(args.base_dir, args.challenges_filename, args.solutions_filename, args.save_base_dir)