import os
import json





def prompt_training(training_challenges_path, training_solutions_path,save_path):


    with open(training_challenges_path, 'r') as f:
        challenges = json.load(f)

    with open(training_solutions_path, 'r') as f:
        solutions = json.load(f)

    # ============================================
    # ARC-AGI Prompt and Output Templates
    # ============================================

    # Prompt template
    ARC_PROMPT_TEMPLATE = """You are given initial example input-output grid pairs from the ARC (Abstraction and Reasoning Corpus) task.
    Each grid is represented as a 2D array of integers ranging from 0 to 9. Each integer corresponds to a specific color:

    0 - Black  
    1 - Blue  
    2 - Red  
    3 - Green  
    4 - Yellow  
    5 - Gray  
    6 - Magenta  
    7 - Orange  
    8 - Light Blue  
    9 - Dark Red  

    Each cell in the grid represents one colored square. The transformation from input to output follows a specific visual or logical pattern.

    Your task:
    1. Study the given initial example input-output pairs carefully.
    2. Some examples may be incorrect or noisy — identify the pattern that the *majority* of examples follow.
    3. Infer the correct transformation rule that maps the input grid to the output grid.
    4. Apply this inferred transformation to the provided test input grid to produce the correct output grid.

    Below are the example pairs (in JSON format):

    {examples}

    Now, here is the test input grid (in JSON format):

    {test_input}

    Generate the output grid that correctly applies the inferred transformation to this test input.

    Your response should be formatted strictly as a JSON code block in the following form:

    ```json
    {{
        "output": [[...]]
    }}
    ```"""

    # Output template (for model-predicted grid)
    ARC_OUTPUT_TEMPLATE = """```json
    {{
        "output": {output_grid}
    }}
    ```"""

    prompts = []
    for challenge_id, challenge in challenges.items():
        solution = solutions.get(challenge_id, [])
        if solution == []:
            continue
        train_examples = challenge.get('train', [])
        test_inputs = challenge.get('test', [{}])
        if train_examples == []:
            continue
        examples_json = "\n".join([
            f"Example {i+1}:\n" + json.dumps(ex, indent=4)
            for i, ex in enumerate(train_examples)
        ])
        for i in range(len(test_inputs)):
            test_input = test_inputs[i]
            test_output = solution
            if not test_input.get('input'):
                continue
            test_input_json = json.dumps(test_input.get('input', []), indent=4)
            final_prompt = ARC_PROMPT_TEMPLATE.format(
                examples=examples_json,
                test_input=test_input_json
            )
            challenge_id_new = challenge_id
            if(len(test_inputs)>1):
                challenge_id_new = f"{challenge_id}_test{i}"
            output = ARC_OUTPUT_TEMPLATE.format(output_grid=json.dumps(test_output, indent=4))
            prompts.append({
                "challenge_id": challenge_id_new,
                "prompt": final_prompt,
                "output": output
            })
    with open(save_path, 'w') as f:
        json.dump(prompts, f, indent=4)

def prompt_test(challenges_path,save_path):


    with open(challenges_path, 'r') as f:
        challenges = json.load(f)


    # ============================================
    # ARC-AGI Prompt and Output Templates
    # ============================================

    # Prompt template
    ARC_PROMPT_TEMPLATE = """You are given initial example input-output grid pairs from the ARC (Abstraction and Reasoning Corpus) task.
    Each grid is represented as a 2D array of integers ranging from 0 to 9. Each integer corresponds to a specific color:

    0 - Black  
    1 - Blue  
    2 - Red  
    3 - Green  
    4 - Yellow  
    5 - Gray  
    6 - Magenta  
    7 - Orange  
    8 - Light Blue  
    9 - Dark Red  

    Each cell in the grid represents one colored square. The transformation from input to output follows a specific visual or logical pattern.

    Your task:
    1. Study the given initial example input-output pairs carefully.
    2. Some examples may be incorrect or noisy — identify the pattern that the *majority* of examples follow.
    3. Infer the correct transformation rule that maps the input grid to the output grid.
    4. Apply this inferred transformation to the provided test input grid to produce the correct output grid.

    Below are the example pairs (in JSON format):

    {examples}

    Now, here is the test input grid (in JSON format):

    {test_input}

    Generate the output grid that correctly applies the inferred transformation to this test input.

    Your response should be formatted strictly as a JSON code block in the following form:

    ```json
    {{
        "output": [[...]]
    }}
    ```"""
    prompts = []
    
    for challenge_id, challenge in challenges.items():
        train_examples = challenge.get('train', [])
        test_inputs = challenge.get('test', [{}])
        if train_examples == []:
            continue
        examples_json = "\n".join([
            f"Example {i+1}:\n" + json.dumps(ex, indent=4)
            for i, ex in enumerate(train_examples)
        ])
        for i in range(len(test_inputs)):
            test_input = test_inputs[i]
            if not test_input.get('input'):
                continue
            test_input_json = json.dumps(test_input.get('input', []), indent=4)
            final_prompt = ARC_PROMPT_TEMPLATE.format(
                examples=examples_json,
                test_input=test_input_json
            )
            challenge_id_new = challenge_id
            if(len(test_inputs)>1):
                challenge_id_new = f"{challenge_id}_test{i}"
            prompts.append({
                "challenge_id": challenge_id_new,
                "prompt": final_prompt,
            })
            
    with open(save_path, 'w') as f:
        json.dump(prompts, f, indent=4)

def prompt_evaluation(challenges_path,solutions_path,save_path,validation_path):


    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    # ============================================
    # ARC-AGI Prompt and Output Templates
    # ============================================

    # Prompt template
    ARC_PROMPT_TEMPLATE = """You are given initial example input-output grid pairs from the ARC (Abstraction and Reasoning Corpus) task.
    Each grid is represented as a 2D array of integers ranging from 0 to 9. Each integer corresponds to a specific color:

    0 - Black  
    1 - Blue  
    2 - Red  
    3 - Green  
    4 - Yellow  
    5 - Gray  
    6 - Magenta  
    7 - Orange  
    8 - Light Blue  
    9 - Dark Red  

    Each cell in the grid represents one colored square. The transformation from input to output follows a specific visual or logical pattern.

    Your task:
    1. Study the given initial example input-output pairs carefully.
    2. Some examples may be incorrect or noisy — identify the pattern that the *majority* of examples follow.
    3. Infer the correct transformation rule that maps the input grid to the output grid.
    4. Apply this inferred transformation to the provided test input grid to produce the correct output grid.

    Below are the example pairs (in JSON format):

    {examples}

    Now, here is the test input grid (in JSON format):

    {test_input}

    Generate the output grid that correctly applies the inferred transformation to this test input.

    Your response should be formatted strictly as a JSON code block in the following form:

    ```json
    {{
        "output": [[...]]
    }}
    ```"""


    # Output template (for model-predicted grid)
    ARC_OUTPUT_TEMPLATE = """```json
    {{
        "output": {output_grid}
    }}
    ```"""

    prompts = []
    validation_prompts = []
    for challenge_id, challenge in challenges.items():
        solution = solutions.get(challenge_id, [])
        if solution == []:
            continue
        train_examples = challenge.get('train', [])
        test_inputs = challenge.get('test', [{}])
        if train_examples == []:
            continue
        examples_json = "\n".join([
            f"Example {i+1}:\n" + json.dumps(ex, indent=4)
            for i, ex in enumerate(train_examples)
        ])
        for i in range(len(test_inputs)):
            test_input = test_inputs[i]
            test_output = solution
            if not test_input.get('input'):
                continue
            test_input_json = json.dumps(test_input.get('input', []), indent=4)
            final_prompt = ARC_PROMPT_TEMPLATE.format(
                examples=examples_json,
                test_input=test_input_json
            )
            challenge_id_new = challenge_id
            output = ARC_OUTPUT_TEMPLATE.format(output_grid=json.dumps(test_output, indent=4))
            if(len(test_inputs)>1):
                challenge_id_new = f"{challenge_id}_test{i}"
            prompts.append({
                "challenge_id": challenge_id_new,
                "prompt": final_prompt,
            })
            validation_prompts.append({
                    "challenge_id": challenge_id_new,
                    "prompt": final_prompt,
                    "output": output 
                })
            
    with open(save_path, 'w') as f:
        json.dump(prompts, f, indent=4)

    with open(validation_path, 'w') as f:
        json.dump(validation_prompts, f, indent=4)
    


if __name__ == "__main__":
    training_challenges_path = os.path.join("../data/arc-agi-2025/problem_augmented/arc-agi_training_challenges.json")
    training_solutions_path = os.path.join("../data/arc-agi-2025/problem_augmented/arc-agi_training_solutions.json")
    evaluation_challenges_path = os.path.join("../data/arc-agi-2025/processed_data_for_eval/arc-agi_evaluation_challenges.json")
    evaluation_solutions_path = os.path.join("../data/arc-agi-2025/processed_data_for_eval/arc-agi_evaluation_solutions.json")
    test_challenges_path = os.path.join("../data/arc-agi-2025/processed_data_for_eval/arc-agi_test_challenges.json")    
    prompt_training(training_challenges_path,training_solutions_path,save_path="../data/arc-agi-2025/prompts/arc-agi_training_prompts.json")
    prompt_evaluation(evaluation_challenges_path,evaluation_solutions_path,save_path="../data/arc-agi-2025/prompts/arc-agi_evaluation_prompts.json",validation_path="../data/arc-agi-2025/prompts/arc-agi_validation_prompts.json")
    prompt_test(test_challenges_path,save_path="../data/arc-agi-2025/prompts/arc-agi_test_prompts.json")

