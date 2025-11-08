import argparse
import os
import json
import dotenv
import tinker
from tinker.types import SamplingParams, ModelInput
import re
import time

dotenv.load_dotenv()


def predict_evaluation(dataset_type, example_count=None, model_name=None, checkpoint_path=None, max_tokens=1000, max_output_tokens=1000):
    """Generate predictions for Evaluation or Test dataset."""
    
    if dataset_type == "Evaluation":
        challenges_path = os.path.join("./data/arc-agi-2025/prompts/arc-agi_evaluation_prompts.json")
        save_path = os.path.join("./data/arc-agi-2025/predictions/arc-agi_evaluation_predictions.json")
        solutions_path = os.path.join("./data/arc-agi-2025/processed_data_for_eval/arc-agi_evaluation_solutions.json")
    elif dataset_type == "Test":
        challenges_path = os.path.join("./data/arc-agi-2025/prompts/arc-agi_test_prompts.json")
        save_path = os.path.join("./data/arc-agi-2025/predictions/arc-agi_test_predictions.json")
        solutions_path = None  # No solutions for test set
    else:
        raise ValueError("Invalid dataset type. Choose 'Evaluation' or 'Test'.")

    # Load challenges
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    # Limit examples if specified
    if example_count is not None:
        challenges = challenges[:example_count]
        print(f"Limiting to {example_count} examples")

    # Initialize Tinker API
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY must be set in the environment.")

    service_client = tinker.ServiceClient(api_key=api_key)
    
    training_client = service_client.create_lora_training_client(
        base_model=model_name
    )
    
    sampling_client = service_client.create_sampling_client(
        model_path=checkpoint_path
    )
    
    tokenizer = training_client.get_tokenizer()

    # Generate predictions
    results = []
    stop_sequences = ["]\n]\n}\n```"]

    for i, example in enumerate(challenges):
        prompt_text = example.get("prompt") or example.get("input")
        example_id = example.get("challenge_id")
        
        if not prompt_text:
            print(f"Skipping example {i+1}: No prompt found")
            continue
        
        print(f"[{time.strftime('%H:%M:%S')}] Generating for example {i+1}/{len(challenges)} (ID: {example_id})...")

        try:
            # Tokenize the prompt
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            
            if len(prompt_tokens) > max_tokens:
                results.append({
                    "challenge_id": example_id,
                    "prompt": prompt_text,
                    "raw": None,
                    "prediction": None
                })
                print(f"Skipped example {i+1} due to length exceeding max tokens.")
                continue
            
            model_input = ModelInput.from_ints(tokens=prompt_tokens)

            sampling_params = SamplingParams(
                max_tokens=max_output_tokens,
                temperature=0,
                seed=42,
                stop=stop_sequences
            )

            output = sampling_client.sample(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=1
            ).result()

            # Decode tokens to text
            tokens = output.sequences[0].tokens
            prediction = tokenizer.decode(tokens)

        except Exception as e:
            print(f"Generation failed for example {i+1}: {e}")
            import traceback
            traceback.print_exc()
            prediction = None

        # Try to extract JSON block from the prediction
        parsed_json = None
        if prediction:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", prediction, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(1))
                    # If JSON contains "output", store only that
                    if isinstance(parsed_json, dict) and "output" in parsed_json:
                        parsed_json = parsed_json["output"]
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON parse failed for example {i+1} ({e})")
                    parsed_json = None

        results.append({
            "challenge_id": example_id,
            "prompt": prompt_text,
            "raw": prediction,
            "prediction": parsed_json
        })

    # Save results
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {save_path}")
    print(f"Prediction complete! Generated {len(results)} predictions.")

    if dataset_type == "Test":
        submissions_path = os.path.join("./data/arc-agi-2025/predictions/submissions.json")
        submissions = {}
        
        for res in results:
            challenge_id_full = res.get("challenge_id")
            prediction = res.get("prediction")
            
            if not challenge_id_full:
                continue
            
            # Split challenge_id into challengeid and tag
            # Format: challengeid_tag
            parts = challenge_id_full.rsplit('_', 1)
            if len(parts) == 2:
                challenge_id, tag = parts
            else:
                # If no underscore, use the whole thing as challenge_id and default tag
                challenge_id = challenge_id_full
                tag = "default"
            
            # Initialize nested structure if needed
            if challenge_id not in submissions:
                submissions[challenge_id] = {}
            
            if tag not in submissions[challenge_id]:
                submissions[challenge_id][tag] = {}
            
            # Add same prediction for both attempts
            submissions[challenge_id][tag]["attempt_1"] = prediction
            submissions[challenge_id][tag]["attempt_2"] = prediction
        
        # Save submissions file
        with open(submissions_path, "w") as f:
            json.dump(submissions, f, indent=4)
        
        print(f"Submissions file saved to {submissions_path}")


    # Compare with solutions if Evaluation dataset
    if solutions_path and os.path.exists(solutions_path):
        print(f"\nEvaluating predictions against solutions...")
        
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

        correct = 0
        total = 0

        for res in results:
            challenge_id = res.get("challenge_id")
            if not challenge_id or challenge_id not in solutions:
                continue
            
            total += 1
            
            try:
                predicted_output = res.get("prediction")
                expected_output = solutions[challenge_id]
                
                if predicted_output == expected_output:
                    correct += 1

            except Exception as e:
                pass  # Silently skip errors

        accuracy = correct / total if total > 0 else 0
        
        # Save evaluation score
        score_path = save_path.replace("_predictions.json", "_scores.json")
        score_data = {
            "dataset": dataset_type,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(score_path, "w") as f:
            json.dump(score_data, f, indent=4)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
        print(f"Score saved to {score_path}")
        print('='*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions for Evaluation and Test')
    parser.add_argument('--dataset', type=str, required=True, choices=['Evaluation', 'Test'],
                        help='Evaluation or Test dataset')
    parser.add_argument('--example_count', type=int, required=False, default=None,
                        help='Number of test samples to predict (optional, defaults to all)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Base model name (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to fine-tuned LoRA checkpoint')
    parser.add_argument('--max_tokens', type=int, default=1000,
                        help='Maximum number of tokens to process per prompt')
    parser.add_argument('--max_output_tokens', type=int, default=1000,
                        help='Maximum number of tokens to generate per prompt')
    
    args = parser.parse_args()
    
    predict_evaluation(
        dataset_type=args.dataset,
        example_count=args.example_count,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        max_tokens=args.max_tokens,
        max_output_tokens=args.max_output_tokens
    )