import argparse
import json
import os
import time
import dotenv
import tinker
from tinker.types import SamplingParams, ModelInput
import re

dotenv.load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LoRA model using Tinker.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Base model name (e.g., Qwen/Qwen3-30B-A3B-Instruct-2507)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to fine-tuned LoRA checkpoint"
    )
    parser.add_argument(
        "--eval_prompts_path",
        type=str,
        required=True,
        help="Path to JSON file containing evaluation prompts"
    )
    parser.add_argument(
        "--eval_solutions_path",
        type=str,
        required=False,
        help="Path to JSON file containing evaluation solutions (optional)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./eval_results.json",
        help="Path to save the evaluation results"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens to process per prompt"
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=1000,
        help="Maximum number of tokens to generate per prompt"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY must be set in the environment.")

    # Initialize service client
    service_client = tinker.ServiceClient(api_key=api_key)

    training_client = service_client.create_lora_training_client(
    base_model=args.model_name
)

    # Create a sampling client using the saved checkpoint path
    sampling_client = service_client.create_sampling_client(
        model_path=args.checkpoint_path
    )

    # Get tokenizer directly from the sampling client!
    tokenizer = training_client.get_tokenizer()

    # Load evaluation prompts
    with open(args.eval_prompts_path, "r") as f:
        eval_prompts = json.load(f)

    
    results = []
    
    # Stop sequences for JSON output
    stop_sequences = [ "]\n]\n}\n```"]

    for i, example in enumerate(eval_prompts):
        prompt_text = example.get("prompt") or example.get("input")
        example_id = example.get("challenge_id")
        if not prompt_text:
            continue
        
        if(i+1==8):
            break

        print(f"[{time.strftime('%H:%M:%S')}] Generating for example {i+1}/{len(eval_prompts)}...")

        try:
            # Tokenize the prompt
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            if len(prompt_tokens)>args.max_tokens:
                results.append({
                "challenge_id": example_id,
                "prompt": prompt_text,
                "prediction": None
                })
                print(f"Skipped example {i} due to length exceeding max tokens.")
                continue
            model_input = ModelInput.from_ints(tokens=prompt_tokens)

            sampling_params = SamplingParams(
                max_tokens=args.max_output_tokens,
                temperature=0.0,
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
            print(f"Generation failed for example {i}: {e}")
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
                    print(f"Warning: JSON parse failed for example {i} ({e})")
                    parsed_json = None

        results.append({
            "challenge_id": example_id,
            "prompt": prompt_text,
            "prediction": parsed_json
        })


    # Save results
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {args.output_path}")

    # Compare with solutions if provided
    if args.eval_solutions_path:
        with open(args.eval_solutions_path, "r") as f:
            eval_solutions = json.load(f)

        correct = 0
        total = 0

        for res in results:
            key = res.get("challenge_id")
            if not key or key not in eval_solutions:
                continue
            
            total += 1
            
            # Parse JSON predictions and compare
            try:
                predicted_output = res.get("prediction")
                expected_output = eval_solutions[key]
                if predicted_output == expected_output:
                    correct += 1
                    print(f"✓ {key}")
                else:
                    print(f"✗ {key} (Expected: {expected_output}, Got: {predicted_output})")

            except json.JSONDecodeError as e:
                print(f"✗ {key} (JSON parse error: {e})")
            except Exception as e:
                print(f"✗ {key} (Error: {e})")

        accuracy = correct / total if total > 0 else 0
        print(f"\nEvaluation Accuracy: {accuracy*100:.2f}% ({correct}/{total})")

    print(f"Evaluation complete!")


if __name__ == "__main__":
    main()