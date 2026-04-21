import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_eval_prompt(correct: str, predicted: str) -> str:
    return f"""You are an expert evaluator for a Visual Question Answering (VQA) task.
Your task is to compare a predicted answer against the ground-truth answer.

Ground-truth answer: {correct}
Predicted answer: {predicted}

Does the predicted answer have the same meaning as the ground-truth answer?
First, answer with 'YES' or 'NO'. 
Second, provide a semantic similarity score from 0 to 5, where 5 is a perfect match and 0 is completely wrong.

Format your response EXACTLY like this:
MATCH: [YES/NO]
SCORE: [0-5]"""

def parse_response(response: str) -> tuple[bool, int]:
    is_match = False
    score = 0
    
    match_search = re.search(r"MATCH:\s*(YES|NO)", response, re.IGNORECASE)
    if match_search:
        is_match = match_search.group(1).upper() == "YES"
        
    score_search = re.search(r"SCORE:\s*([0-5])", response)
    if score_search:
        score = int(score_search.group(1))
        
    return is_match, score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions using an open-source LLM as a judge.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the baseline predictions CSV.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Open source LLM to use as judge.")
    return parser.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} predictions from {csv_path}")
    
    # Load LLM
    logger.info(f"Loading evaluator model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    llm_correct = []
    llm_scores = []
    
    logger.info("Evaluating predictions...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        correct_ans = str(row.get("correct", ""))
        predicted_ans = str(row.get("predicted", ""))
        
        prompt = build_eval_prompt(correct_ans, predicted_ans)
        messages = [
            {"role": "system", "content": "You are a strict and objective evaluator."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        is_match, score = parse_response(response)
        llm_correct.append(int(is_match))
        llm_scores.append(score)
        
    df["llm_is_correct"] = llm_correct
    df["llm_score"] = llm_scores
    
    output_path = csv_path.parent / f"{csv_path.stem}_llm_evaluated.csv"
    df.to_csv(output_path, index=False)
    
    accuracy = sum(llm_correct) / len(llm_correct) * 100
    avg_score = sum(llm_scores) / len(llm_scores)
    
    logger.info(f"Evaluation complete!")
    logger.info(f"LLM Judge Accuracy: {accuracy:.2f}%")
    logger.info(f"LLM Judge Average Score: {avg_score:.2f} / 5.0")
    logger.info(f"Saved evaluated results to {output_path}")

if __name__ == "__main__":
    main()
