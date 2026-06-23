"""
Example: Sentiment Analysis Validation on News Articles

This example demonstrates the LLM validation framework on a sentiment analysis task
for financial news articles about crypto investing.
"""

import os
import re
import pathlib
from datetime import datetime

# Set working directory to script location
os.chdir(pathlib.Path(__file__).parent.resolve())

from utils.test_backtranslation import test_backtranslation
from utils.test_separation import test_separation
from utils.test_validity import test_validity
from utils.llm_api_calls import llm_completion, llm_embedding
from utils.count_tokens import count_tokens_tiktoken

# Configuration
MODEL = 'anthropic.claude-sonnet-4-5-20250929-v1:0'  # Choose your preferred model
TEMPERATURE = 1
LABELS = ['positive', 'neutral', 'negative']

# Example news article to validate
ORIGINAL_TEXT = """Crypto Investors Celebrated for Most of 2025. Then Came the Hangover. Bitcoin finished the year in the red as investors grappled with the AI trade and macroeconomic risk."""


def simulation_func(label, num_texts):
    """
    Function f: Generates texts from a given sentiment label.

    Args:
        label: One of 'LABELS'
        num_texts: Number of texts to generate per prompt

    Returns:
        List of generated texts
    """
    original_text_length = count_tokens_tiktoken(ORIGINAL_TEXT)

    prompt = f"""
    TASK: Write {num_texts} arbitrary passages from hypothetical Wall Street Journal articles
    with sentiment characterized as {label}.
    The passages should contain the title and subtitle of hypothetical articles around crypto investing.
    The length of each passage should be around {original_text_length} tokens.

    RESPOND WITH ONLY A JSON OBJECT (no markdown, no explanation, no code blocks):

    For successful analysis:
    {{
    "failed": false,
    "text_1": <passage_1>,
    "text_2": <passage_2>,
    ...,
    "text_n": <passage_n>
    }}

    For failed analysis:
    {{
    "failed": true,
    "failure_reason": "<specific reason>"
    }}

    Now generate the passages and provide your JSON response:
    """

    # Call LLM
    response = llm_completion(model=MODEL, temperature=TEMPERATURE, user_prompt=prompt)
    content = response.choices[0].message.content

    # Extract texts from JSON response
    texts = []
    for i in range(num_texts):
        match = re.search(f'"text_{i+1}":\s*"([^"]+)"', content)
        if match:
            texts.append(match.group(1))
        else:
            texts.append(None)

    return texts


def annotation_func(text):
    """
    Function f^{-1}: Annotates a given text with a sentiment label.

    Args:
        text: Text to annotate

    Returns:
        Predicted label (one of LABELS)
    """
    prompt = f"""
    TASK: Read the following passage and classify the sentiment using one of the labels:
    {LABELS}.

    PASSAGE:
    {text}

    RESPOND WITH ONLY A JSON OBJECT (no markdown, no explanation, no code blocks):

    For successful analysis:
    {{
    "classification_failed": false,
    "label": <one of {LABELS}>,
    "explanation": "<1-2 sentence explanation>"
    }}

    For failed analysis:
    {{
    "classification_failed": true,
    "failure_reason": "<specific reason>"
    }}

    Now analyze the passage above and provide your JSON response:
    """

    # Call LLM
    response = llm_completion(model=MODEL, temperature=TEMPERATURE, user_prompt=prompt)
    content = response.choices[0].message.content

    # Extract label from JSON response
    match = re.search(r'"label":\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    else:
        return None


def main():
    """
    Run the complete validation framework on the example news article.
    """
    print("=" * 70)
    print("LLM ANNOTATION VALIDATION FRAMEWORK")
    print("=" * 70)
    print(f"\nOriginal Text: {ORIGINAL_TEXT}\n")

    # Step 1: Get LLM annotation
    print("Step 1: Annotating the text with LLM...")
    label_hat = annotation_func(ORIGINAL_TEXT)
    print(f"   LLM Label: {label_hat}\n")

    if label_hat is None or label_hat not in LABELS:
        print("ERROR: Failed to get valid annotation from LLM")
        return

    # Step 2: Test Backtranslation Property
    print("Step 2: Testing Annotation Backtranslation Property...")
    print("   (Testing if f^{-1}(f(β)) = β)")
    backtranslation_results = test_backtranslation(
        annotation_func=annotation_func,
        simulation_func=simulation_func,
        labels=LABELS,
        tested_labels=label_hat,
        num_sim=20  # Reduced for demo purposes
    )

    accuracy = sum(backtranslation_results["label"] == backtranslation_results["label_hat"]) / len(backtranslation_results)
    print(f"   Backtranslation Accuracy: {accuracy:.2%}")

    if accuracy < 0.90:
        print("   WARNING: Low backtranslation accuracy suggests inconsistent annotation/simulation functions\n")
    else:
        print("   PASS: Annotation backtranslation property satisfied\n")

    # Step 3: Test Separation Property
    print("Step 3: Testing Separation Property...")
    print("   (Testing if texts from different labels are semantically distinct)")

    counterfactual_labels = [l for l in LABELS if l != label_hat]
    simulated_texts = list(backtranslation_results["simulated_text"])

    separation_results, _ = test_separation(
        filename="demo",
        simulation_func=simulation_func,
        label=label_hat,
        counterfactual_labels=counterfactual_labels,
        simulated_text=simulated_texts,
        num_sim_tau=10,  # Reduced for demo purposes
        num_sim_boot=20
    )

    all_separated = (~separation_results['reject']).all()
    print(f"   Separation test results:")
    for _, row in separation_results.iterrows():
        status = "NOT SEPARATED" if row['reject'] else "SEPARATED"
        print(f"      {label_hat} vs {row['label_']}: {status} (τ={row['tau']:.3f}, crit={row['crit']:.3f})")

    if all_separated:
        print("   PASS: Separation property satisfied\n")
    else:
        print("   WARNING: Some labels not well-separated\n")

    # Check prerequisites before running validity test
    prerequisites_passed = (accuracy >= 0.90) and all_separated

    # Step 4: Test Validity (only if prerequisites pass)
    if prerequisites_passed:
        print("Step 4: Testing Annotation Validity...")
        print("   (Testing if simulated text from label_hat is similar to original text)")

        emb_original = llm_embedding(ORIGINAL_TEXT)

        validity_results, _ = test_validity(
            filename="demo",
            simulation_func=simulation_func,
            label_hat=label_hat,
            emb_original=emb_original,
            simulated_text=simulated_texts,
            num_sim_tau=20,
            num_sim_boot=20
        )

        is_valid = not validity_results['reject'][0]
        print(f"   τ = {validity_results['tau'][0]:.3f}")
        print(f"   Critical value = {validity_results['crit'][0]:.3f}")

        if is_valid:
            print("   PASS: Annotation is VALID\n")
        else:
            print("   FAIL: Annotation is NOT VALID\n")

        validity_outcome = 'VALID' if is_valid else 'NOT VALID'
    else:
        print("Step 4: Skipping Validity Test")
        print("   UNDETERMINED\n")
        validity_outcome = 'UNDETERMINED'

    # Final Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original Text: {ORIGINAL_TEXT}")
    print(f"LLM Annotation: {label_hat}")
    print(f"Backtranslation Property: {'PASS' if accuracy >= 0.90 else 'FAIL'}")
    print(f"Separation Property: {'PASS' if all_separated else 'FAIL'}")
    print(f"Validity: {validity_outcome}")
    print("=" * 70)


if __name__ == "__main__":
    main()
