import openai
import random
import time
import os
import json
import numpy as np

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("SAMBANOVA_API_KEY")
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.sambanova.ai/v1")

def load_data(dataset_name):
    """
    Load data from the NYT dataset and group it by class.

    Args:
    dataset_name (str): The name of the dataset to load

    Returns:
    data_by_class (dict): A dictionary with class names as keys and a list of texts as values
    """
    data_by_class = {}
    if dataset_name == 'nyt_data':
        # Load NYT data
        with open('Data/nyt_data/phrase_text.txt', 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f]

        with open('Data/nyt_data/topics_label.txt', 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f]

        with open('Data/nyt_data/topics.txt', 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f]

        class_dict = {str(i): cls for i, cls in enumerate(classes)}
        for text, label in zip(data, labels):
            cls = class_dict.get(label, 'Unknown')
            if cls not in data_by_class:
                data_by_class[cls] = []
            data_by_class[cls].append(text)

    elif dataset_name == 'yelp_data':
        # Load Yelp data
        with open('Data/yelp_data/phrase_text.txt', 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f]

        with open('Data/yelp_data/food_label.txt', 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f]

        with open('Data/yelp_data/food.txt', 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f]

        class_dict = {str(i): cls for i, cls in enumerate(classes)}
        for text, label in zip(data, labels):
            cls = class_dict.get(label, 'Unknown')
            if cls not in data_by_class:
                data_by_class[cls] = []
            data_by_class[cls].append(text)


    return data_by_class

def sample_data(data_by_class, num_samples=10):
    """
    Randomly sample entries from each class in the dataset.

    Args:
    data_by_class (dict): A dictionary with class names as keys and a list of texts as values
    num_samples (int): The number of samples to take from each class

    Returns:
    test_samples (list): A list of sampled texts
    """
    test_samples = []
    test_labels = []
    for cls, texts in data_by_class.items():
        sampled_texts = random.sample(texts, min(num_samples, len(texts)))
        test_samples.extend(sampled_texts)
        test_labels += [cls] * len(sampled_texts)
    return test_samples, test_labels

def estimate_tokens(text):
    """
    Estimate the number of tokens in a given text.

    Args:
    text (str): The text to estimate the token count for

    Returns:
    token_count (int): The estimated number of tokens
    """
    return len(text.split())

def direct_prompt(sample):
    """
    Direct prompting method to classify a text.

    Args:
    sample (str): The text sample to classify

    Returns:
    prompt (str): The prompt for the text classification
    """
    return f"{sample}\nClassify this text."

def chain_of_thought_prompt(sample):
    """
    Chain of thought prompting method to classify a text.

    Args:
    sample (str): The text sample to classify

    Returns:
    prompt (str): The prompt for the text classification
    """
    return f"{sample}\nThink step by step and explain your reasoning before classifying this text."

def few_shot_prompt(sample, few_shot_examples):
    """
    Few-shot prompting method to classify a text.

    Args:
    sample (str): The text sample to classify
    few_shot_examples (list): A list of few-shot examples

    Returns:
    prompt (str): The prompt for the text classification
    """
    examples_text = "\n".join(
        [f"Text: {ex['text']}\nClass: {ex['class']}" for ex in few_shot_examples]
    )
    return f"{examples_text}\n\nNow, classify the following text:\n{sample}"

def evaluate_prompts(test_samples, prompting_method, labels, model_name, data_by_class, token_budget=512):
    """
    Evaluate the accuracy of a model with different prompting methods.

    Args:
    test_samples (list): A list of text samples to classify
    prompting_method (str): The prompting method to use
    labels (list): The ground truth labels for the test samples
    model_name (str): The name of the model to evaluate
    data_by_class (dict): A dictionary with class names as keys and a list of texts as values
    token_budget (int): The token budget for the model

    Returns:
    accuracy (float): The accuracy of the model on the test samples
    """
    correct_predictions = 0
    total_samples = len(test_samples)

    for i, sample in enumerate(test_samples):
        prompt = ""

        if prompting_method == "direct":
            prompt = direct_prompt(sample)
        elif prompting_method == "chain_of_thought":
            prompt = chain_of_thought_prompt(sample)
        elif prompting_method == "few_shot":
            few_shot_examples = []
            # Collect few-shot examples from other samples
            for cls in data_by_class.keys():
                examples = [text for text in data_by_class[cls] if text != sample]
                if len(examples) > 0:
                    ex_text = random.choice(examples)
                    few_shot_examples.append({'text': ex_text, 'class': cls})
            prompt = few_shot_prompt(sample, few_shot_examples)

        # Estimate tokens and truncate if necessary
        token_count = estimate_tokens(prompt)

        if token_count > token_budget:
            prompt = ' '.join(prompt.split()[:token_budget])

        while True:
            try:
                system_message = f"You are classifying texts into the following categories: {list(data_by_class.keys())}. Just provide the class name."
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    top_p=0.1
                )

                predicted_class = response.choices[0].message.content.strip()
                actual_label = labels[i]

                if predicted_class.lower() == actual_label.lower():
                    correct_predictions += 1

                time.sleep(1)  # Avoid rate limits
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying after a short pause...")
                time.sleep(5)  # Wait before retrying

    accuracy = correct_predictions / total_samples
    return accuracy