"""
function.py contains the functions to load data from NYT dataset, sample data, define prompting methods, call API for responses and calculate accuracy.
"""

import openai
import random
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()  

API_KEY = os.getenv("SAMBANOVA_API_KEY")
client = openai.OpenAI(api_key=API_KEY, base_url="https://api.sambanova.ai/v1")

def load_data():
    """
    Load data from the NYT dataset and group it by class.

    Returns:
    data_by_class (dict): A dictionary where keys are class names and values are lists of text samples belonging to that class.
    """
    with open(r'Data\NYT-Topics\classes.txt', 'r') as f:
        classes = [line.strip() for line in f]

    with open(r'Data\NYT-Topics\dataset.txt', 'r') as f:
        data = [line.strip().split('\t') for line in f]

    with open(r'Data\NYT-Topics\labels.txt', 'r') as f:
        labels = [line.strip() for line in f]

    data_by_class = {cls: [] for cls in classes}
    for i, entry in enumerate(data):
        text = entry[0]
        cls =  classes[int(labels[i])]
        if cls in data_by_class:
            data_by_class[cls].append(text)

    return data_by_class

def sample_data(data_by_class, num_samples=10):
    """
    Randomly sample entries from each class.

    Parameters:
    data_by_class (dict): A dictionary where keys are class names and values are lists of text samples belonging to that class.
    num_samples (int): Number of samples to take from each class.

    Returns:
    test_samples (list): A list of text samples.
    labels (list): A list of class labels corresponding to the text samples.
    """
    test_samples = []
    labels = []
    for cls, texts in data_by_class.items():
        test_samples.extend(random.sample(texts, min(num_samples, len(texts))))
        labels += [cls] * min(num_samples, len(texts))
    return test_samples, labels

def estimate_tokens(text):
    """
    Estimate the number of tokens in a given text.

    Parameters:
    text (str): Input text.

    Returns:
    token_count (int): Number of tokens in the text.
    """
    return len(text.split())  

def direct_prompt(sample):
    """
    Generate a direct prompt for the given text sample.

    Parameters:
    sample (str): Input text sample.

    Returns:
    prompt (str): Prompt for the text classification task.
    """
    return f"{sample}\nClassify this text."

def evaluate_prompts(test_samples, prompting_method, labels, model_name):
    """
    Call the API for responses and calculate the accuracy of the model.

    Parameters:
    test_samples (list): A list of text samples.
    prompting_method (str): The prompting method to use for classification.
    labels (list): A list of class labels corresponding to the text samples.
    model_name (str): The name of the model to use for classification.

    Returns:
    accuracy (float): The accuracy of the model on the test samples.
    """
    correct_predictions = 0
    total_samples = len(test_samples)
    
    for i, sample in enumerate(test_samples):
        prompt = ""
        
        if prompting_method == "direct":
            prompt = direct_prompt(sample)
        
        token_count = estimate_tokens(prompt)

        if token_count > 512:  
            prompt = ' '.join(prompt.split()[:512])  

        while True:  
            try:
                response = client.chat.completions.create(
                    model=model_name,  
                    messages=[
                        {"role": "system", "content": "You are trying to classify a text. Your options are: [business, politics, sports, health, education, estate, arts, science, technology]. Just type the class name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    top_p=0.1
                )

                predicted_class = response.choices[0].message.content.strip()
                actual_label = labels[i] 

                if predicted_class == actual_label:
                    correct_predictions += 1

                time.sleep(1)  # Wait for 1 second before processing the next sample

            except Exception as e:
                print(f"Error: {e}")
                print("Rate limit exceeded. Retrying after a short pause...")
                time.sleep(5)  # Wait for 5 seconds before retrying

    accuracy = correct_predictions / total_samples
    return accuracy
