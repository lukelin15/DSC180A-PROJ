#!/usr/bin/env python

import json
import matplotlib.pyplot as plt
import numpy as np
from functions import load_data, sample_data, evaluate_prompts

def main():
    with open('llm_config.json', 'r') as f:
        config = json.load(f)

    data_by_class = load_data()
    test_samples, labels = sample_data(data_by_class, num_samples=2)  

    models = config['models']
    model_name_condensed = config['model_name_condensed']
    prompting_methods = config['prompting_methods']
    
    # Initialize a dictionary to hold results
    results = {model: [] for model in models}

    # Evaluate Each Model 
    for model in models:
        for method in prompting_methods:
            accuracy = evaluate_prompts(test_samples, method, labels, model)
            results[model].append(accuracy)
            print(f"Accuracy for {model} with {method} prompting: {accuracy:.2%}")

    bar_width = 0.25  
    x = np.arange(len(models))  

    for method_index, method in enumerate(prompting_methods):
        accuracies = [results[model][method_index] for model in models]
        plt.bar(x + method_index * bar_width, accuracies, width=bar_width, label=method)

    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(x, model_name_condensed)  
    plt.ylim(0, 1)  
    plt.legend()
    plt.grid(axis='y')  
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
