import json
import matplotlib.pyplot as plt
import numpy as np
import os
from functions import load_data, sample_data, evaluate_prompts

def main():
    # Load configuration
    with open('llm_config.json', 'r') as f:
        config = json.load(f)

    datasets = config['datasets']
    models = config['models']
    model_names_condensed = config['model_names_condensed']
    prompting_methods = config['prompting_methods']
    token_budget = config['token_budget']
    num_samples_per_class = config['num_samples_per_class']

    overall_results = {}

    for dataset_name in datasets:
        print(f"\nEvaluating on dataset: {dataset_name}")
        data_by_class = load_data(dataset_name)
        test_samples, labels = sample_data(data_by_class, num_samples=num_samples_per_class)

        # Hold results for this dataset
        results = {model: [] for model in models}

        for model in models:
            for method in prompting_methods:
                accuracy = evaluate_prompts(
                    test_samples,
                    method,
                    labels,
                    model,
                    data_by_class,
                    token_budget
                )
                results[model].append(accuracy)
                print(f"Accuracy for {model} with {method} prompting: {accuracy:.2%}")

        overall_results[dataset_name] = results

        # Plotting the results for this dataset
        bar_width = 0.25
        x = np.arange(len(models))

        plt.figure(figsize=(10, 6))
        for method_index, method in enumerate(prompting_methods):
            accuracies = [results[model][method_index] for model in models]
            plt.bar(
                x + method_index * bar_width,
                accuracies,
                width=bar_width,
                label=method
            )

        plt.title(f'Model Accuracy Comparison on {dataset_name}')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(x + bar_width, model_names_condensed)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()