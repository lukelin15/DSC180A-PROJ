# LLM Model Evaluation

This project evaluates various LLAMA models using different prompting methods on multiple datasets. The evaluation is based on text samples from the `nyt_data` and `yelp_data` datasets.

## Requirements

- Python 3.10.x
- OpenAI Python package
- Matplotlib
- NumPy
- Python-dotenv (for loading environment variables)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lukelin15/DSC180A-PROJ.git
   cd your-repo
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Ensure your `requirements.txt` includes:

   ```
   openai
   matplotlib
   numpy
   python-dotenv
   ```

3. Obtain your SambaNova API key and create a `.env` file in the root directory:

   ```bash
   touch .env
   ```

   Add your API key to the `.env` file:

   ```env
   SAMBANOVA_API_KEY=your_api_key_here
   ```

4. Prepare the datasets:

   - Download and unzip the `nyt_data` and `yelp_data` datasets.
   - Place the datasets inside the `Data` directory:

     ```
     Data/
     ├── nyt_data/
     │   ├── phrase_text.txt
     │   ├── topics_label.txt
     │   └── topics.txt
     └── yelp_data/
         ├── phrase_text.txt
         └── label.txt
     ```

## Configuration

Edit the `llm_config.json` file to modify the models, datasets, and prompting methods used in the evaluation. The default configuration is:

```json
{
    "datasets": [
        "nyt_data",
        "yelp_data"
    ],
    "models": [
        "Meta-Llama-3.2-1B-Instruct",
        "Meta-Llama-3.2-3B-Instruct",
        "Meta-Llama-3.1-8B-Instruct"
    ],
    "model_names_condensed": [
        "LLAMA-1B",
        "LLAMA-3B",
        "LLAMA-8B"
    ],
    "prompting_methods": [
        "direct",
        "chain_of_thought",
        "few_shot"
    ],
    "token_budget": 512,
    "num_samples_per_class": 10
}
```

## Usage

Run the main script to execute the model evaluation:

```bash
python main.py
```