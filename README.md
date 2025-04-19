# Fine-Tuning GPT-2 for Multi-Label Text Classification
This project demonstrates how to fine-tune **GPT-2** using the **Hugging Face Transformers** library for a **multi-label text classification** task. It includes preprocessing, tokenization, training, and evaluation pipelines designed to optimize model performance on custom datasets.

## Project Highlights
- Fine-tuned a pre-trained GPT-2 model on custom multi-label datasets
- Implemented advanced text preprocessing and tokenization
- Achieved **90% precision** across multiple classes
- Designed a robust evaluation pipeline to monitor performance per label

## Tools & Libraries
- **Python**
- **Hugging Face Transformers**
- **PyTorch**
- **scikit-learn**
- **Pandas**, **NumPy**, **Matplotlib**
- *(Optional)* Google Colab / Jupyter Notebooks for experimentation
  
## Project Structure
gpt2-multilabel/ ├── dataset.csv # Custom dataset (multi-label format) ├── gpt2_finetune.py # Fine-tuning script ├── evaluation.py # Evaluation logic ├── utils.py # Preprocessing/tokenization functions ├── README.md # Project documentation └── requirements.txt # Dependencies

## How It Works

1. **Data Preprocessing**  
   Clean, format, and tokenize multi-label text data for compatibility with GPT-2.
2. **Model Training**  
   Fine-tune GPT-2 using `Trainer` API from Hugging Face on the labeled dataset.
3. **Evaluation**  
   Generate precision, recall, F1-score per class using `scikit-learn` metrics.

## Example Results

- **Precision**: 90%
- **Recall**: ~87%
- **F1-Score**: ~88%
- Supports multiple overlapping labels per input text.

## Future Improvements

- Use GPT-Neo or DistilGPT for faster training.
- Add dataset augmentation for low-resource labels.
- Convert into a Gradio or Streamlit app for demo.

## Installation

```bash
pip install transformers datasets torch scikit-learn pandas matplotlib




