# Fine-tuned RAG with Local Embeddings, Qdrant, and OpenAI LLM

This project implements a Retrieval-Augmented Generation (RAG) system using fine-tuned local embeddings, Qdrant vector storage, GPT-4o.

## Installation

1. Clone the repository and navigate to the project folder:
   ```
   git clone https://github.com/s-smits/RAG-finetuned-embeddings
   cd RAG-finetuned-embeddings
   ```

2. Create a virtual environment named `venv_RAG_finetuned_embeddings` and activate it:
   - For macOS and Linux:
     ```
     python3 -m venv venv_RAG_finetuned_embeddings
     source venv_RAG_finetuned_embeddings/bin/activate
     ```
   - For Windows:
     ```
     python -m venv venv_RAG_finetuned_embeddings
     venv_RAG_finetuned_embeddings\Scripts\activate
     ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Rename `.env.example` to `.env` in the project's root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

   Make sure to enter your own OpenAI API key here. This key is necessary for using GPT-4 in this project.

## Usage

Start the script:
```
   python rag_finetune.py
```

This opens a Gradio interface where you can:
1. Upload PDF files
2. Process the PDFs
3. Start fine-tuning the embeddings
4. Ask questions based on the processed documents while using the fine-tuned embeddings

## Requirements

See `requirements.txt` for a full list of dependencies.

## License

[MIT License](LICENSE)

## Demo
<video src="https://github.com/user-attachments/assets/b854534c-fcd4-44af-b907-94c309552709" controls="controls" style="max-width: 730px;">
</video>
