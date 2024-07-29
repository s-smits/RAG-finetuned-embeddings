# rag_agent.py

from dotenv import load_dotenv
import os
import torch
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
# from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
import keras
from keras.layers import Input
import traceback
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import bm25s
import Stemmer
from langchain_community.vectorstores import Qdrant as LangchainQdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import nltk
import asyncio
import time
import logging
from typing import List, Dict, Any

nltk.download('punkt', quiet=True)

# Global variables
qdrant_collection = None
base_qdrant_collection = None
fine_tuned_embeddings = None
all_documents = []
txt_files = []
is_fine_tuned = False
bm25_collection = None
base_embeddings = None  # Add this line
fine_tuned_embeddings = None  # Add this line if not already present

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() for page in reader.pages)

class AlignedHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name, documents, n_trials=10, optuna_callback=None, **kwargs):
        try:
            print("Initializing AlignedHuggingFaceEmbeddings")
            self.base_embeddings = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
            self.fine_tuned_model, self.study = self.finetune_embeddings(documents, n_trials=n_trials, optuna_callback=optuna_callback)
            print("AlignedHuggingFaceEmbeddings initialized successfully")
        except Exception as e:
            print(f"Error in AlignedHuggingFaceEmbeddings initialization: {str(e)}")
            raise

    def finetune_embeddings(self, chunked_documents, epochs=5, n_trials=10, optuna_callback=None):
        try:
            print("Starting finetune_embeddings with Optuna")
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            base_model = SentenceTransformer(self.base_embeddings.model_name).to(device)
            
            # Extract text from all chunked documents
            texts = [chunk["text"] for chunk in chunked_documents]
            base_embeddings = base_model.encode(texts)
            
            print(f"Number of chunks: {len(texts)}")
            
            def calculate_metric(original, fine_tuned):
                similarities = cosine_similarity(original, fine_tuned)
                return np.mean(similarities.diagonal())
            
            best_model = None
            best_metric = 0

            def objective(trial: Trial):
                nonlocal best_model, best_metric
                
                # Start with 3e-4 and allow exploration up to 1e-2
                learning_rate = trial.suggest_float("learning_rate", 3e-4, 1e-2, log=True)
                
                if best_model is None:
                    model = keras.Sequential([
                        Input(shape=(base_model.get_sentence_embedding_dimension(),)),
                        keras.layers.Dense(base_model.get_sentence_embedding_dimension(), activation="tanh")
                    ])
                else:
                    model = keras.models.clone_model(best_model)
                    model.set_weights(best_model.get_weights())
                
                optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss="mse")
                
                for epoch in range(epochs):
                    for embedding in base_embeddings:
                        chunk_embedding = embedding.reshape(1, -1)
                        model.fit(chunk_embedding, chunk_embedding, epochs=1, verbose=0)
                    
                    fine_tuned_embeddings = model.predict(base_embeddings)
                    metric = calculate_metric(base_embeddings, fine_tuned_embeddings)
                    
                    trial.report(metric, epoch)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                    
                    if metric > best_metric:
                        best_metric = metric
                        best_model = keras.models.clone_model(model)
                        best_model.set_weights(model.get_weights())
                
                return metric

            # Use TPESampler with a fixed random seed for reproducibility
            sampler = TPESampler(seed=42)
            # Bayesian Optimization
            study = optuna.create_study(
                direction="maximize", 
                sampler=sampler, 
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            )
            
            # Ensure the first trial uses the starting learning rate of 3e-4
            study.enqueue_trial({"learning_rate": 3e-4})
            
            study.optimize(objective, n_trials=n_trials, callbacks=[optuna_callback] if optuna_callback else None)
            
            best_lr = study.best_params["learning_rate"]
            print(f"Best learning rate: {best_lr}")
            
            # Use the best model from trials
            final_model = best_model
            
            print("finetune_embeddings completed successfully")
            return final_model, study
        except Exception as e:
            print(f"Error in finetune_embeddings: {str(e)}")
            raise

    def embed_documents(self, texts):
        try:
            print("Starting embed_documents")
            base_embeddings = self.base_embeddings.embed_documents(texts)
            result = self.fine_tuned_model.predict(np.array(base_embeddings))
            print("embed_documents completed successfully")
            return result.tolist()  # Convert numpy array to list
        except Exception as e:
            print(f"Error in embed_documents: {str(e)}")
            raise

    def embed_query(self, text):
        try:
            print("Starting embed_query")
            base_embedding = self.base_embeddings.embed_query(text)
            result = self.fine_tuned_model.predict(np.array([base_embedding]))[0]
            print("embed_query completed successfully")
            return result.tolist()  # Convert numpy array to list
        except Exception as e:
            print(f"Error in embed_query: {str(e)}")
            raise

def initialize_bm25s(documents):
    # Create a stemmer
    stemmer = Stemmer.Stemmer("english")
    
    # Tokenize the corpus
    corpus = [doc["text"] for doc in documents]
    corpus_tokens = [bm25s.tokenize(doc, stopwords="en", stemmer=stemmer) for doc in corpus]
    
    # Ensure corpus_tokens is a list of lists of strings
    corpus_tokens = [[str(token) for token in doc] for doc in corpus_tokens]
    
    # Create and index the BM25 model
    bm25_collection = bm25s.BM25()
    bm25_collection.index(corpus_tokens)
    
    return bm25_collection

def process_uploaded_pdfs(pdf_files, progress=gr.Progress()):
    global all_documents, txt_files, is_fine_tuned, bm25_collection
    
    try:
        print("Starting process_uploaded_pdfs")
        if not pdf_files:
            return "No PDF files uploaded. Please upload PDF files and try again.", gr.update(interactive=False)

        all_documents = []
        txt_files = []
        is_fine_tuned = False  # Reset fine-tuning status

        total_files = len(pdf_files)
        for i, pdf_file in enumerate(pdf_files):
            progress((i + 1) / total_files, f"Processing PDF {i + 1}/{total_files}")
            
            try:
                file_name = os.path.basename(pdf_file.name)
                txt_file_name = os.path.splitext(file_name)[0] + ".txt"
                txt_files.append(txt_file_name)

                extracted_text = extract_text_from_pdf(pdf_file.name)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=512,
                    chunk_overlap=100,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                # Splitting the extracted text into multiple chunks
                chunks = text_splitter.split_text(extracted_text)
                
                # Adding each chunk as a separate document in the all_documents list
                all_documents.extend([{
                    "text": chunk,
                    "metadata": {"source": txt_file_name}
                } for chunk in chunks])

            except Exception as file_error:
                print(f"Error processing file {file_name}: {str(file_error)}")
                continue  # Skip to the next file if there's an error

        progress(0.9, "Initializing BM25")
        # Initialize bm25s after processing all documents
        try:
            bm25_collection = initialize_bm25s(all_documents)
        except Exception as bm25_error:
            print(f"Error initializing BM25: {str(bm25_error)}")
            # Continue without BM25 if there's an error

        progress(1.0, "Processing complete")
        print("process_uploaded_pdfs completed successfully")
        return f"Processed {len(pdf_files)} PDFs. Created {len(all_documents)} chunks. Click 'Start Fine-tuning' to continue.", gr.update(interactive=True)
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in process_uploaded_pdfs: {error_details}")
        return f"Error processing PDFs: {str(e)}\n\nFull error details:\n{error_details}", gr.update(interactive=False)

def create_qdrant_collection(embeddings, documents, collection_name):
    client = QdrantClient(":memory:")
    
    # Get the embedding dimension
    test_embedding = embeddings.embed_query("test")
    embedding_size = len(test_embedding) if isinstance(test_embedding, list) else test_embedding.shape[0]
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )
    
    qdrant_documents = [
        Document(page_content=doc["text"], metadata=doc["metadata"])
        for doc in documents
    ]
    
    langchain_qdrant = LangchainQdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    langchain_qdrant.add_documents(qdrant_documents)
    return langchain_qdrant

def generate_answer(question, context, llm):
    prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

    return llm.invoke(prompt).content

def evaluate_embeddings(base_embeddings, fine_tuned_embeddings, test_documents, test_questions, llm, progress=gr.Progress()):
    # Create Qdrant collections
    base_qdrant = create_qdrant_collection(base_embeddings, test_documents, "base_collection")
    fine_tuned_qdrant = create_qdrant_collection(fine_tuned_embeddings, test_documents, "fine_tuned_collection")
    
    base_results = []
    fine_tuned_results = []
    
    # Add tqdm progress bar for the evaluation process
    for i, (question, relevant_doc) in enumerate(test_questions):
        progress((i + 1) / len(test_questions), f"Evaluating embeddings: {i + 1}/{len(test_questions)}")
        
        # Base embeddings
        base_context = "\n\n".join([doc.page_content for doc in base_qdrant.similarity_search(question, k=3)])
        base_answer = generate_answer(question, base_context, llm)
        
        # Fine-tuned embeddings
        fine_tuned_context = "\n\n".join([doc.page_content for doc in fine_tuned_qdrant.similarity_search(question, k=3)])
        fine_tuned_answer = generate_answer(question, fine_tuned_context, llm)
        
        base_results.append({
            "question": question,
            "context": base_context,
            "answer": base_answer,
            "relevant_doc": relevant_doc['text']
        })
        
        fine_tuned_results.append({
            "question": question,
            "context": fine_tuned_context,
            "answer": fine_tuned_answer,
            "relevant_doc": relevant_doc['text']
        })
    
    return base_results, fine_tuned_results

def calculate_metrics(results):
    rouge = Rouge()
    bleu_scores = []
    relevant_context_count = 0
    
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for item in results:
        reference = item['relevant_doc']
        hypothesis = item['answer']
        
        # ROUGE scores
        rouge_scores = rouge.get_scores(hypothesis, reference)[0]
        rouge_1_scores.append(rouge_scores['rouge-1']['f'])
        rouge_2_scores.append(rouge_scores['rouge-2']['f'])
        rouge_l_scores.append(rouge_scores['rouge-l']['f'])
        
        # BLEU score
        reference_tokens = nltk.word_tokenize(reference.lower())
        hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
        bleu_score = sentence_bleu([reference_tokens], hypothesis_tokens)
        bleu_scores.append(bleu_score)
        
        # Check if relevant document is in the context
        if item['relevant_doc'] in item['context']:
            relevant_context_count += 1
    
    avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0
    avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    retrieval_accuracy = relevant_context_count / len(results) if results else 0
    
    return {
        "rouge-1": avg_rouge_1,
        "rouge-2": avg_rouge_2,
        "rouge-l": avg_rouge_l,
        "bleu": avg_bleu,
        "retrieval_accuracy": retrieval_accuracy
    }

def compare_embeddings(base_embeddings, fine_tuned_embeddings, test_documents, llm, progress=gr.Progress()):
    test_questions = generate_questions(test_documents, num_questions=10)
    
    base_results, fine_tuned_results = evaluate_embeddings(
        base_embeddings, fine_tuned_embeddings, test_documents, test_questions, llm, progress
    )
    
    base_metrics = calculate_metrics(base_results)
    fine_tuned_metrics = calculate_metrics(fine_tuned_results)
    
    return {
        "base_embeddings": base_metrics,
        "fine_tuned_embeddings": fine_tuned_metrics
    }

def generate_questions(documents, num_questions=10):
    questions = []
    for doc in documents[:num_questions]:
        text = doc['text']
        sentences = nltk.sent_tokenize(text)
        if sentences:
            first_sentence = sentences[0]
            words = first_sentence.split()
            if len(words) > 5:
                subject = ' '.join(words[:3])
                question = f"What does the text say about {subject}?"
            else:
                question = f"What is the main topic of this text?"
            questions.append((question, doc))
    return questions

class HybridRetriever:
    def __init__(self):
        self.bm25_collection = None
        self.base_qdrant_collection = None
        self.qdrant_collection = None
        self.base_embeddings = None
        self.fine_tuned_embeddings = None
        self.all_documents = []

    def initialize(self, all_documents, base_embeddings, fine_tuned_embeddings, bm25_collection):
        self.all_documents = all_documents
        self.base_embeddings = base_embeddings
        self.fine_tuned_embeddings = fine_tuned_embeddings
        self.bm25_collection = bm25_collection

        self.base_qdrant_collection = create_qdrant_collection(
            base_embeddings,
            all_documents,
            "base_pdf_collection"
        )

        self.qdrant_collection = create_qdrant_collection(
            fine_tuned_embeddings,
            all_documents,
            "fine_tuned_pdf_collection"
        )

    def bm25_retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        stemmer = Stemmer.Stemmer("english")
        tokenized_query = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)
        
        results, _ = self.bm25_collection.retrieve(tokenized_query, k=k)
        
        # results is a 2D array, we need to flatten it
        top_k_indices = results.flatten()
        
        # Return the documents corresponding to the top-k indices
        return [self.all_documents[i] for i in top_k_indices]

    async def hybrid_retrieve(self, question: str, embeddings, k: int = 5) -> List[Dict[str, Any]]:
        try:
            logging.info(f"Starting hybrid retrieval for question: {question}")
            
            # BM25 retrieval
            bm25_results = self.bm25_retrieve(question, k)
            
            # Vector retrieval
            if embeddings == self.base_embeddings:
                vector_results = self.base_qdrant_collection.similarity_search(question, k=k)
            elif embeddings == self.fine_tuned_embeddings:
                vector_results = self.qdrant_collection.similarity_search(question, k=k)
            else:
                raise ValueError("Invalid embeddings provided")
            
            # Combine results (simple approach: concatenate and deduplicate)
            combined_results = bm25_results + [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in vector_results
            ]
            
            # Deduplicate based on text content
            seen = set()
            unique_results = []
            for result in combined_results:
                if result["text"] not in seen:
                    seen.add(result["text"])
                    unique_results.append(result)
            
            # Sort results by relevance (you may need to implement a scoring mechanism)
            # For now, we'll assume the order of appearance is the relevance order
            sorted_results = unique_results
            
            logging.info(f"Hybrid retrieval completed, returning {len(sorted_results)} results")
            return sorted_results[:k]
        except Exception as e:
            logging.error(f"Error in hybrid_retrieve: {str(e)}", exc_info=True)
            if isinstance(e, ValueError):
                logging.error("Invalid embeddings provided for vector retrieval")
            elif isinstance(e, AttributeError):
                logging.error("BM25 or vector retrieval method not found")
            else:
                logging.error("Unexpected error occurred during hybrid retrieval")
            return []

# Create a global instance of HybridRetriever
hybrid_retriever = HybridRetriever()

def start_fine_tuning(num_trials, progress=gr.Progress()):
    global hybrid_retriever, fine_tuned_embeddings, is_fine_tuned, bm25_collection, base_embeddings

    try:
        print("Starting start_fine_tuning")
        progress(0.0, desc="Initializing fine-tuning process")
        
        model_name = "all-MiniLM-L6-v2"
        local_model_path = os.path.join(os.getcwd(), "local_models", model_name)

        progress(0.05, desc="Loading model")
        if not os.path.exists(local_model_path):
            SentenceTransformer(model_name).save(local_model_path)

        # Capture Optuna's output
        optuna_output = []

        def optuna_callback(study, trial):
            if trial.state == optuna.trial.TrialState.PRUNED:
                optuna_output.append(f"Trial {trial.number} pruned.")
            elif trial.state == optuna.trial.TrialState.COMPLETE:
                optuna_output.append(f"Trial {trial.number}: LR = {trial.params['learning_rate']:.6f}, Best Metric = {trial.value:.6f}")
            
            # Add more detailed information about the optimization process
            best_trial = study.best_trial
            optuna_output.append(f"Best trial so far: {best_trial.number}")
            optuna_output.append(f"Best value: {best_trial.value:.6f}")
            optuna_output.append(f"Best params: {best_trial.params}")
            
            # Adjust progress calculation
            optimization_progress = 0.1 + (0.5 * (trial.number + 1) / num_trials)
            progress(optimization_progress, desc=f"Optimizing embeddings: Trial {trial.number + 1}/{num_trials}")
            
            return "\n".join(optuna_output)

        progress(0.1, desc="Starting optimization")
        fine_tuned_embeddings = AlignedHuggingFaceEmbeddings(
            model_name=local_model_path,
            documents=all_documents,
            n_trials=num_trials,
            model_kwargs={"device": "cpu"},
            optuna_callback=optuna_callback
        )

        progress(0.60, desc="Creating base embeddings")
        base_embeddings = HuggingFaceEmbeddings(model_name=local_model_path, model_kwargs={"device": "cpu"})

        progress(0.65, desc="Initializing hybrid retriever")
        hybrid_retriever.initialize(all_documents, base_embeddings, fine_tuned_embeddings, bm25_collection)

        # Evaluate embeddings
        progress(0.80, desc="Evaluating embeddings")
        evaluation_results = compare_embeddings(base_embeddings, fine_tuned_embeddings, all_documents, llm, progress)

        is_fine_tuned = True
        progress(1.0, desc="Fine-tuning, collection creation, and evaluation complete")
        
        optuna_summary = "\n".join(optuna_output)
        best_model = f"**Model {fine_tuned_embeddings.study.best_trial.number + 1} (LR: {fine_tuned_embeddings.study.best_params['learning_rate']:.6f})**"
        final_message = f"Fine-tuning complete. You can now ask questions.\n\n"
        final_message += f"Optuna Optimization Summary:\n{optuna_summary}\n\n"
        final_message += f"Chosen model for best results: {best_model}\n"
        final_message += f"Best Metric (Cosine Similarity): {fine_tuned_embeddings.study.best_trial.value:.6f}\n\n"
        final_message += f"Evaluation Results:\n{evaluation_results}"
        
        print("start_fine_tuning completed successfully")
        return final_message, gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in start_fine_tuning: {error_details}")
        return f"Error during fine-tuning: {str(e)}\n\nFull error details:\n{error_details}", gr.update(interactive=False), gr.update(interactive=False)

async def generate_response_async(question, include_raw_text=False):
    global hybrid_retriever, fine_tuned_embeddings, base_embeddings, is_fine_tuned

    try:
        logging.info("Starting generate_response_async")
        if not is_fine_tuned:
            yield {"Error": "Please complete the fine-tuning process before asking questions."}
            return
        
        if base_embeddings is None or fine_tuned_embeddings is None:
            yield {"Error": "Embeddings are not initialized. Please run the fine-tuning process first."}
            return

        start_time = time.time()
        
        # Hybrid retrieval (both base and fine-tuned)
        base_hybrid_results = await hybrid_retriever.hybrid_retrieve(question, base_embeddings)
        ft_hybrid_results = await hybrid_retriever.hybrid_retrieve(question, fine_tuned_embeddings)
        
        # Normal retrieval (both base and fine-tuned)
        base_results = hybrid_retriever.base_qdrant_collection.similarity_search(question, k=5)
        ft_results = hybrid_retriever.qdrant_collection.similarity_search(question, k=5)
        
        # Generate contexts
        base_context = "\n\n".join(doc.page_content for doc in base_results)
        ft_context = "\n\n".join(doc.page_content for doc in ft_results)
        base_hybrid_context = "\n\n".join(doc["text"] for doc in base_hybrid_results)
        ft_hybrid_context = "\n\n".join(doc["text"] for doc in ft_hybrid_results)
        
        # Generate responses
        responses = {}
        for model_type, context in [
            ("Base", base_context),
            ("Fine-tuned", ft_context),
            ("Base Hybrid", base_hybrid_context),
            ("Fine-tuned Hybrid", ft_hybrid_context)
        ]:
            prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
            response = await asyncio.to_thread(llm.invoke, prompt)
            responses[f"{model_type} Response"] = response.content
            
            if include_raw_text:
                responses[f"{model_type} Raw Text"] = context
        
        end_time = time.time()
        total_time = end_time - start_time
        responses["Processing Time"] = f"{total_time:.2f} seconds"
        
        logging.info("generate_response_async completed successfully")
        yield responses
    except Exception as e:
        logging.error(f"Error in generate_response_async: {str(e)}", exc_info=True)
        yield {"Error": f"Error generating response: {str(e)}"}

# Setup
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Fine-tuned RAG with local embeddings, Qdrant and OpenAI LLM")
    
    with gr.Row():
        pdf_files = gr.File(file_count="multiple", label="Upload PDF files")
        process_btn = gr.Button("Process PDFs")

    with gr.Row():
        num_trials = gr.Number(value=10, label="Number of Trials", minimum=1, maximum=100, step=1)
        fine_tune_btn = gr.Button("Start Fine-tuning", interactive=False)
    
    status_output = gr.Textbox(label="Status", lines=10)
    
    with gr.Row():
        question_input = gr.Textbox(label="Question")
        submit_btn = gr.Button("Submit", interactive=False)
    
    with gr.Row():
        include_raw_text = gr.Checkbox(label="Include Raw Text")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Base")
            base_output = gr.Markdown(label="Response")
            base_raw_text = gr.Textbox(label="Raw Text", visible=False)

        with gr.Column():
            gr.Markdown("## Fine-tuned")
            ft_output = gr.Markdown(label="Response")
            ft_raw_text = gr.Textbox(label="Raw Text", visible=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Base Hybrid")
            base_hybrid_output = gr.Markdown(label="Response")
            base_hybrid_raw_text = gr.Textbox(label="Raw Text", visible=False)

        with gr.Column():
            gr.Markdown("## Fine-tuned Hybrid")
            ft_hybrid_output = gr.Markdown(label="Response")
            ft_hybrid_raw_text = gr.Textbox(label="Raw Text", visible=False)

    with gr.Row():
        processing_time = gr.Textbox(label="Processing Time")
    
    async def update_outputs_async(question, include_raw_text):
        try:
            logging.info("Starting update_outputs_async")
            async for outputs in generate_response_async(question, include_raw_text):
                if "Error" in outputs:
                    error_message = outputs["Error"]
                    yield {
                        base_output: gr.Markdown(value=f"Error: {error_message}"),
                        ft_output: gr.Markdown(value=f"Error: {error_message}"),
                        base_hybrid_output: gr.Markdown(value=f"Error: {error_message}"),
                        ft_hybrid_output: gr.Markdown(value=f"Error: {error_message}"),
                        base_raw_text: gr.update(value="", visible=False),
                        ft_raw_text: gr.update(value="", visible=False),
                        base_hybrid_raw_text: gr.update(value="", visible=False),
                        ft_hybrid_raw_text: gr.update(value="", visible=False),
                        processing_time: ""
                    }
                else:
                    yield {
                        base_output: gr.Markdown(value=outputs.get("Base Response", "")),
                        ft_output: gr.Markdown(value=outputs.get("Fine-tuned Response", "")),
                        base_hybrid_output: gr.Markdown(value=outputs.get("Base Hybrid Response", "")),
                        ft_hybrid_output: gr.Markdown(value=outputs.get("Fine-tuned Hybrid Response", "")),
                        base_raw_text: gr.update(value=outputs.get("Base Raw Text", ""), visible=include_raw_text),
                        ft_raw_text: gr.update(value=outputs.get("Fine-tuned Raw Text", ""), visible=include_raw_text),
                        base_hybrid_raw_text: gr.update(value=outputs.get("Base Hybrid Raw Text", ""), visible=include_raw_text),
                        ft_hybrid_raw_text: gr.update(value=outputs.get("Fine-tuned Hybrid Raw Text", ""), visible=include_raw_text),
                        processing_time: outputs.get("Processing Time", "")
                    }
            logging.info("update_outputs_async completed successfully")
        except Exception as e:
            logging.error(f"Error in update_outputs_async: {str(e)}", exc_info=True)
            error_message = f"Error updating outputs: {str(e)}"
            yield {
                base_output: gr.Markdown(value=error_message),
                ft_output: gr.Markdown(value=error_message),
                base_hybrid_output: gr.Markdown(value=error_message),
                ft_hybrid_output: gr.Markdown(value=error_message),
                base_raw_text: gr.update(value="", visible=False),
                ft_raw_text: gr.update(value="", visible=False),
                base_hybrid_raw_text: gr.update(value="", visible=False),
                ft_hybrid_raw_text: gr.update(value="", visible=False),
                processing_time: ""
            }
    
    def update_fine_tune_btn(result):
        status, fine_tune_btn_update = result
        return status, fine_tune_btn_update
    
    def update_status_and_submit_btn(output):
        status, submit_btn_update = output
        return status, submit_btn_update

    def start_fine_tuning_with_trials(num_trials, progress=gr.Progress()):
        return start_fine_tuning(int(num_trials), progress)

    process_btn.click(process_uploaded_pdfs, inputs=[pdf_files], outputs=[status_output, fine_tune_btn])
    fine_tune_btn.click(start_fine_tuning_with_trials, inputs=[num_trials], outputs=[status_output, submit_btn])
    submit_btn.click(
        update_outputs_async,
        inputs=[question_input, include_raw_text],
        outputs=[base_output, ft_output, base_hybrid_output, ft_hybrid_output, base_raw_text, ft_raw_text, base_hybrid_raw_text, ft_hybrid_raw_text, processing_time]
    )

if __name__ == "__main__":
    try:
        print("Launching Gradio interface")
        result = iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
        print(f"Gradio interface launched: {result}")
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error launching Gradio interface: {error_details}")
