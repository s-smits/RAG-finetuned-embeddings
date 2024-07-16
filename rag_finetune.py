# rag_agent.py

from dotenv import load_dotenv
import os
import torch
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
import keras
from keras.layers import Input
import traceback
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from keras import models

# Global variables
qdrant_collection = None
fine_tuned_embeddings = None
all_documents = []
txt_files = []
is_fine_tuned = False

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() for page in reader.pages)

class FineTunedHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name, documents, n_trials=10, optuna_callback=None, **kwargs):
        try:
            print("Initializing FineTunedHuggingFaceEmbeddings")
            self.base_embeddings = HuggingFaceEmbeddings(model_name=model_name, **kwargs)
            self.fine_tuned_model, self.study = self.finetune_embeddings(documents, n_trials=n_trials, optuna_callback=optuna_callback)
            print("FineTunedHuggingFaceEmbeddings initialized successfully")
        except Exception as e:
            print(f"Error in FineTunedHuggingFaceEmbeddings initialization: {str(e)}")
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

def process_uploaded_pdfs(pdf_files):
    global all_documents, txt_files, is_fine_tuned
    
    try:
        print("Starting process_uploaded_pdfs")
        if not pdf_files:
            return "No PDF files uploaded. Please upload PDF files and try again.", gr.update(interactive=False)

        all_documents = []
        txt_files = []
        is_fine_tuned = False  # Reset fine-tuning status

        for pdf_file in pdf_files:
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

        print("process_uploaded_pdfs completed successfully")
        return f"Processed {len(pdf_files)} PDFs. Created {len(all_documents)} chunks. Click 'Start Fine-tuning' to continue.", gr.update(interactive=True)
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in process_uploaded_pdfs: {error_details}")
        return f"Error processing PDFs: {str(e)}\n\nFull error details:\n{error_details}", gr.update(interactive=False)

def start_fine_tuning(progress=gr.Progress()):
    global qdrant_collection, fine_tuned_embeddings, is_fine_tuned

    try:
        print("Starting start_fine_tuning")
        progress(0.0, desc="Initializing fine-tuning process")
        
        model_name = "all-MiniLM-L6-v2"
        local_model_path = os.path.join(os.getcwd(), "local_models", model_name)

        progress(0.1, desc="Loading model")
        if not os.path.exists(local_model_path):
            SentenceTransformer(model_name).save(local_model_path)

        n_trials = 10

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
            optimization_progress = 0.1 + (0.7 * (trial.number + 1) / n_trials)
            progress(optimization_progress, desc=f"Optimizing embeddings: Trial {trial.number + 1}/{n_trials}")
            
            return "\n".join(optuna_output)

        progress(0.1, desc="Starting optimization")
        fine_tuned_embeddings = FineTunedHuggingFaceEmbeddings(
            model_name=local_model_path,
            documents=all_documents,
            n_trials=n_trials,
            model_kwargs={"device": "cpu"},
            optuna_callback=optuna_callback
        )

        progress(0.80, desc="Optimization complete")

        qdrant_documents = [
            Document(page_content=doc["text"], metadata=doc["metadata"])
            for doc in all_documents
        ]

        progress(0.90, desc="Creating Qdrant collection")
        qdrant_collection = Qdrant.from_documents(
            qdrant_documents,
            fine_tuned_embeddings,
            location=":memory:", 
            collection_name="pdf_collection",
        )

        is_fine_tuned = True
        progress(1.0, desc="Fine-tuning complete")
        
        optuna_summary = "\n".join(optuna_output)
        best_model = f"**Model {fine_tuned_embeddings.study.best_trial.number + 1} (LR: {fine_tuned_embeddings.study.best_params['learning_rate']:.6f})**"
        final_message = f"Fine-tuning complete. You can now ask questions.\n\nOptuna Optimization Summary:\n{optuna_summary}\n\nChosen model for best results: {best_model}\nBest Metric (Cosine Similarity): {fine_tuned_embeddings.study.best_trial.value:.6f}"
        
        print("start_fine_tuning completed successfully")
        return final_message, gr.update(interactive=True), gr.update(interactive=True)  # Enable the submit button here
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in start_fine_tuning: {error_details}")
        return f"Error during fine-tuning: {str(e)}\n\nFull error details:\n{error_details}", gr.update(interactive=False), gr.update(interactive=False)

def get_relevant_document(query: str) -> str:
    try:
        print("Starting get_relevant_document")
        if not is_fine_tuned:
            raise ValueError("Fine-tuning has not been performed. Please start fine-tuning first.")
        if qdrant_collection is None:
            raise ValueError("Qdrant collection is not initialized. Please process PDFs and start fine-tuning first.")
        query_embedding = fine_tuned_embeddings.embed_query(query)
        results = qdrant_collection.similarity_search_by_vector(query_embedding, k=5)
        print("get_relevant_document completed successfully")
        return "\n\n".join(doc.page_content for doc in results)
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in get_relevant_document: {error_details}")
        return f"Error retrieving relevant documents: {str(e)}\n\nFull error details:\n{error_details}"

def generate_response(question, include_raw_text=False):
    try:
        print("Starting generate_response")
        if not is_fine_tuned:
            return {"Main Response": "Please complete the fine-tuning process before asking questions."}
        
        relevant_doc = get_relevant_document(question)
        
        prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{relevant_doc}

Question: {question}

Answer:"""

        main_response = llm.invoke(prompt).content
        
        outputs = {"Main Response": main_response}
        
        if include_raw_text:
            outputs["Raw Text"] = relevant_doc
        
        print("generate_response completed successfully")
        return outputs
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error in generate_response: {error_details}")
        return {"Main Response": f"Error generating response: {str(e)}\n\nFull error details:\n{error_details}"}

def evaluate_retrieval(embeddings_model, documents, queries, top_k=5):
    # Create BM25 index for lexical search baseline
    corpus = [doc["text"] for doc in documents]
    bm25 = BM25Okapi(corpus)
    
    embedding_hits = 0
    bm25_hits = 0
    
    for query in queries:
        # Semantic search
        query_embedding = embeddings_model.embed_query(query)
        semantic_results = qdrant_collection.similarity_search_by_vector(query_embedding, k=top_k)
        semantic_docs = [doc.page_content for doc in semantic_results]
        
        # BM25 search
        bm25_results = bm25.get_top_n(query.split(), corpus, n=top_k)
        
        # Compare results (assuming relevant docs contain the query terms)
        embedding_hits += sum(1 for doc in semantic_docs if any(term in doc.lower() for term in query.lower().split()))
        bm25_hits += sum(1 for doc in bm25_results if any(term in doc.lower() for term in query.lower().split()))
    
    embedding_precision = embedding_hits / (len(queries) * top_k)
    bm25_precision = bm25_hits / (len(queries) * top_k)
    
    return {
        "embedding_precision": embedding_precision,
        "bm25_precision": bm25_precision
    }

# Setup
load_dotenv()
llm = ChatOpenAI(model_name="gpt-4", api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Fine-tuned RAG with local embeddings, Qdrant and OpenAI LLM")
    
    with gr.Row():
        pdf_files = gr.File(file_count="multiple", label="Upload PDF files")
        process_btn = gr.Button("Process PDFs")
        fine_tune_btn = gr.Button("Start Fine-tuning", interactive=False)
    
    status_output = gr.Textbox(label="Status", lines=10)
    
    with gr.Row():
        question_input = gr.Textbox(label="Question")
        submit_btn = gr.Button("Submit", interactive=False)
    
    with gr.Row():
        include_raw_text = gr.Checkbox(label="Include Raw Text")
    
    with gr.Row():
        main_output = gr.Textbox(label="Main Response")
        raw_text_output = gr.Textbox(label="Raw Text", visible=False)
    
    def update_outputs(question, include_raw_text):
        try:
            print("Starting update_outputs")
            outputs = generate_response(question, include_raw_text)
            print("update_outputs completed successfully")
            return {
                main_output: outputs["Main Response"],
                raw_text_output: gr.update(value=outputs.get("Raw Text", ""), visible=include_raw_text)
            }
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error in update_outputs: {error_details}")
            return {
                main_output: f"Error updating outputs: {str(e)}\n\nFull error details:\n{error_details}",
                raw_text_output: gr.update(value="", visible=False)
            }
    
    def update_fine_tune_btn(result):
        status, fine_tune_btn_update = result
        return status, fine_tune_btn_update
    
    def update_status_and_submit_btn(output):
        status, submit_btn_update = output
        return status, submit_btn_update

    process_btn.click(process_uploaded_pdfs, inputs=[pdf_files], outputs=[status_output, fine_tune_btn])
    fine_tune_btn.click(start_fine_tuning, outputs=[status_output, submit_btn])
    submit_btn.click(update_outputs, inputs=[question_input, include_raw_text], outputs=[main_output, raw_text_output])

if __name__ == "__main__":
    try:
        print("Launching Gradio interface")
        result = iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
        print(f"Gradio interface launched: {result}")
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error launching Gradio interface: {error_details}")