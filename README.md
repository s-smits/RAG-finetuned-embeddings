# Fine-tuned RAG met lokale Embeddings, Qdrant en OpenAI LLM

Dit project implementeert een Retrieval-Augmented Generation (RAG) systeem met behulp van fine-tuned lokale embeddings, Qdrant vectoropslag en OpenAI's GPT-4 taalmodel.

## Installatie

1. Clone de repository en navigeer naar de projectmap:
   ```
   git clone https://github.com/s-smits/RAG-finetuned-embeddings
   cd RAG-finetuned-embeddings
   ```

2. Maak een virtuele omgeving aan met de naam `venv_RAG_finetuned_embeddings` en activeer deze:
   - Voor macOS en Linux:
     ```
     python3 -m venv_RAG_finetuned_embeddings
     source venv_RAG_finetuned_embeddings/bin/activate
     ```
   - Voor Windows:
     ```
     python -m venv venv_RAG_finetuned_embeddings
     venv_RAG_finetuned_embeddings\Scripts\activate
     ```

3. Installeer de vereiste dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Maak een `.env` bestand aan in de hoofdmap van het project en voeg je OpenAI API-sleutel toe:
   ```
   OPENAI_API_KEY=jouw_api_sleutel_hier
   ```

   Zorg ervoor dat je je eigen OpenAI API-sleutel hier invoert. Deze sleutel is noodzakelijk voor het gebruik van de OpenAI GPT-4 model in dit project.

## Gebruik

Start het script:
```
   python rag_finetune.py
```

Dit opent een Gradio-interface waar je:
1. PDF-bestanden kunt uploaden
2. De PDF's kunt verwerken
3. Het fine-tunen van de embeddings kunt starten
4. Vragen kunt stellen op basis van de verwerkte documenten terwijl de gefine-tunede embeddings gebruikt worden.

## Vereisten

Zie `requirements.txt` voor een volledige lijst van dependencies.

## Licentie

[MIT-licentie](LICENSE)

## Demo
<video src="https://github.com/user-attachments/assets/b854534c-fcd4-44af-b907-94c309552709" controls="controls" style="max-width: 730px;">
</video>
