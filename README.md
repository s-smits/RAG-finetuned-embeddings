# Verfijnde RAG met Lokale Embeddings, Qdrant en OpenAI LLM

Dit project implementeert een Retrieval-Augmented Generation (RAG) systeem met behulp van fine-tuned lokale embeddings, Qdrant vectoropslag en OpenAI's GPT-4 taalmodel.

## Installatie

1. Kloon de repository en navigeer naar de projectmap:
   ```
   git clone <repository-url>
   cd <project-directory>
   ```

2. Maak een virtuele omgeving aan met de naam `qdrant_demo` en activeer deze:
   - Voor macOS en Linux:
     ```
     python3 -m venv qdrant_demo
     source qdrant_demo/bin/activate
     ```
   - Voor Windows:
     ```
     python -m venv qdrant_demo
     qdrant_demo\Scripts\activate
     ```

3. Installeer de vereiste afhankelijkheden:
   ```
   pip install -r requirements.txt
   ```

4. Maak een `.env` bestand aan in de hoofdmap van het project en voeg je OpenAI API-sleutel toe:
   ```
   OPENAI_API_KEY=jouw_api_sleutel_hier
   ```

   Zorg ervoor dat je je eigen OpenAI API-sleutel hier invoert. Deze sleutel is noodzakelijk voor het gebruik van de OpenAI GPT-4 model in dit project.

## Gebruik

Start het script:```
python rag_finetune.py
```

Dit opent een Gradio-interface waar je:
1. PDF-bestanden kunt uploaden
2. De PDF's kunt verwerken
3. Het fijnafstemmen van de embeddings kunt starten
4. Vragen kunt stellen op basis van de verwerkte documenten

## Vereisten

Zie `requirements.txt` voor een volledige lijst van afhankelijkheden.

## Licentie

[MIT-licentie](LICENSE)