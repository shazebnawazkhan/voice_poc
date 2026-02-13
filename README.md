# POC for Voice Agent using various frameworks

## Intro

- Python setup

```bash
python -m venv .venv
```

```bash
.\.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```


## Livekit

- Usage
```bash
python .\src\livekit_test.py download-files
```
```bash
python .\src\livekit_test.py console
```


## RAG

- RAG sources

&emsp;The code refers to a local directory named 'rag_docs' within the root folder of code, and reads all the PDF documents to index for RAG.
The code also maintains an 'indexes' directory under the root, where it stores persisted copies of index for RAG. Structure of index is as below:
1) indexes/rag_docs_indexed_files.json - list of files indexed
2) indexes/rag_docs/index.faiss        - index in faiss
3) indexes/rag_docs/index.pkl          - pkl of index

- Readings

[https://medium.com/@versatile_umber_ant_241/implementing-role-based-access-control-in-rag-de4a4e129215](https://medium.com/@versatile_umber_ant_241/implementing-role-based-access-control-in-rag-de4a4e129215)

## Google ADK