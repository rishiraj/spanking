import jax
import jax.numpy as jnp
import json
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PIL import Image
import requests
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorDB:
    def __init__(self, model_name='avsolatorio/GIST-small-Embedding-v0', image_model_name="google/siglip2-so400m-patch16-384"):
        self.model_name = model_name
        self.image_model_name = image_model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.image_classifier = pipeline(task="zero-shot-image-classification", model=image_model_name)
        self.texts = []
        self.embeddings = []
        self.metadatas = []

    def add_texts(self, texts, metadatas=None):
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("Texts and metadatas must have the same length")
        
        new_embeddings = jnp.array(self.model.encode(texts, normalize_embeddings=True))
        if not self.embeddings:
            self.embeddings = new_embeddings
        else:
            self.embeddings = jnp.concatenate((self.embeddings, new_embeddings), axis=0)
        
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def delete_text(self, index):
        if 0 <= index < len(self.texts):
            self.texts.pop(index)
            self.embeddings = jnp.delete(self.embeddings, index, axis=0)
            self.metadatas.pop(index)
        else:
            raise IndexError("Invalid index")

    def update_text(self, index, new_text, new_metadata=None):
        if 0 <= index < len(self.texts):
            self.texts[index] = new_text
            new_embedding = jnp.array(self.model.encode([new_text], normalize_embeddings=True)).squeeze()
            self.embeddings = self.embeddings.at[index].set(new_embedding)
            if new_metadata is not None:
                self.metadatas[index] = new_metadata
        else:
            raise IndexError("Invalid index")
    
    def add_docs(self, list_file_path, chunk_size=600, chunk_overlap=40):
        loaders = [PyPDFLoader(x) for x in list_file_path]
        pages = []
        for loader in loaders:
            pages.extend(loader.load())
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        doc_splits = text_splitter.split_documents(pages)
        texts = [doc.page_content for doc in doc_splits]
        metadatas = [doc.metadata for doc in doc_splits]
        
        self.add_texts(texts, metadatas)
    
    def search(self, query, top_k=5, type='text'):
        if type == 'text':
            query_embedding = jnp.array(self.model.encode([query], normalize_embeddings=True))
            similarities = jnp.dot(self.embeddings, query_embedding.T).squeeze()
        elif type == 'image':
            if isinstance(query, str):
                query = Image.open(requests.get(query, stream=True).raw)
            outputs = self.image_classifier(query, candidate_labels=self.texts)
            similarities = jnp.array([round(output["score"], 4) for output in outputs])
        else:
            raise ValueError("Invalid search type. Supported types are 'text' and 'image'.")
        top_indices = jnp.argsort(similarities)[-top_k:][::-1]
        
        results = [{
            "text": self.texts[i],
            "similarity": float(similarities[i]),
            "metadata": self.metadatas[i]
        } for i in top_indices]
        
        return json.dumps(results, indent=4)

    def save(self, file_path):
        data = {
            "model_name": self.model_name,
            "image_model_name": self.image_model_name,
            "texts": self.texts,
            "embeddings": self.embeddings.tolist(),
            "metadatas": self.metadatas
        }
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        vdb = VectorDB(data["model_name"], data["image_model_name"])
        vdb.texts = data["texts"]
        vdb.embeddings = jnp.array(data["embeddings"])
        vdb.metadatas = data["metadatas"]
        return vdb

    def to_df(self):
        data = {
            'text': self.texts,
            'embedding': [embedding.tolist() for embedding in self.embeddings],
            'metadata': self.metadatas
        }
        return pd.DataFrame(data)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.metadatas[index]

    def __iter__(self):
        return iter(zip(self.texts, self.metadatas))

def main():
    print("🍑👋")
