import jax
import jax.numpy as jnp
import pickle
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        self.model = SentenceTransformer(model_name)
        self.texts = []
        self.embeddings = None
    
    def add_texts(self, texts):
        new_embeddings = self.model.encode(texts, normalize_embeddings=True)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = jnp.concatenate((self.embeddings, new_embeddings), axis=0)
        self.texts.extend(texts)
    
    def delete_text(self, index):
        if 0 <= index < len(self.texts):
            self.texts.pop(index)
            self.embeddings = jnp.delete(self.embeddings, index, axis=0)
        else:
            raise IndexError("Invalid index")
    
    def update_text(self, index, new_text):
        if 0 <= index < len(self.texts):
            self.texts[index] = new_text
            new_embedding = self.model.encode([new_text], normalize_embeddings=True).squeeze()
            self.embeddings = (self.embeddings).at[index].set(new_embedding)
        else:
            raise IndexError("Invalid index")
    
    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = jnp.dot(self.embeddings, query_embedding.T).squeeze()
        top_indices = jnp.argsort(similarities)[-top_k:][::-1]
        return [(self.texts[i], similarities[i]) for i in top_indices]
    
    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index]
    
    def __iter__(self):
        return iter(self.texts)

def main():
    print("🍑👋")