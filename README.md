# spanking 🍑👋

To use the 🍑👋 `VectorDB` class, you can follow these steps:

1. Create an instance of the 🍑👋 `VectorDB` class:
```python
from spanking import VectorDB
vector_db = VectorDB(model_name='BAAI/bge-base-en-v1.5')
```
You can optionally specify a different pre-trained sentence transformer model by passing its name to the constructor.

2. Add texts to the database:
```python
texts = ["i eat pizza", "i play chess", "i drive bus"]
vector_db.add_texts(texts)
```
This will encode the texts into embeddings and store them in the database.

3. Search for similar texts or images:
```python
text_query = "we play football"
text_results = vector_db.search(text_query, top_k=2, type='text')
print("Text search results:")
for text, similarity in text_results:
    print(f"Text: {text}, Similarity: {similarity}")

image_url = "https://example.com/image.jpg"
image_results = vector_db.search(image_url, top_k=2, type='image')
print("\nImage search results:")
for text, similarity in image_results:
    print(f"Text: {text}, Similarity: {similarity}")
```
This will retrieve the top-2 most similar texts or images to the query based on cosine similarity. The `search` method returns a list of tuples, where each tuple contains the text and its similarity score. You can specify the search type using the `type` parameter (`'text'` for text search and `'image'` for image search).

4. Delete a text from the database:
```python
index = 1
vector_db.delete_text(index)
```
This will remove the text and its corresponding embedding at the specified index.

5. Update a text in the database:
```python
index = 0
new_text = "i enjoy eating pizza"
vector_db.update_text(index, new_text)
```
This will update the text and its corresponding embedding at the specified index with the new text.

6. Save the database to a file:
```python
vector_db.save('vector_db.pkl')
```
This will save the current state of the `VectorDB` instance to a file named 'vector_db.pkl'.

7. Load the database from a file:
```python
vector_db = VectorDB.load('vector_db.pkl')
```
This will load the `VectorDB` instance from the file named 'vector_db.pkl' and return it.

8. Convert the database to a Pandas Dataframe
```python
df = vector_db.to_df()
```
This will convert the current state of the `VectorDB` instance to a Pandas Dataframe.

9. Iterate over the stored texts:
```python
for text in vector_db:
    print(text)
```
This will iterate over all the texts stored in the database.

10. Access individual texts by index:
```python
index = 2
text = vector_db[index]
print(text)
```
This will retrieve the text at the specified index.

11. Get the number of texts in the database:
```python
num_texts = len(vector_db)
print(num_texts)
```
This will return the number of texts currently stored in the database.

Here's an example usage of the 🍑👋 `VectorDB` class:

```python
from spanking import VectorDB
vector_db = VectorDB()

# Add texts to the database
texts = ["i eat pizza", "i play chess", "i drive bus"]
vector_db.add_texts(texts)

# Search for similar texts
query = "we play football"
top_results = vector_db.search(query, top_k=2)
print("Top results:")
for text, similarity in top_results:
    print(f"Text: {text}, Similarity: {similarity}")

# Update a text
vector_db.update_text(1, "i enjoy playing chess")

# Delete a text
vector_db.delete_text(2)

# Save the database
vector_db.save('vector_db.pkl')

# Load the database
loaded_vector_db = VectorDB.load('vector_db.pkl')

# Iterate over the stored texts in the loaded database
print("\nStored texts in the loaded database:")
for text in loaded_vector_db:
    print(text)

# Convert to dataframe
df = loaded_vector_db.to_df()
print(df.head())
```

This example demonstrates how to create a 🍑👋 `VectorDB` instance, add texts, search for similar texts, update and delete texts, and iterate over the stored texts.
