from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
from main import VectorDB

app = Flask(__name__)
db = VectorDB()

@app.route('/')
def index():
    texts = db.texts
    return render_template('index.html', texts=texts)

@app.route('/add_text', methods=['POST'])
def add_text():
    text = request.form.get('text')
    if text:
        db.add_texts([text])
    return redirect(url_for('index'))

@app.route('/delete_text/<int:index>', methods=['POST'])
def delete_text(index):
    try:
        db.delete_text(index)
    except IndexError as e:
        return str(e), 400
    return redirect(url_for('index'))

@app.route('/update_text/<int:index>', methods=['POST'])
def update_text(index):
    new_text = request.form.get('text')
    try:
        db.update_text(index, new_text)
    except IndexError as e:
        return str(e), 400
    return redirect(url_for('index'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        search_type = request.form.get('type')
        results = db.search(query, type=search_type)
        return render_template('search_results.html', results=results, query=query)
    return render_template('search.html')

@app.route('/save', methods=['POST'])
def save():
    file_path = request.form.get('file_path')
    db.save(file_path)
    return redirect(url_for('index'))

@app.route('/load', methods=['POST'])
def load():
    file_path = request.form.get('file_path')
    global db
    db = VectorDB.load(file_path)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
