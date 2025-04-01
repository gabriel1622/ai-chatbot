from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from knowledge_base import embed_texts, load_index, load_texts
from train_knowledge import extract_text_from_file
import sqlite3
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_PATH = 'chat_history.db'

def save_to_db(user_input, ai_response):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO chats (user_input, ai_response) VALUES (?, ?)', (user_input, ai_response))
        conn.commit()

def load_history():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT user_input, ai_response FROM chats')
        return cursor.fetchall()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    memory_mode = data.get('memoryMode', False)

    # Retrieve history if memory mode is on
    history = load_history() if memory_mode else []

    # Retrieve relevant knowledge chunks
    relevant_chunks = []
    if os.path.exists('vector.index'):
        index = load_index('vector.index')
        texts = load_texts('texts.pkl')
        embedded_input = embed_texts([user_input])
        scores, indices = index.search(embedded_input, k=3)
        for idx in indices[0]:
            relevant_chunks.append(texts[idx])

    # Combine all context
    context = ""
    if history:
        context += "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in history[-5:]])
    if relevant_chunks:
        context += "\nRelevant info:\n" + "\n".join(relevant_chunks)

    # Generate response (stubbed for now)
    ai_response = f"Echo: {user_input}\n\nContext:\n{context}"

    save_to_db(user_input, ai_response)
    return jsonify({'response': ai_response})

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    full_text = ""

    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        text = extract_text_from_file(file_path)
        full_text += text + "\n"

    embed_texts([full_text], save_index_as='vector.index', save_texts_as='texts.pkl')
    return jsonify({"message": "Files uploaded and knowledge base updated."})

# Required for Render deployment
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

