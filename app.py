from flask import Flask, render_template, request, send_file
import openai
import os
import io
import sqlite3
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from knowledge_base import embed_texts, load_index, load_texts

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

DB_PATH = "chat_history.db"

def save_to_db(user_input, ai_response):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_input, ai_response) VALUES (?, ?)", (user_input, ai_response))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_input, ai_response FROM chats")
    rows = c.fetchall()
    conn.close()
    history = [{"role": "user", "content": row[0]} if i % 2 == 0 else {"role": "assistant", "content": row[1]} for i, row in enumerate(rows)]
    return history

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chats")
    conn.commit()
    conn.close()

@app.route("/", methods=["GET", "POST"])
def index():
    ai_response = ""
    memory_mode = False
    history = []

    if request.method == "POST":
        if "reset" in request.form:
            clear_history()
            ai_response = "?? Chat history cleared."
        elif "upload_files" in request.files:
            uploaded_files = request.files.getlist("upload_files")
            content = ""

            for file in uploaded_files:
                filename = file.filename.lower()
                path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)

                if filename.endswith(".pdf"):
                    reader = PdfReader(path)
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            content += text + "\n"
                else:
                    text = file.read().decode("utf-8")
                    content += text + "\n"

            history.append({
                "role": "system",
                "content": f"You are an AI trained on the following:\n\n{content}"
            })
            ai_response = "? Files uploaded successfully."

        elif "user_input" in request.form:
            user_input = request.form["user_input"]
            memory_mode = "memory_mode" in request.form

            if memory_mode:
                history = load_history()

            history.append({"role": "user", "content": user_input})

            # Vector-based retrieval
            try:
                index = load_index()
                texts = load_texts()
                user_embedding = embed_texts([user_input])
                D, I = index.search(user_embedding, k=3)
                relevant = "\n".join([texts[i] for i in I[0] if i < len(texts)])
                context = f"Use this info:\n{relevant}\n\n"
                if history and history[0]["role"] == "system":
                    history[0]["content"] += "\n" + context
                else:
                    history.insert(0, {"role": "system", "content": context})
            except:
                pass  # fallback if no vector data

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=history
            )
            ai_response = response.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": ai_response})
            save_to_db(user_input, ai_response)

    else:
        history = load_history()

    return render_template("index.html", response=ai_response, history=history, memory_mode=memory_mode)

@app.route("/download")
def download():
    history = load_history()
    output = io.StringIO()
    for msg in history:
        role = msg["role"].capitalize()
        output.write(f"{role}: {msg['content']}\n\n")
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name="chat_history.txt"
    )

@app.route("/download_pdf")
def download_pdf():
    from fpdf import FPDF
    history = load_history()
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for msg in history:
        role = msg["role"].capitalize()
        content = f"{role}: {msg['content']}"
        pdf.multi_cell(0, 10, txt=content)
    pdf_path = "chat_history.pdf"
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
