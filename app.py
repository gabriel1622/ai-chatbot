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

conversation = []
pdf_content = ""

def save_to_db(user_input, ai_response):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO chats (user_input, ai_response) VALUES (?, ?)", (user_input, ai_response))
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("SELECT user_input, ai_response FROM chats")
    rows = c.fetchall()
    conn.close()
    history = [{"role": "user", "content": row[0]} for row in rows]
    history += [{"role": "assistant", "content": row[1]} for row in rows]
    return history

@app.route("/", methods=["GET", "POST"])
def index():
    global conversation, pdf_content
    ai_response = ""
    memory_mode = False

    try:
        if request.method == "POST":
            memory_mode = "memory_mode" in request.form

            if "reset" in request.form:
                conversation.clear()
                ai_response = "?? Chat history has been reset."

            elif "upload_files" in request.files:
                uploaded_files = request.files.getlist("upload_files")
                pdf_content = ""

                for file in uploaded_files:
                    filename = file.filename.lower()

                    if filename.endswith(".pdf"):
                        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                        file.save(path)
                        reader = PdfReader(path)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                pdf_content += text + "\n"

                    elif filename.endswith((".txt", ".md", ".csv", ".json")):
                        text = file.read().decode("utf-8")
                        pdf_content += text + "\n"

                conversation.clear()
                conversation.append({
                    "role": "system",
                    "content": f"You are an AI trained on this content:\n\n{pdf_content}"
                })
                ai_response = "? Files uploaded and loaded."

            elif "user_input" in request.form:
                user_input = request.form["user_input"]

                if not conversation:
                    ai_response = "?? Please upload a knowledge file first."
                else:
                    conversation.append({"role": "user", "content": user_input})

                    # Vector-based knowledge context
                    index = load_index()
                    texts = load_texts()
                    user_embedding = embed_texts([user_input])
                    D, I = index.search(user_embedding, k=3)
                    relevant = "\n".join([texts[i] for i in I[0] if i < len(texts)])
                    context = f"Use this relevant info:\n{relevant}\n\n"
                    conversation[0]["content"] += "\n" + context

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=conversation
                    )
                    ai_response = response.choices[0].message.content.strip()
                    conversation.append({"role": "assistant", "content": ai_response})

                    if memory_mode:
                        save_to_db(user_input, ai_response)

    except Exception as e:
        ai_response = f"?? Internal Error: {str(e)}"
        print(f"Error: {str(e)}")

    return render_template("index.html", response=ai_response, history=conversation, memory_mode=memory_mode)

@app.route("/download")
def download():
    if not conversation:
        return "?? No chat history available.", 400

    output = io.StringIO()
    for msg in conversation:
        role = msg["role"].capitalize()
        output.write(f"{role}: {msg['content']}\n\n")
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype="text/plain", as_attachment=True, download_name="chat_history.txt")

@app.route("/download_pdf")
def download_pdf():
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for msg in conversation:
        role = msg["role"].capitalize()
        lines = f"{role}: {msg['content']}".splitlines()
        for line in lines:
            pdf.cell(200, 10, txt=line, ln=True)
    path = os.path.join(app.config["UPLOAD_FOLDER"], "chat_history.pdf")
    pdf.output(path)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
