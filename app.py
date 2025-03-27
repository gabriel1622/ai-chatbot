from flask import Flask, render_template, request, send_file
from flask import Flask, render_template, request, send_file
import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

conversation = []
pdf_content = ""

# Ensure uploads folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    global conversation, pdf_content
    ai_response = ""

    try:
        if request.method == "POST":
            if "pdf_file" in request.files and request.files["pdf_file"].filename.endswith(".pdf"):
                pdf_file = request.files["pdf_file"]
                pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
                pdf_file.save(pdf_path)

                # Try reading the PDF and extracting content
                reader = PdfReader(pdf_path)
                pdf_content = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_content += text

                conversation.clear()
                conversation.append({
                    "role": "system",
                    "content": f"You are an AI that has read this PDF:\n\n{pdf_content}"
                })

                ai_response = "? PDF uploaded and loaded successfully."

            elif "user_input" in request.form:
                user_input = request.form["user_input"]

                if not conversation:
                    ai_response = "?? Please upload a PDF first before asking questions."
                else:
                    conversation.append({"role": "user", "content": user_input})
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=conversation
                    )
                    ai_response = response.choices[0].message.content.strip()
                    conversation.append({"role": "assistant", "content": ai_response})

    except Exception as e:
        # Log the full exception for debugging
        ai_response = f"?? Internal Error: {str(e)}"
        print(f"Error: {str(e)}")

    return render_template("index.html", response=ai_response, history=conversation)

@app.route("/download")
def download():
    if not conversation:
        return "?? No chat history available.", 400

    output = io.StringIO()
    for msg in conversation:
        role = msg["role"].capitalize()
        output.write(f"{role}: {msg['content']}\n\n")
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name="chat_history.txt"
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
