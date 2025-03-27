from flask import Flask, render_template, request
from flask import Flask, render_template, request
import openai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

conversation = []
pdf_content = ""

@app.route("/", methods=["GET", "POST"])
def index():
    global conversation, pdf_content
    ai_response = ""

    if request.method == "POST":
        if "pdf_file" in request.files:
            pdf_file = request.files["pdf_file"]
            if pdf_file.filename.endswith(".pdf"):
                pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], pdf_file.filename)
                pdf_file.save(pdf_path)

                reader = PdfReader(pdf_path)
                pdf_content = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_content += text

                conversation = [
                    {"role": "system", "content": f"You are an AI that has read this PDF:\n\n{pdf_content}"}
                ]
                ai_response = "PDF uploaded and loaded successfully."

        elif "user_input" in request.form:
            user_input = request.form["user_input"]
            conversation.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation
            )

            ai_response = response.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": ai_response})

    return render_template("index.html", response=ai_response, history=conversation)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
