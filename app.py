from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from keyword_extractor import extract_keywords_from_resume
from interview_generator import generate_interview_questions, get_interview_flow
import uuid
import requests

# -----------------------------------------------------------
# App Setup
# -----------------------------------------------------------

basedir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(basedir, 'static')

app = Flask(__name__, static_folder=static_dir, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HF_TOKEN = "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"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct:novita"


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------

def allowed_file(filename):
    """Check if file has .pdf extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def send_hf_chat_completion(prompt):
    """
    Sends a chat-completion request to HuggingFace Router.
    Returns model response or raises Exception.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert AI interviewer."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 180,
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=20)

        print("\n====== HF DEBUG ======")
        print("Status:", response.status_code)
        print("Response:", response.text)
        print("======================\n")

        # HTTP Error
        if response.status_code != 200:
            raise Exception(f"HF API Error {response.status_code}: {response.text}")

        data = response.json()

        # Parse HF message
        return data["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        raise Exception("HuggingFace API timeout. Please try again.")

    except Exception as e:
        raise Exception(f"HF Error: {str(e)}")


# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


# ----------------------- FILE UPLOAD -----------------------

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'File name is empty'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400

        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = extract_keywords_from_resume(filepath)
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f"Processing error: {str(e)}"}), 500


# --------------------- INTERVIEW PAGE ----------------------

@app.route('/interview')
def interview():
    return render_template('interview.html')


# ------------------ QUESTION GENERATOR ---------------------

@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    try:
        data = request.json or {}

        domain = data.get('domain', 'General / Other')
        seniority = data.get('seniority', 'Not Specified')
        tech_keywords = data.get('tech_keywords', [])

        questions = generate_interview_questions(domain, seniority, tech_keywords)
        flow = get_interview_flow(domain, seniority)

        return jsonify({
            'questions': questions,
            'flow': flow,
            'domain': domain,
            'seniority': seniority
        })

    except Exception as e:
        return jsonify({'error': f"Question generation error: {str(e)}"}), 500


# ------------------ AI INTERVIEW RESPONSE ------------------

@app.route('/api/ai-response', methods=['POST'])
def ai_response():
    try:
        data = request.json or {}

        user_input = data.get('user_input', '').strip()
        question = data.get('question', '').strip()

        if not user_input:
            return jsonify({'error': 'User answer missing'}), 400
        if not question:
            return jsonify({'error': 'Question context missing'}), 400

        prompt = (
            f"You are an AI interviewer conducting a technical interview.\n"
            f"Question: {question}\n"
            f"Candidate Answer: {user_input}\n\n"
            f"Respond with:\n"
            f"- A relevant follow-up question, OR\n"
            f"- Helpful feedback + new question.\n"
            f"Keep it short, professional, realistic."
        )

        ai_reply = send_hf_chat_completion(prompt)

        return jsonify({"ai_response": ai_reply})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -----------------------------------------------------------
# Start Server
# -----------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=5000)
