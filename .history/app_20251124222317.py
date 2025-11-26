from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from keyword_extractor import extract_keywords_from_resume
from interview_generator import generate_interview_questions, get_interview_flow
import uuid

# Get the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))
static_dir = os.path.join(basedir, 'static')

app = Flask(__name__, static_folder=static_dir, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400
    
    try:
        # Generate unique filename
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract keywords
        result = extract_keywords_from_resume(filepath)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/interview')
def interview():
    """Interview page with camera access."""
    return render_template('interview.html')


@app.route('/api/generate-questions', methods=['POST'])
def generate_questions():
    """Generate interview questions based on domain and seniority."""
    try:
        data = request.json
        domain = data.get('domain', 'General / Other')
        seniority = data.get('seniority', 'Not Specified')
        tech_keywords = data.get('tech_keywords', [])
        
        questions = generate_interview_questions(domain, seniority, tech_keywords)
        interview_flow = get_interview_flow(domain, seniority)
        
        return jsonify({
            'questions': questions,
            'flow': interview_flow,
            'domain': domain,
            'seniority': seniority
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

