import os
import tempfile
import base64
from flask import Flask, request, jsonify, render_template, send_file, url_for
from werkzeug.utils import secure_filename
from threading import Thread
from chatcast import process_pdf_to_podcast

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Store job status in memory (would use a database in production)
jobs = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if the file was selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Verify file type
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Save the file temporarily
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    # Generate a unique job ID
    job_id = base64.urlsafe_b64encode(os.urandom(16)).decode('utf-8')
    jobs[job_id] = {
        'status': 'processing',
        'filename': filename,
        'input_path': file_path,
        'output_path': None,
        'error': None
    }
    
    # Start processing in a background thread
    thread = Thread(target=process_in_background, args=(job_id, file_path))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'processing',
        'status_url': url_for('check_status', job_id=job_id)
    })

def process_in_background(job_id, file_path):
    try:
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + '.wav'
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Process the PDF to create podcast audio
        process_pdf_to_podcast(file_path, output_path)
        
        # Update job status
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['output_path'] = output_path
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'status': job['status'],
        'filename': job['filename']
    }
    
    if job['status'] == 'completed':
        response['download_url'] = url_for('download_audio', job_id=job_id)
    elif job['status'] == 'failed':
        response['error'] = job['error']
    
    return jsonify(response)

@app.route('/download/<job_id>', methods=['GET'])
def download_audio(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Audio not ready yet'}), 400
    
    # Clean filename for download
    filename = os.path.splitext(job['filename'])[0] + '.wav'
    
    return send_file(
        job['output_path'],
        as_attachment=True,
        download_name=filename,
        mimetype='audio/wav'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)