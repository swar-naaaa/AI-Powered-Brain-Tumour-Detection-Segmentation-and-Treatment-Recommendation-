from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import google.generativeai as genai
import json
import os
import hashlib
from werkzeug.utils import secure_filename
import io
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import textwrap

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Upload configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# User data file
USER_DATA_FILE = 'users.json'

# Global model variable
model = None

# ============= BACKEND CODE (UNCHANGED) =============
# Set matplotlib backend to non-GUI for web environment
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Gemini configuration
try:
    genai.configure(api_key="AIzaSyALQl3IlQPXT_dD8k5kvBA9j3aXenmfDAg")
    gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")
    GEMINI_AVAILABLE = True
    print("✓ Gemini AI configured successfully")
except Exception as e:
    print(f"Warning: Gemini configuration failed: {e}")
    GEMINI_AVAILABLE = False

def get_model(checkpoint_path):
    num_classes = 2
    model = maskrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def transform_image(image):
    return F.to_tensor(image)

def get_prediction(model, image, score_threshold=0.5):
    image_tensor = transform_image(image)
    with torch.no_grad():
        prediction = model([image_tensor])
    return prediction[0]

def plot_predictions(image, prediction, score_threshold=0.5):
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode
    
    image_np = np.array(image)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)
    
    if "scores" not in prediction:
        plt.close(fig)
        return None, "No predictions found!"
    
    pred_scores = prediction["scores"].detach().cpu()
    keep = pred_scores >= score_threshold
    
    if keep.sum() == 0:
        plt.close(fig)
        return None, "No detections above the score threshold."
    
    pred_masks = prediction["masks"].detach().cpu()[keep]
    pred_boxes = prediction["boxes"].detach().cpu()[keep]
    pred_scores = pred_scores[keep]
    
    for i in range(len(pred_masks)):
        mask = pred_masks[i][0]
        binary_mask = mask > 0.5
        ax.imshow(binary_mask, alpha=0.5, cmap='Reds')
        box = pred_boxes[i].numpy()
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="blue", linewidth=2)
        ax.add_patch(rect)
        score = pred_scores[i].item()
        label_text = f"Tumor: {score:.2f}"
        ax.text(xmin, ymin - 10, label_text, color="blue", fontsize=12, weight="bold", backgroundcolor="white")
    
    ax.axis('off')
    
    # Save plot to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)  # Important: close the figure to free memory
    
    return img_str, None

def generate_response_(text):
    prompt = f'''You are a helpful medical chatbot specialized in neuro-oncology. {text}'''
    
    if not GEMINI_AVAILABLE:
        return "Gemini AI is not available. Please check your API configuration."
    
    try:
        response = gemini_model.generate_content(prompt)
        
        # Check if response has text
        if hasattr(response, 'text') and response.text:
            return response.text
        
        # Fallback if no text available
        return "Unable to generate a response. Please consult a medical professional."
            
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "I'm sorry, couldn't generate a response"
def extract_tumor_features(prediction, score_threshold=0.5):
    boxes = prediction.get("boxes", [])
    scores = prediction.get("scores", [])

    if scores is None or len(scores) == 0:
        return []

    scores = scores.detach().cpu()
    boxes = boxes.detach().cpu()

    valid = scores >= score_threshold
    boxes = boxes[valid]

    features = []

    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        width = 512
        height = 512

        side = "left hemisphere" if cx < width/2 else "right hemisphere"

        if cy < height * 0.33:
            region = "frontal lobe"
        elif cy < height * 0.66:
            region = "parietal lobe"
        else:
            region = "occipital lobe"

        features.append({
            "location": f"{side}, {region}"
        })

    return features

def get_region_details(region_text):
    region_text = region_text.lower()

    if "frontal" in region_text:
        return {
            "functions": "decision-making, personality, voluntary movement, and speech production",
            "symptoms": [
                "Changes in personality or behavior",
                "Difficulty in decision-making or concentration",
                "Weakness or paralysis on one side of the body",
                "Speech difficulties (especially in left frontal lobe)"
            ]
        }

    elif "parietal" in region_text:
        return {
            "functions": "sensory perception, spatial awareness, and coordination",
            "symptoms": [
                "Difficulty recognizing objects or shapes",
                "Loss of spatial awareness (e.g., bumping into objects)",
                "Problems with hand-eye coordination",
                "Numbness or reduced sensation in parts of the body"
            ]
        }

    elif "occipital" in region_text:
        return {
            "functions": "visual processing and interpretation",
            "symptoms": [
                "Blurred or partial vision loss",
                "Difficulty recognizing faces or objects",
                "Visual hallucinations",
                "Problems interpreting visual information"
            ]
        }

    else:
        return {
            "functions": "general neurological functions",
            "symptoms": [
                "Headaches",
                "Cognitive disturbances",
                "General neurological discomfort"
            ]
        }
def get_treatment_plan_from_gemini(prediction, score_threshold=0.5):
    scores = prediction.get("scores", None)

    # -------- NO TUMOR --------
    if scores is None or len(scores) == 0:
        return """**Result Summary:**
No tumor detected in the provided image.

**What This Means:**
The AI did not identify tumor-related abnormalities.

**Recommended Next Steps:**
- Continue regular checkups
- Consult a doctor if symptoms persist

**Important Note:**
This is an AI-assisted tool and not a medical diagnosis.
"""

    # -------- TUMOR DETECTED --------
    max_score = float(scores.max().item())
    num_detections = len(scores[scores >= score_threshold])

    features = extract_tumor_features(prediction, score_threshold)

    if features:
        region_text = ", ".join([f['location'] for f in features])
        feature_text = "\n".join([
            f"- Tumor {i+1}: Located in {f['location']}"
            for i, f in enumerate(features)
        ])
    else:
        region_text = "unknown region"
        feature_text = "- Region could not be determined"

    details = get_region_details(region_text)
    functions = details["functions"]
    symptoms_list = details["symptoms"]

    symptoms_text = "\n".join([f"- {s}" for s in symptoms_list])
    # 🔥 STRONG PROMPT
    prompt = f"""
You are a neuro-oncology AI assistant.

Brain MRI Analysis:

- Confidence: {max_score:.1%}
- Tumors detected: {num_detections}
- Tumor Location: {region_text}

Detailed Regions:
{feature_text}

STRICT INSTRUCTIONS:
- MUST clearly mention tumor location ({region_text})
- Explain based on that brain region
- DO NOT give generic answers
- Make response specific and unique

Give:

1. Result Summary (mention location clearly)
2. What This Means (based on region)
3. Possible Symptoms (region-specific)
4. Recommended Next Steps
5. Urgency Level
"""

    # -------- GEMINI --------
    try:
        response = gemini_model.generate_content(prompt)

        if hasattr(response, 'text') and response.text:
            return response.text.strip()

    except Exception as e:
        print("Gemini Error:", e)

    # -------- FALLBACK (FIXED - NOW HAS REGION) --------
    return f"""**Result Summary:**
The AI model has detected a potential tumor region in the brain scan with a confidence score of {max_score:.1%}. A total of {num_detections} suspicious region(s) were identified. The tumor is located in the {region_text}.

**What This Means:**
The AI has identified abnormal tissue patterns commonly associated with brain tumors. The presence of a tumor in the {region_text} indicates that specific functions controlled by this area of the brain may be affected.

**Basic Interpretation:**
A confidence score of {max_score:.1%} indicates a high likelihood of tumor presence. The number of detected regions ({num_detections}) suggests the tumor may be localized. However, AI predictions must always be validated clinically.

**Functional Impact:**
The {region_text} plays an important role in brain activity. A tumor in this region can interfere with normal neurological functions depending on its size and progression.

**Functional Impact:**
The tumor is located in the {region_text}, which is responsible for {functions}. Disruption in this region may affect these functions.

**Possible Symptoms:**
{symptoms_text} 

**Recommended Next Steps:**
- Consult a neurologist or neurosurgeon immediately  
- Perform MRI with contrast for better evaluation  
- Follow clinical diagnostic procedures (biopsy if required)  
- Seek a second medical opinion if necessary  

**Urgency Level:** High  
Given the high confidence score and involvement of a critical brain region ({region_text}), immediate medical evaluation is strongly advised.

**Important Note:**
This is an AI-assisted screening tool for educational purposes only. It is not a substitute for professional medical diagnosis.
"""# ============= END BACKEND CODE =============

# User management functions
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('detection'))
    return render_template('home.html')

@app.route('/home')
def home_redirect():
    if 'username' in session:
        return redirect(url_for('detection'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        users = load_users()
        
        if username in users and users[username]['password'] == hash_password(password):
            session['username'] = username
            session['user_data'] = users[username]
            flash('Login successful!', 'success')
            return redirect(url_for('detection'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = load_users()
        
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if username in users:
            flash('Username already exists!', 'error')
        elif password != confirm_password:
            flash('Passwords do not match!', 'error')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
        else:
            # Save user
            users[username] = {
                'first_name': request.form['first_name'],
                'last_name': request.form['last_name'],
                'email': email,
                'password': hash_password(password),
                'age': request.form['age'],
                'purpose': request.form['purpose'],
                'created_at': datetime.now().isoformat()
            }
            save_users(users)
            
            session['username'] = username
            session['user_data'] = users[username]
            flash('Account created successfully!', 'success')
            return redirect(url_for('detection'))
    
    return render_template('signup.html')

@app.route('/detection')
def detection():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('detection.html', user=session.get('user_data'))

@app.route('/load_model', methods=['POST'])
def load_model():
    global model
    checkpoint_path = "rcnn_brain_tumor_epoch_10.pth"
    
    try:
        model = get_model(checkpoint_path)
        return jsonify({'success': True, 'message': 'Model loaded successfully!'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading model: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        return jsonify({'success': False, 'message': 'Please load the model first.'})

    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'success': False, 'message': 'No file(s) uploaded.'})

    results = []
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            results.append({'filename': file.filename or 'unknown', 'success': False, 'message': 'Invalid file type.'})
            continue
        try:
            image = Image.open(file).convert("RGB")
            prediction = get_prediction(model, image)

            scores = prediction.get('scores', None)
            tumor_detected = scores is not None and len(scores) > 0 and float(scores.max().item()) >= 0.5

            if tumor_detected:
                plot_img, error = plot_predictions(image, prediction)
                if error:
                    results.append({'filename': file.filename, 'success': False, 'message': error})
                    continue
                result_label = 'Tumor Detected'
            else:
                # No tumor — return the original image without overlays
                buf = io.BytesIO()
                image.save(buf, format='PNG')
                buf.seek(0)
                plot_img = base64.b64encode(buf.getvalue()).decode()
                result_label = 'No Tumor Detected'

            treatment_plan = get_treatment_plan_from_gemini(prediction)

            results.append({
                'filename': file.filename,
                'success': True,
                'plot_image': plot_img,
                'treatment_plan': treatment_plan,
                'tumor_detected': tumor_detected,
                'result_label': result_label
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({'filename': file.filename, 'success': False, 'message': f'Error: {str(e)}'})

    return jsonify({'success': True, 'results': results})


@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate a downloadable PDF report from analysis results."""
    try:
        data = request.get_json()
        results = data.get('results', [])
        user_name = data.get('user_name', 'Patient')
        generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch
        )

        styles = getSampleStyleSheet()
        story = []

        # ---- Custom Styles ----
        title_style = ParagraphStyle('ReportTitle', parent=styles['Title'],
            fontSize=22, textColor=colors.HexColor('#1e40af'), spaceAfter=6, alignment=TA_CENTER)
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
            fontSize=11, textColor=colors.HexColor('#475569'), alignment=TA_CENTER, spaceAfter=4)
        section_header_style = ParagraphStyle('SectionHeader', parent=styles['Heading2'],
            fontSize=13, textColor=colors.HexColor('#1e40af'), spaceBefore=14, spaceAfter=6,
            borderPad=4)
        body_style = ParagraphStyle('Body', parent=styles['Normal'],
            fontSize=10, leading=15, textColor=colors.HexColor('#1e293b'),
            spaceAfter=6, alignment=TA_JUSTIFY)
        label_style = ParagraphStyle('Label', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#64748b'), spaceAfter=2)
        result_detected_style = ParagraphStyle('Detected', parent=styles['Normal'],
            fontSize=12, textColor=colors.HexColor('#dc2626'), fontName='Helvetica-Bold', spaceAfter=4)
        result_clear_style = ParagraphStyle('Clear', parent=styles['Normal'],
            fontSize=12, textColor=colors.HexColor('#16a34a'), fontName='Helvetica-Bold', spaceAfter=4)
        disclaimer_style = ParagraphStyle('Disclaimer', parent=styles['Normal'],
            fontSize=8.5, textColor=colors.HexColor('#64748b'), leading=12,
            alignment=TA_CENTER, spaceBefore=10)

        # ---- Report Header ----
        story.append(Paragraph("Brain Tumor Detection Report", title_style))
        story.append(Paragraph("AI-Assisted Neuro-Oncology Screening", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceAfter=10))

        # ---- Patient / Report Info ----
        info_data = [
            [Paragraph('<b>Patient Name:</b>', label_style), Paragraph(user_name, body_style),
             Paragraph('<b>Report Date:</b>', label_style), Paragraph(generated_at, body_style)],
            [Paragraph('<b>Total Scans Analyzed:</b>', label_style), Paragraph(str(len(results)), body_style),
             Paragraph('<b>System:</b>', label_style), Paragraph('Brain Tumor Detection AI v1.0', body_style)],
        ]
        info_table = Table(info_data, colWidths=[1.5*inch, 2.3*inch, 1.5*inch, 2.3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f0f9ff')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#bfdbfe')),
            ('INNERGRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dbeafe')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 14))

        # ---- Per-Image Results ----
        for idx, result in enumerate(results, 1):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceAfter=8))
            story.append(Paragraph(f"Scan #{idx} — {result.get('filename', 'Image')}", section_header_style))

            # Result label
            tumor_detected = result.get('tumor_detected', False)
            label_text = "🔴  Tumor Detected" if tumor_detected else "🟢  No Tumor Detected"
            label_st = result_detected_style if tumor_detected else result_clear_style
            story.append(Paragraph(label_text, label_st))
            story.append(Spacer(1, 6))

            # Scan image
            if result.get('plot_image'):
                try:
                    img_data = base64.b64decode(result['plot_image'])
                    img_buf = io.BytesIO(img_data)
                    pil_img = Image.open(img_buf)
                    w, h = pil_img.size
                    max_w = 5.5 * inch
                    max_h = 3.5 * inch
                    scale = min(max_w / w, max_h / h, 1.0)
                    rl_img = RLImage(io.BytesIO(img_data), width=w * scale, height=h * scale)
                    story.append(rl_img)
                    story.append(Spacer(1, 8))
                except Exception as img_err:
                    story.append(Paragraph(f"[Image could not be rendered: {img_err}]", body_style))

            # AI description
            story.append(Paragraph("AI Medical Insights", ParagraphStyle('InsightHeader', parent=styles['Heading3'],
                fontSize=11, textColor=colors.HexColor('#0f172a'), spaceBefore=6, spaceAfter=4)))

            treatment_text = result.get('treatment_plan', 'No analysis available.')
            # Convert markdown bold to HTML bold for ReportLab
            formatted = treatment_text.replace('**', '<b>', 1)
            count = 1
            while '**' in formatted:
                if count % 2 == 1:
                    formatted = formatted.replace('**', '</b>', 1)
                else:
                    formatted = formatted.replace('**', '<b>', 1)
                count += 1
            # Split into paragraphs
            for para_text in treatment_text.split('\n'):
                para_text = para_text.strip()
                if not para_text:
                    story.append(Spacer(1, 4))
                    continue
                # Bold headers (lines starting with **)
                if para_text.startswith('**') and para_text.endswith('**'):
                    clean = para_text.strip('*')
                    story.append(Paragraph(f'<b>{clean}</b>', ParagraphStyle('BoldLine', parent=body_style,
                        textColor=colors.HexColor('#1e40af'), spaceBefore=6)))
                elif para_text.startswith('-') or para_text.startswith('•'):
                    story.append(Paragraph(f'&nbsp;&nbsp;&nbsp;{para_text}', body_style))
                else:
                    # Inline bold replacement
                    import re
                    clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', para_text)
                    story.append(Paragraph(clean, body_style))

            story.append(Spacer(1, 10))

        # ---- Disclaimer ----
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0'), spaceBefore=10))
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI-assisted screening tool for educational purposes only. "
            "It is NOT a substitute for professional medical diagnosis or advice. "
            "Please consult a qualified healthcare professional for accurate evaluation and treatment.",
            disclaimer_style
        ))

        doc.build(story)
        pdf_buffer.seek(0)

        filename = f"BrainTumorReport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Error generating report: {str(e)}'}), 500

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)