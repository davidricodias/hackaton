from flask import Flask, render_template, request, jsonify
import os
import torch
from PIL import Image
from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
import base64
import io
import tempfile
import uuid
import logging
import time
import json
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app with proper static folder
app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global models
ai_detector = None
caption_model = None
caption_processor = None

def load_models():
    global ai_detector, caption_model, caption_processor

    logger.info("Loading AI detection model...")
    ai_detector = pipeline("image-classification", model="Organika/sdxl-detector")

    logger.info("Loading captioning model...")
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    caption_processor = AutoProcessor.from_pretrained(model_path)
    caption_model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    # Move to GPU/MPS if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    caption_model = caption_model.to(device)
    logger.info(f"Models loaded and running on {device}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_ai():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file temporarily
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    logger.info(f"Image saved temporarily at {filepath}")

    try:
        # Load image and run AI detection
        image = Image.open(filepath)
        logger.info("Running AI detection on uploaded image")
        results = ai_detector(image)
        logger.info(f"AI detection results: {results}")

        # Check if the image is likely AI-generated
        ai_score = 0
        for result in results:
            if result['label'].lower() == 'artificial':
                ai_score = result['score']

        if ai_score > 0.5:  # If AI score > 50%
            logger.info(f"Image detected as AI-generated with score {ai_score}")
            return jsonify({
                'is_ai': True,
                'detection_results': results,
                'message': 'This image appears to be AI-generated. Cannot process for compensation.'
            })
        else:
            logger.info(f"Image detected as likely real with AI score {ai_score}")
            # For non-AI images, we don't generate a caption here
            # The user will provide the description in the next step
            return jsonify({
                'is_ai': False,
                'detection_results': results,
                'message': 'Image appears to be real. Please provide a description.'
            })
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)})
    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Temporary file {filepath} removed")

@app.route('/calculate_compensation', methods=['POST'])
def calculate_compensation():
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'})

    if 'description' not in request.form:
        logger.error("No description provided")
        return jsonify({'error': 'No description provided'})

    description = request.form['description']
    logger.info(f"Received description: {description}")

    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file temporarily
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    logger.info(f"Image saved temporarily at {filepath}")

    try:
        # Set a timeout limit for the entire process
        start_time = time.time()

        # Load image
        image = Image.open(filepath)
        image_path = "./processing.jpg"
        shutil.copy(filepath, image_path)
        logger.info("Image loaded successfully")

        # Resize image if it's too large to improve processing time
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))
            logger.info("Image resized for faster processing")

        # Run multimodal analysis
        logger.info("Running multimodal analysis with image and description")

        # Check model loading status
        if caption_model is None or caption_processor is None:
            logger.error("Models not properly loaded")
            return jsonify({'error': 'Internal server error - models not loaded'})

        # Generate compensation based on image and text
        compensation = generate_compensation(image_path, description)
        description = generate_description(image_path, description)

        processing_time = time.time() - start_time
        logger.info(f"Generated compensation: {compensation} in {processing_time:.2f} seconds")

        return jsonify({
            'compensation': compensation,
            'description': description,
            'processing_time': f"{processing_time:.2f} seconds"
        })
    except Exception as e:
        logger.error(f"Error calculating compensation: {str(e)}")
        # Return a fallback compensation in case of errors
        return jsonify({
            'compensation': '$150.00',
            'error': 'Error processing image, providing fallback compensation',
            'details': str(e)
        })
    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Temporary file {filepath} removed")

def generate_compensation(image_path, description):
    global ai_detector, caption_model, caption_processor
    try:
        logger.info("Starting compensation generation for image")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": "Estimate the cost to repair the damages in the image. Just give me a dollar amount."},
                ]
            },
        ]

        inputs = caption_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(caption_model.device, dtype=torch.bfloat16)

        generated_ids = caption_model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        logging.info(generated_texts)
        result = ""
        assistant_prefix = "Assistant: "
        if assistant_prefix in generated_texts:
            result = generated_texts.split(assistant_prefix)[1]
        else:
            result = generated_texts
        return result
    except Exception as e:
        logger.error(f"Error in generate_compensation: {str(e)}")
        # Return a fallback compensation in case of errors
        return "$150.00"

def generate_description(image_path, description):
    global ai_detector, caption_model, caption_processor
    try:
        logger.info("Starting description generation for image")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": f"With this description: {description} generate a list of the repairs needed to repair all the damages."},
                ]
            },
        ]

        inputs = caption_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(caption_model.device, dtype=torch.bfloat16)

        generated_ids = caption_model.generate(**inputs, do_sample=False, max_new_tokens=128)
        generated_texts = caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        logging.info(generated_texts)
        result = ""
        assistant_prefix = "Assistant: "
        if assistant_prefix in generated_texts:
            result = generated_texts.split(assistant_prefix)[1]
        else:
            result = generated_texts
        return result
    except Exception as e:
        logger.error(f"Error in generate_compensation: {str(e)}")
        # Return a fallback compensation in case of errors
        return "No description available :("


@app.route('/submit_claim', methods=['POST'])
def submit_claim():
    # Check if all required fields are provided
    required_fields = ['full_name', 'email', 'insurance_number']
    missing_fields = [field for field in required_fields if field not in request.form]

    if missing_fields:
        logger.error(f"Missing required fields in claim submission: {missing_fields}")
        return jsonify({'error': f"Missing required fields: {', '.join(missing_fields)}"})

    # Get form data
    full_name = request.form['full_name']
    email = request.form['email']
    insurance_number = request.form['insurance_number']
    description = request.form.get('description', '')
    compensation = request.form.get('compensation', '$0.00')

    logger.info(f"Received claim submission from {full_name} ({email})")

    # Check if image file is provided
    image_data = None
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            # Save file information
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"Claim image saved temporarily at {filepath}")

            # You could store the file path or upload to a storage service here
            # For demonstration, we'll just read the file and convert to base64
            try:
                with open(filepath, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')

                # Clean up temporary file after reading
                os.remove(filepath)
                logger.info(f"Temporary file {filepath} removed after reading")
            except Exception as e:
                logger.error(f"Error processing claim image: {str(e)}")
                if os.path.exists(filepath):
                    os.remove(filepath)

    # In a real application, you would store this information in a database
    # For demonstration purposes, we'll just log it and save to a JSON file
    claim_data = {
        'id': str(uuid.uuid4()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'full_name': full_name,
        'email': email,
        'insurance_number': insurance_number,
        'description': description,
        'compensation': compensation,
        'status': 'pending',
        # 'image_data': image_data  # Uncomment to store image data (could be large)
    }

    # Save claim to a file (in a real app, this would go to a database)
    claims_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'claims')
    os.makedirs(claims_dir, exist_ok=True)

    claim_file = os.path.join(claims_dir, f"claim_{claim_data['id']}.json")
    with open(claim_file, 'w') as f:
        json.dump(claim_data, f, indent=4)

    logger.info(f"Claim saved to {claim_file}")

    # In a real application, you would send an email notification here
    logger.info(f"Would send confirmation email to {email}")

    return jsonify({
        'success': True,
        'message': 'Claim submitted successfully. We will contact you shortly.',
        'claim_id': claim_data['id']
    })

def run_app():
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    run_app()
