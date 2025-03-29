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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
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
        # Load image and run multimodal analysis
        image = Image.open(filepath)
        logger.info("Running multimodal analysis with image and description")

        # Generate compensation based on image and text
        compensation = generate_compensation(image, description)
        logger.info(f"Generated compensation: {compensation}")

        return jsonify({
            'compensation': compensation
        })
    except Exception as e:
        logger.error(f"Error calculating compensation: {str(e)}")
        return jsonify({'error': str(e)})
    finally:
        # Clean up temporary file
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Temporary file {filepath} removed")

def generate_compensation(image, description):
    # Convert PIL image to processor's expected input format
    processor_inputs = caption_processor(images=image, return_tensors="pt").to(caption_model.device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"This is an image of a damaged item with the following description: {description}. Based on the image and description, calculate a fair compensation amount in USD for this damage. Only respond with a dollar amount."},
                {"type": "image"}
            ]
        },
    ]

    # Generate the prompt text
    prompt = caption_processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Get the input_ids from the processed image and prompt
    inputs = caption_processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(caption_model.device)

    # Fix tensor type issue - ensure input_ids and attention_mask remain long/int type
    # while other tensors can be bfloat16
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            if key in ['input_ids', 'attention_mask']:
                inputs[key] = inputs[key].to(torch.long)  # Ensure these are long type
            else:
                inputs[key] = inputs[key].to(torch.bfloat16)  # Other tensors can be bfloat16

    # Generate the compensation recommendation
    logger.info("Generating compensation recommendation from multimodal model")
    generated_ids = caption_model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64
    )

    generated_text = caption_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    # Extract just the assistant's response
    assistant_prefix = "Assistant: "
    if assistant_prefix in generated_text:
        return generated_text.split(assistant_prefix)[1]
    return generated_text

def generate_caption(image):
    # Convert PIL image to processor's expected input format
    processor_inputs = caption_processor(images=image, return_tensors="pt").to(caption_model.device)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you describe this image?"},
            ]
        },
    ]

    # Generate the prompt text
    prompt = caption_processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Get the input_ids from the processed image and prompt
    inputs = caption_processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(caption_model.device)

    # Fix tensor type issue - ensure input_ids and attention_mask remain long/int type
    # while other tensors can be bfloat16
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            if key in ['input_ids', 'attention_mask']:
                inputs[key] = inputs[key].to(torch.long)  # Ensure these are long type
            else:
                inputs[key] = inputs[key].to(torch.bfloat16)  # Other tensors can be bfloat16

    # Generate the caption
    generated_ids = caption_model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=64
    )

    generated_text = caption_processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    # Extract just the assistant's response
    assistant_prefix = "Assistant: "
    if assistant_prefix in generated_text:
        return generated_text.split(assistant_prefix)[1]
    return generated_text

def run_app():
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5001)

if __name__ == '__main__':
    run_app()
