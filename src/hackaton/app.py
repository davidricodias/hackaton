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

    try:
        logger.info("Loading AI detection model...")
        ai_detector = pipeline("image-classification", model="Organika/sdxl-detector")
        logger.info("AI detection model loaded successfully")

        # Determine available device - prefer CUDA, then MPS, fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Set appropriate dtype based on device
        if device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32  # Use float32 for MPS and CPU
        
        logger.info(f"Using dtype: {dtype}")

        logger.info("Loading captioning model...")
        model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        
        # Load processor first
        try:
            caption_processor = AutoProcessor.from_pretrained(model_path)
            logger.info("Caption processor loaded successfully")
        except Exception as e:
            logger.error(f"Error loading caption processor: {str(e)}")
            raise
            
        # Then load the model with proper error handling
        try:
            caption_model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=device if device == "cuda" else None
            )
            logger.info("Caption model loaded successfully")
            
            # Move to device if not CUDA (for CUDA, device_map handles it)
            if device != "cuda":
                caption_model = caption_model.to(device)
                
            logger.info(f"Models loaded and running on {device}")
        except Exception as e:
            logger.error(f"Error loading caption model: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
        # Re-raise to prevent app from starting with incomplete models
        raise

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
        compensation = generate_compensation(image, description)
        
        processing_time = time.time() - start_time
        logger.info(f"Generated compensation: {compensation} in {processing_time:.2f} seconds")

        return jsonify({
            'compensation': compensation,
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

def generate_compensation(image, description):
    try:
        logger.info("Starting compensation generation for image")
        # Set a timeout value for model operations
        timeout_seconds = 30
        
        # Convert PIL image to processor's expected input format
        processor_inputs = caption_processor(images=image, return_tensors="pt").to(caption_model.device)
        logger.info("Image processed for compensation generation")

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
        logger.info("Prompt prepared for model")

        # Get the input_ids from the processed image and prompt
        inputs = caption_processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(caption_model.device)
        logger.info("Inputs prepared and moved to device")

        # Fix tensor type issue - ensure input_ids and attention_mask remain long/int type
        # while other tensors maintain the correct dtype for the device
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                if key in ['input_ids', 'attention_mask']:
                    inputs[key] = inputs[key].to(torch.long)  # Ensure these are long type
                elif caption_model.device.type == "mps":
                    # For MPS devices, ensure we use float32
                    inputs[key] = inputs[key].to(torch.float32)
                elif caption_model.device.type == "cuda":
                    # For CUDA devices, we can use bfloat16
                    inputs[key] = inputs[key].to(torch.bfloat16)

        # Generate the compensation recommendation with timeout
        logger.info("Generating compensation recommendation from multimodal model")
        
        # Using a smaller max_new_tokens value to reduce processing time
        generated_ids = caption_model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=32,  # Reduced from 64
            num_beams=1  # Use a simpler beam search
        )
        logger.info("Model generation completed")

        generated_text = caption_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        logger.info(f"Generated text: {generated_text}")

        # Extract just the assistant's response
        assistant_prefix = "Assistant: "
        if assistant_prefix in generated_text:
            result = generated_text.split(assistant_prefix)[1]
        else:
            result = generated_text
            
        # Ensure we have a valid dollar amount
        if not result.strip():
            return "$150.00"  # Default fallback if empty result
            
        return result
    except Exception as e:
        logger.error(f"Error in generate_compensation: {str(e)}")
        # Return a fallback compensation in case of errors
        return "$150.00"

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
    # while other tensors maintain the correct dtype for the device
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            if key in ['input_ids', 'attention_mask']:
                inputs[key] = inputs[key].to(torch.long)  # Ensure these are long type
            elif caption_model.device.type == "mps":
                # For MPS devices, ensure we use float32
                inputs[key] = inputs[key].to(torch.float32)
            elif caption_model.device.type == "cuda":
                # For CUDA devices, we can use bfloat16
                inputs[key] = inputs[key].to(torch.bfloat16)

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
