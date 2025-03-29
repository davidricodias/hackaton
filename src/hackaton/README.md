# AI Image Analysis Tool

This application provides a web interface for analyzing images to:
1. Detect if an image is AI-generated or human-generated
2. Generate descriptive captions for uploaded images

## Features

- Drag-and-drop image upload interface
- AI-generated content detection with confidence scores
- Image captioning/description generation
- Clean, responsive interface

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application:

```bash
python -m hackaton
```

Or directly:

```bash
python src/hackaton/app.py
```

Then navigate to http://localhost:5000 in your web browser.

## Models Used

- AI Detection: `Organika/sdxl-detector` - A specialized model for detecting AI-generated images
- Image Captioning: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` - A vision-language model for generating image descriptions

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Flask
- Internet connection (for initial model downloads)
- ~4GB of RAM for model operations

## Notes

The first run will download the required models which may take some time depending on your internet connection.
