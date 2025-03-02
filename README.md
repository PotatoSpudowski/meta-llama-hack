# Meta Llama Hack: AI-driven Facebook Ad Automation

Meta Llama Hack is an end-to-end framework that leverages advanced AI models to analyze product images and automatically generate ad creatives, copy, and targeting suggestions tailored for Facebook campaigns. This repository integrates computer vision, natural language processing, and translation services to provide a seamless, automated workflow for creating compelling ad assets.

## Key Features

- **Image Analysis & Preprocessing**: Resizes and optimizes input images, calculates complementary colors, and prepares images for AI processing.
- **Background & Product Description Generation**: Uses AI (via Groq or Together) to generate a concise background prompt and a detailed product description based on the input image.
- **Ad Creative Generation**: Integrates with Replicate to generate creative ad images that align with the background prompt.
- **Ad Copy & Facebook-Specific Ad Copy**: Automatically generates engaging ad copy along with Facebook-optimized versions, including headlines, body copy, and calls-to-action.
- **Multi-Lingual Translations**: Utilizes the Sarvam API to translate ad copy into multiple languages, ensuring wider reach across diverse markets.
- **Targeting Suggestions**: Provides ad targeting insights tailored to MSME sellers by analyzing product description and ad imagery.
- **Output Packaging & Upload**: Compresses output images into a zip file and uploads them to a Minio S3 bucket, yielding a download link for easy access.

## Project Structure

```
meta-llama-hack/
├── fonts/                # Font files used for overlaying text on images
├── static/               # Static resources for any front-end components
├── uploads/              # Directory for storing uploaded input images
├── README.md             # This file
├── index.css             # Front-end styles (if applicable)
├── sarvam.py             # Script for handling multi-lingual translations via Sarvam API
└── test.py               # Main Flask application and ad-generation pipeline
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd meta-llama-hack
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies** (create a `requirements.txt` file or install dependencies manually):
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:

   Create a `.env` file in the root directory with the following keys:
   
   ```ini
   # API keys for AI services
   GROQ=<your_groq_api_key>
   TOGETHER=<your_together_api_key>
   SRVM=<your_sarvam_api_subscription_key>
   
   # Minio S3 configuration
   MINIO_ENDPOINT=<minio_endpoint>
   MINIO_ACCESS_KEY=<minio_access_key>
   MINIO_SECRET_KEY=<minio_secret_key>
   MINIO_BUCKET=<minio_bucket_name>
   ```

5. **Ensure required fonts are available** in the `fonts/` directory. These fonts are used to overlay ad text on the generated images.

## Usage

### Running the Inference API

The main application is a Flask server that exposes an inference endpoint.

```bash
python test.py
```

This will start the Flask server in debug mode. The following endpoint is available:

- **POST /infer**: Accepts an image file along with form data (e.g., additional_info, product_info, location, budget, age_group, business_category, target_gender) and performs the full ad generation pipeline.

### Example Request

You can test the API using curl:

```bash
curl -X POST \
  -F "image=@/path/to/your/image.jpg" \
  -F "additional_info=Gloves" \
  -F "product_info=Gardening gloves with attached claws" \
  -F "location=USA" \
  -F "budget=500" \
  -F "age_group=18-50" \
  -F "business_category=Gardening" \
  -F "target_gender=all" \
  http://localhost:5000/infer
```

The API will return a JSON object containing:

- Generated background prompt
- Product description
- Creative ad URLs
- Standard and Facebook-optimized ad copy
- Targeting suggestions
- Reach and conversion estimates
- A download link for the packaged output zip file

## How It Works

1. **Image Preprocessing**: The input image is resized, optimized, and its dominant color is analyzed to determine text overlay colors.
2. **Content Generation**: Using AI services via Groq or Together, the system generates a background prompt for creative scenes and extracts detailed product descriptions from the image.
3. **Ad Creative & Copy Generation**: The system then creates ad visuals via a Replicate model and generates both standard and Facebook-specific ad copy.
4. **Translation & Localization**: Ad copy is translated into multiple languages using the Sarvam API, broadening its applicability.
5. **Packaging & Upload**: Final outputs (images with overlay text) are zipped and uploaded to a Minio S3 bucket for easy sharing.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix (e.g., `git checkout -b feature/your-feature`).
3. Commit your changes and push to your fork.
4. Submit a pull request detailing your changes.

## License

[Specify License Here]

## Contact

For questions or suggestions, please open an issue or contact the maintainer.
