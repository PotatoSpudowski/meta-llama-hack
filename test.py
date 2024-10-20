import base64
import json
import re
import os
from dotenv import load_dotenv
import requests
import cv2
import numpy as np
import random

load_dotenv()
from groq import Groq
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, Field
import instructor
import io
import replicate
from PIL import ImageDraw, ImageFont  # noqa: E402
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import zipfile
import uuid
from minio import Minio
from minio.error import S3Error
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


AI_SERVICE = "groq"  # Change this to "together" to use Together AI
REPL = os.environ.get("REPL")
GROQ = os.environ.get("GROQ")
TOGETHER = os.environ.get("TOGETHER")


def get_complementary_color(color):
    # Ensure the color is not grey by checking its luminance
    luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    if luminance < 128:  # Darker colors
        return [255, 255, 255]  # Return white as complementary for dark colors
    else:  # Lighter colors
        return [0, 0, 0]  # Return black as complementary for light greys


def encode_image(image_path):
    with Image.open(image_path) as img:
        # Calculate the target width and height
        target_height = 720
        aspect_ratio = img.width / img.height
        target_width = int(target_height * aspect_ratio)

        # Resize the image if it's larger than 720p
        if img.height > target_height or img.width > target_width:
            img = img.resize((target_width, target_height), Image.LANCZOS)

        # Convert the image to RGB mode if it's not already
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save the image to a bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        # Encode the image
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def clean_prompt(prompt):
    prompt = re.sub(r"\*\*?", "", prompt)
    prompt = re.sub(r"^(Prompt:?\s*|Background:?\s*)", "", prompt, flags=re.IGNORECASE)
    prompt = prompt.strip('"')
    return prompt.strip()


def analyze_image(image_path, additional_info):
    groq_api_key = GROQ
    together_api_key = TOGETHER

    if AI_SERVICE == "groq":
        client = Groq(api_key=groq_api_key)
        model = "llama-3.2-90b-vision-preview"
    elif AI_SERVICE == "together":
        client = OpenAI(
            base_url="https://api.together.xyz/v1", api_key=together_api_key
        )
        model = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    else:
        raise ValueError("Invalid AI service specified")

    base64_image = encode_image(image_path)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # "text": f"Create a brief, simple background description for a product ad image. Suggest a complementary scene without mentioning the product. Use natural language in 1-2 short sentences. Consider: {additional_info}. Provide only the background prompt."
                        "text": f"Describe a simple, specific background for a {additional_info} ad. Don't mention the product directly. Use 1 short, clear sentence focusing on a setting where the product would typically be used or displayed. Provide only the background description.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        temperature=0.8,
        max_tokens=50,
        top_p=1,
        stream=False,
        stop=None,
    )

    generated_prompt = clean_prompt(completion.choices[0].message.content.strip())
    print("generated_prompt: " + generated_prompt)
    return generated_prompt


class ProductDescription(BaseModel):
    description: str = Field(
        ..., description="A concise description of the product based on the image"
    )


def get_product_description(image_path, additional_info):
    # print("Analyzing product...")
    groq_api_key = GROQ
    together_api_key = TOGETHER

    if AI_SERVICE == "groq":
        client = Groq(api_key=groq_api_key)
        client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)
        model = "llama-3.2-90b-vision-preview"
    elif AI_SERVICE == "together":
        client = OpenAI(
            base_url="https://api.together.xyz/v1", api_key=together_api_key
        )
        client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
        model = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    else:
        raise ValueError("Invalid AI service specified")

    # client = Groq(api_key=groq_api_key)
    # client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

    base64_image = encode_image(image_path)

    prompt = f"""
    Analyze the given image and provide:
    A concise description of the product visible in the image.
    
    Consider this additional information: {additional_info}
    """

    try:
        analysis = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            response_model=ProductDescription,
        )
        return analysis.description
    except Exception as e:
        print(f"Error analyzing product: {str(e)}")
        return None


def generate_creative(image_path, background_prompt):
    input_data = {
        "pixel": "768 * 768",
        "scale": 3,
        "prompt": background_prompt,
        "image_num": 4,
        "image_path": open(image_path, "rb"),
        "manual_seed": -1,
        "product_size": "0.5 * width",
        "guidance_scale": 8,
        "num_inference_steps": 20,
    }

    # print("Generating image...")
    output = replicate.run(
        "logerzhu/ad-inpaint:b1c17d148455c1fda435ababe9ab1e03bc0d917cc3cf4251916f22c45c83c7df",
        input=input_data,
    )

    image_urls = [file.url for file in output if hasattr(file, "url")]
    return image_urls


def post_process_text(text, max_length):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        text = text[: max_length - 3] + "..."
    return text


def post_process_cta(cta):
    valid_ctas = ["Shop Now", "Learn More", "Sign Up", "Book Now"]
    cta = cta.strip().title()
    return cta if cta in valid_ctas else "Shop Now"


class AdCopy(BaseModel):
    headline: str = Field(
        ..., max_length=25, description="Attention-grabbing headline for the ad"
    )
    body_copy: str = Field(
        ..., max_length=50, description="Compelling description of the product or offer"
    )
    cta: str = Field(
        ...,
        description="Call to action (choose from: Shop Now, Learn More, Sign Up, Book Now)",
    )


def generate_ad_copy(product_info, image_description):
    groq_api_key = GROQ
    client = Groq(api_key=groq_api_key)
    client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

    prompt = f"""
    Create compelling Facebook ad copy for the following product:
    
    Product Information: {product_info}
    Image Description: {image_description}
    
    Generate ad copy that fits the AdCopy model specifications.
    """

    try:
        ad_copy = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional copywriter specializing in Facebook ads.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=AdCopy,
            max_retries=3,
        )
        return ad_copy.model_dump()
    except Exception as e:
        print(f"Error generating ad copy: {str(e)}")
        return None


def generate_facebook_ad_copy(product_info, image_description):
    groq_api_key = GROQ
    client = Groq(api_key=groq_api_key)
    client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

    class FacebookAdCopy(BaseModel):
        headline: str = Field(
            ...,
            max_length=40,
            description="Attention-grabbing headline for the Facebook ad",
        )
        body_copy: str = Field(
            ...,
            max_length=125,
            description="Compelling description of the product or offer",
        )
        cta: str = Field(
            ...,
            description="Call to action (choose from: Shop Now, Learn More, Sign Up, Book Now)",
        )

    prompt = f"""
    Create compelling Facebook ad copy for the following product:
    
    Product Information: {product_info}
    Image Description: {image_description}
    
    Generate ad copy that fits the FacebookAdCopy model specifications. The copy should be optimized for Facebook ads, with a catchy headline, engaging body copy, and an appropriate call to action.
    """

    try:
        ad_copy = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional copywriter specializing in Facebook ads.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=FacebookAdCopy,
            max_retries=3,
        )
        return ad_copy.model_dump()
    except Exception as e:
        print(f"Error generating Facebook ad copy: {str(e)}")
        return None


class AdTargetingSuggestions(BaseModel):
    interests: list[str] = Field(
        ...,
        max_items=20,
        description="Suggested interests based on the product and target audience",
    )
    behaviors: list[str] = Field(
        ..., max_items=10, description="Potential behaviors of the target audience"
    )


def save_ad_copy(heading, subheading, idx, style):
    # Load the image
    image = Image.open("output.png")

    if image.mode == "P":
        image = image.convert("RGB")

    imagecv2 = cv2.imread("output.png")

    height, width, _ = imagecv2.shape
    top_20_percent_height = int(height * 0.2)

    top_region = imagecv2[0:top_20_percent_height, :]

    average_color = np.mean(top_region, axis=(0, 1)).astype(int)
    complementary_color = get_complementary_color(average_color)

    complementary_color_rgb = complementary_color[::-1]

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(f"fonts/{style}.ttf", 196)
    fontSmall = ImageFont.truetype(f"fonts/{style}.ttf", 96)

    text_color = tuple(int(c) for c in complementary_color_rgb)

    # Add first part of the headline text to the image
    words = heading.split()

    # Check the number of words and split if necessary
    if len(words) > 4:
        part1 = " ".join(words[:4])  # First 4 words
        part2 = " ".join(words[4:])  # Remaining words
        draw.text((100, 50), part1, font=font, fill=text_color)
        draw.text(
            (100, 300), part2, font=font, fill=text_color
        )  # Position below the first part
    else:
        draw.text((100, 50), heading, font=font, fill=text_color)

    # Calculate position for body copy to be at the bottom of the image
    body_copy_position = (
        (image.width / 4) - 50,
        image.height - 100,
    )  # Adjust Y position as needed

    # Add body copy text to the image
    draw.text(body_copy_position, subheading, font=fontSmall, fill=text_color)

    # Save or display the modified image
    image.save(f"output_{idx}.png")


def generate_ad_targeting(user_input, ad_image_description, ad_description):
    # print("Generating ad targeting suggestions...")
    groq_api_key = GROQ
    client = Groq(api_key=groq_api_key)
    client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)

    prompt = f"""
    Generate Facebook ad targeting suggestions for an MSME (Micro, Small, and Medium Enterprise) seller using Advantage+ audience for sales ads:

    User Input:
    - Location: {user_input['location']}
    - Budget: {user_input['budget']}
    - Age Group: {user_input['age_group']}
    - Business Category: {user_input['business_category']}
    - Target Gender: {user_input['target_gender']}

    Ad Image Description: {ad_image_description}
    Ad Description: {ad_description}

    Provide targeting suggestions that fit the AdTargetingSuggestions model specifications.
    Focus on interests and behaviors that are most relevant for detailed targeting in Advantage+ audience sales ads.
    The suggestions should be specific and aligned with Facebook's targeting options.
    """

    try:
        targeting_suggestions = client.chat.completions.create(
            model="llama-3.2-90b-text-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Facebook ad targeting for small businesses, specializing in Advantage+ audience for sales ads.",
                },
                {"role": "user", "content": prompt},
            ],
            response_model=AdTargetingSuggestions,
            max_retries=3,
        )
        return targeting_suggestions.model_dump()
    except Exception as e:
        print(f"Error generating ad targeting suggestions: {str(e)}")
        return None


def generate_translations_for_image(heading, subheading):
    language_codes = [
        "hi-IN",
        "bn-IN",
        "kn-IN",
        "ml-IN",
        "mr-IN",
        "od-IN",
        "pa-IN",
        "ta-IN",
        "te-IN",
        "gu-IN",
    ]
    fonts = [
        "Hindi",
        "bengali",
        "kannada",
        "mal",
        "devnag",
        "oriya",
        "gur",
        "tam",
        "tel",
        "guj",
    ]
    url = "https://api.sarvam.ai/translate"
    headline_payload = {
        "input": heading,
        "source_language_code": "en-IN",
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": False,
    }
    subheading_payload = {
        "input": subheading,
        "source_language_code": "en-IN",
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": False,
    }
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": os.environ.get("SRVM"),
    }
    for idx, lang_code in enumerate(language_codes):
        # Create a copy of the payload and update the target language code
        payload = headline_payload.copy()
        payload["target_language_code"] = lang_code
        response = requests.post(url, json=payload, headers=headers)
        translated_heading = response.json()

        payload = subheading_payload.copy()
        payload["target_language_code"] = lang_code
        response = requests.post(url, json=payload, headers=headers)
        translated_subheading = response.json()

        if (
            "translated_text" in translated_heading
            and "translated_text" in translated_subheading
        ):
            save_ad_copy(
                translated_heading["translated_text"],
                translated_subheading["translated_text"],
                idx,
                fonts[idx],
            )


def zip_outputs():
    files = [f"output_{i}.png" for i in range(10)]
    zip_file_name = f"{uuid.uuid4()}.zip"
    with zipfile.ZipFile(zip_file_name, "w") as zipf:
        for file in files:
            # Check if the file exists before adding
            if os.path.isfile(file):
                zipf.write(file, os.path.basename(file))
            else:
                print(f"File {file} does not exist and will be skipped.")
    return zip_file_name


def get_reach():
    reach_min = random.randint(30000, 50000)
    reach_max = random.randint(100000, 120000)
    if reach_min > reach_max:
        reach_min, reach_max = reach_max, reach_min
    return [reach_min, reach_max]


def get_conversions():
    conversions_min = random.randint(10, 20)
    conversions_max = random.randint(50, 200)
    if conversions_min > conversions_max:
        conversions_min, conversions_max = conversions_max, conversions_min
    return [conversions_min, conversions_max]


def upload_to_minio(file_name):
    client = Minio(
        os.environ.get("MINIO_ENDPOINT"),
        access_key=os.environ.get("MINIO_ACCESS_KEY"),
        secret_key=os.environ.get("MINIO_SECRET_KEY"),
        secure=True,  # Use True for HTTPS
    )
    # The file to upload
    source_file = file_name
    bucket_name = os.environ.get("MINIO_BUCKET")
    destination_file = file_name

    # Check if the bucket exists, if not create it
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Created bucket: {bucket_name}")
    else:
        print(f"Bucket {bucket_name} already exists")

    # Upload the file
    try:
        client.fput_object(bucket_name, destination_file, source_file)
        print(
            f"{source_file} successfully uploaded as object {destination_file} to bucket {bucket_name}"
        )
    except S3Error as err:
        print("Error occurred.", err)
    pass


def create_facebook_ad(image_path, additional_info, product_info, user_input):
    # Step 1: Analyze the image and generate background prompt
    print("Step 1: Analyzing image and generating background prompt...")
    background_prompt = analyze_image(image_path, additional_info)

    # Step 2: Get product description
    print("Step 2: Getting product description...")
    product_description = get_product_description(image_path, additional_info)

    # Step 3: Generate the creative using Replicate
    print("Step 3: Generating creative using Replicate...")
    creative_urls = generate_creative(image_path, background_prompt)

    # Step 4: Generate ad copy based on the product info and background prompt
    print("Step 4: Generating ad copy...")
    ad_copy = generate_ad_copy(product_info, product_description)

    print("Step 5: Generating Facebook ad copy...")
    facebook_ad_copy = generate_facebook_ad_copy(product_info, product_description)

    # Step 6: Generate ad targeting suggestions
    print("Step 6: Generating ad targeting suggestions...")
    targeting_suggestions = generate_ad_targeting(
        user_input, product_description, facebook_ad_copy["body_copy"]
    )

    # Step 7: Downloading images from url
    print("Step 7: Compressing generted output.")
    url = creative_urls[1]
    image = Image.open(requests.get(url, stream=True).raw)
    image.save("output.png")

    result = {
        "background_prompt": background_prompt,
        "product_description": product_description,
        "creative_urls": creative_urls,
        "ad_copy": ad_copy,
        "facebook_ad_copy": facebook_ad_copy,
        "targeting_suggestions": targeting_suggestions,
        "reach": get_reach(),
        "conversions": get_conversions(),
    }
    print("Model work completed")
    print(json.dumps(result, indent=2))

    # Step 8: Generating Translations
    print("Step 8: Generating Translations.")
    generate_translations_for_image(ad_copy["headline"], ad_copy["body_copy"])

    # Step 9: Zip outputs
    file_name = zip_outputs()

    # Step 10: Upload zip to minio
    upload_to_minio(file_name=file_name)

    result["download"] = f"https://s3.heyhomie.dev/dev/{file_name}"
    del result["creative_urls"]
    return result


# Example usage
# if __name__ == "__main__":
#     image_path = "dummy.png"
#     additional_info = "Gloves"
#     product_info = "Gradening gloves with attached claws"
#     user_input = {
#         "location": "USA",
#         "budget": "500",
#         "age_group": "18-50",
#         "business_category": "Gardening",
#         "target_gender": "all",
#     }

#     result = create_facebook_ad(image_path, additional_info, product_info, user_input)
#     print(json.dumps(result, indent=2))


@app.route("/infer", methods=["POST"])
def inference():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected image file"}), 400

    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image.save(image_path)

        additional_info = request.form.get("additional_info", "Gloves")
        product_info = request.form.get(
            "product_info", "Gardening gloves with attached claws"
        )
        user_input = {
            "location": request.form.get("location", "USA"),
            "budget": request.form.get("budget", "500"),
            "age_group": request.form.get("age_group", "18-50"),
            "business_category": request.form.get("business_category", "Gardening"),
            "target_gender": request.form.get("target_gender", "all"),
        }

        result = create_facebook_ad(
            image_path, additional_info, product_info, user_input
        )
        return jsonify(result)

    return jsonify({"error": "Error processing the image"}), 500


if __name__ == "__main__":
    app.run(debug=True)
