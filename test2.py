from PIL import Image, ImageDraw, ImageFont

# Define headline parts
headline_part1 = "Dig & Plant with"
headline_part2 = "Ease"
body_copy = "Grip & dig with ease gloves with claws!"
cta = "Shop Now"

# Load the image
image = Image.open("dummy.png")

# Create a drawing context
draw = ImageDraw.Draw(image)

# Define fonts
font = ImageFont.truetype("fonts/Roboto-Bold.ttf", 76)
fontSmall = ImageFont.truetype("fonts/Roboto-Bold.ttf", 36)

# Set text color to black
text_color = (0, 0, 0)  # RGB for black

# Add first part of the headline text to the image
draw.text((50, 50), headline_part1, font=font, fill=text_color)

# Add second part of the headline text below the first part
draw.text(
    (50, 130), headline_part2, font=font, fill=text_color
)  # Adjust Y position as needed

# Calculate position for body copy to be at the bottom of the image
body_copy_position = (
    (image.width / 4) - 50,
    image.height - 100,
)  # Adjust Y position as needed

# Add body copy text to the image
draw.text(body_copy_position, body_copy, font=fontSmall, fill=text_color)

# Save or display the modified image
image.save("output.png")
