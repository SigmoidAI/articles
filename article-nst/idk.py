from PIL import Image
import os

input_folder = "images"  # Replace with the path to your input folder
output_folder = "images"  # Replace with the path to your output folder

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)

    # Check if the file is an image
    if os.path.isfile(input_path) and any(filename.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".webp", ".jfif", ".bmp", ".tiff", ".gif")):
        try:
            # Open the image
            image = Image.open(input_path)

            # Save the image as PNG in the output folder
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            image.save(output_path, "PNG")
            print(f"Converted: {filename} to {os.path.basename(output_path)}")
        except Exception as e:
            print(f"Error converting {filename}: {str(e)}")
    else:
        print(f"Ignored: {filename}")

print("Conversion complete.")
