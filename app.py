from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms

app = Flask(__name__)

# Define a static folder for serving images
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
model.eval()

# Define a function to process the image
def remove_background(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Resize and preprocess the image
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()

    # Resize the mask back to the original image size
    mask_resized = cv2.resize(output_predictions, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a binary mask for humans (class ID 15 corresponds to humans in COCO)
    human_mask = (mask_resized == 15).astype(np.uint8) * 255

    # Apply the mask to extract the human
    foreground = cv2.bitwise_and(image, image, mask=human_mask)

    # Replace the background with white
    background_white = np.ones_like(image) * 255
    inverse_mask = cv2.bitwise_not(human_mask)
    background_with_human = cv2.bitwise_and(background_white, background_white, mask=inverse_mask)
    result_image = cv2.add(background_with_human, foreground)

    return result_image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/remove-bg', methods=['POST'])
def upload_and_process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            original_image = f"/static/uploads/{filename}"

            # Generate processed image
            result_image = remove_background(file_path)

            result_filename = f"processed_{filename.replace('.jpg', '.png')}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_image)

            processed_image = f"/static/uploads/{result_filename}"

            return render_template('result.html', original_image=f"/static/uploads/{filename}", processed_image=f"/static/uploads/{result_filename}")

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)

