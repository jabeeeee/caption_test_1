from transformers import AutoProcessor, AutoModelForCausalLM
import cv2
from PIL import Image

# Loading the models here..
processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")

# Capturwing a pic from camera using cv
cap = cv2.VideoCapture(0)  # 0 is defailt camera

ret, frame = cap.read()

# Check if the camera captured any thing
if not ret:
    print("Error: Unable to capture image from the camera")
else:
    # Convert frame to a image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Process the image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # AI magic captioning
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Print generated caption
    print(generated_caption)

# Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
