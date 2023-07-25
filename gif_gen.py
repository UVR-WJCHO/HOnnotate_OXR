import os
from PIL import Image

def convert_images_to_gif(directory, output_file):
    images = []
    
    # Get all image files in the directory
    for idx, filename in enumerate(os.listdir(directory)):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            filepath = os.path.join(directory, 'output'+str(idx)+'.png')
            
            # Open the image and add it to the list
            
            try:
                image = Image.open(filepath)
                image = image.resize((round(image.size[0]*0.8), round(image.size[1]*0.8)))
                images.append(image)
            except Exception as e:
                continue
            
    # Save the images as a GIF file

    images[0].save(output_file, save_all=True, append_images=images[1:], duration=10, loop=0)

# Example usage
image_directory = "/minjay/HOnnotate_OXR/vis"
output_gif = "./output.gif"

convert_images_to_gif(image_directory, output_gif)
