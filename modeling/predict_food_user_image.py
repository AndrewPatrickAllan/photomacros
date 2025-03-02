import sys
import os
sys.path.append(os.path.abspath('/Users/allan/Documents/GitHub/photomacros'))
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, BATCH_SIZE,NUM_EPOCHS,MEAN,STD
import torch
from modeling.predict import load_model_into_eval_model



# Set device globally
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")



from train import load_data, get_model_architecture,get_validation_transforms 



def load_class_labels():
    class_labels_path = MODELS_DIR / "class_names.txt"
    with open(class_labels_path, "r") as f:
        return [line.strip() for line in f.readlines()]
    

def predict_food_user_single_image(image_path,  model_path   ):
    """
    Predicts the food category of a given image using a trained PyTorch model.
    
    :param image_path: Path to the input image.
    :param model_path: path to a trained PyTorch model.
   
    :return: Predicted food category as a string.
    """

    model=load_model_into_eval_model(model_path)

    class_labels = load_class_labels()
 
    transform = get_validation_transforms()  #  image transformations (should match the preprocessing used in training)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)  # Forward pass
        predicted_class = torch.argmax(outputs, dim=1).item()  # Get class index
        
    return class_labels[predicted_class]

from pathlib import Path
from photomacros.config import MODELS_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE, MEAN, STD, BATCH_SIZE, NUM_EPOCHS


aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/croquettas.jpeg")
aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/avocado.jpeg")
aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/london_burger.jpeg")
aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/burger.jpeg")
aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/nachos2.jpeg")
aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/guac.jpeg")
aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/risotto.jpeg")
aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/sushi.jpeg")
aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_test_images/pizza.jpeg")

model_path=Path(MODELS_DIR / f"model_{NUM_EPOCHS}epochs_BetterModel_LR_Earlystop.pkl")
guess=predict_food_user_single_image(aa_test_guac_image_path, model_path)
print('guess=', guess)