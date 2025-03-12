import sys
import os
sys.path.append(os.path.abspath('/Users/allan/Documents/GitHub/photomacros'))
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from photomacros.config import MODELS_DIR, NUM_EPOCHS #, PROCESSED_DATA_DIR, IMAGE_SIZE, BATCH_SIZE,NUM_EPOCHS,MEAN,STD
import torch
from modeling.predict import load_model_into_eval_model
from modeling.nutrition import get_nutrition_info

from pathlib import Path




# Set device globally
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set device globally
device = torch.device("cpu")
print(f"Using device: {device}")

# Set device globally
device = torch.device("mps")
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

    image_size=228/2.0
 
    transform = get_validation_transforms(image_size)  #  image transformations (should match the preprocessing used in training)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)  # Forward pass
        predicted_class = torch.argmax(outputs, dim=1).item()  # Get class index

    predicted_food_string=class_labels[predicted_class]

    # Get nutrition information for the predicted food item
    df_specific=get_nutrition_info(predicted_food_string, weight_grams=False)
        
    return class_labels[predicted_class]



aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/croquettas.jpeg")
# aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/avocado.jpeg")
# aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/london_burger.jpeg")
# aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/burger.jpeg")
# aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/nachos2.jpeg")
# aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/guac.jpeg")
# aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/risotto.jpeg")
# aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/sushi.jpeg")
# aa_test_guac_image_path = os.path.expanduser("~/Documents/GitHub/photomacros/all_user_phone_test_images/pizza.jpeg")

model_path=Path(MODELS_DIR / f"model_{NUM_EPOCHS}epochs_BetterModel_LR_Earlystop_pretrainedDenseNet.pkl")
guess=predict_food_user_single_image(aa_test_guac_image_path, model_path)
print('guess=', guess)