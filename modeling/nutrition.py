from pathlib import Path
import pandas as pd

nutrition_path = Path("~/Documents/GitHub/photomacros/nutrition.csv")
nutrition_df = pd.read_csv(nutrition_path)
# print(nutrition_df.head())
# print(nutrition_df.info())


def get_nutrition_info(food_label, weight_grams=False):
    """
    Get nutrition information for a specific food item.

    Args:
        label (str): The food item to get nutrition information for.

    Returns:
        pd.DataFrame: DataFrame containing nutrition information for the specified food item.
    """
    df_specific = nutrition_df.loc[nutrition_df['label'] == food_label]
    if weight_grams != False:
        df_specific =  df_specific.loc[nutrition_df['weight'] == weight_grams]
    print(df_specific)
    return df_specific

# food_label='beef_tartare'
# get_nutrition_info(food_label, weight_grams=False)





