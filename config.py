import os

BASE_DIR = r"C:\Digital Tech\PPM_MODEL_CHECKER\final_new_ai_dph_chl_py"

DATASET_PATH = os.path.join(BASE_DIR,"dataset")

MODEL_PATH = os.path.join(BASE_DIR,"models","chlorine_model.pkl")

PPM_MAPPING = {

    "0_PPM":0.0,
    "0.2_to_1_PPM":0.6,
    "1_to_1.5_PPM":1.25,
    "1.5_to_2_PPM":1.75,
    "2_to_2.5_PPM":2.25,
    "2.5_to_3_PPM":2.75,
    "3_to_3.5_PPM":3.25,
    "3.5_to_3.8 PPM":3.65
}