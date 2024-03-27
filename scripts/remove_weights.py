import os
import glob
import shutil
# shutil.rmtree('/path/to/your/dir/')

if __name__ == "__main__":
    weight_dir = "../weights_fmsys/"
    
    for dataset_model_folder in os.listdir(weight_dir):
        dataset_model_dir = os.path.join(weight_dir, dataset_model_folder)
        for exp_folder in os.listdir(dataset_model_dir):
            exp_dir = os.path.join(dataset_model_dir, exp_folder)
            exp_dir_files = glob.glob(f"{exp_dir}/*.pt")
            if len(exp_dir_files) == 0:
                print(exp_dir)
                shutil.rmtree(exp_dir)
        if len(os.listdir(dataset_model_dir)) == 0:
            shutil.rmtree(dataset_model_dir)
        
            