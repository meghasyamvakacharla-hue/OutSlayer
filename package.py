import zipfile
import os

def package_submission():
    zip_name = "submission_package.zip"
    files_to_include = [
        "best_model.pth", 
        "train.py", 
        "eval.py", 
        "dataset.py", 
        "app.py", 
        "requirements.txt",
        "do.md",
        "report.md",
        "loss_graph.png",
        "failure_case.png"
    ]
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file} to {zip_name}")
            else:
                print(f"WARNING: {file} not found. Skipping.")
                
    print(f"\nSuccessfully created {zip_name}!")

if __name__ == "__main__":
    package_submission()
