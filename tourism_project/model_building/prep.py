# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi



# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/SilviaMartin/Visitwithus/tourism.csv"
tour_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

#The column Gender is having two values to represent Female category, so combining them into one category to represent Female

tour_dataset["Gender"].replace("Fe Male","Female",inplace=True)

#Discarding the CustomerID column as it is a unique identifier
# FIX: Assign the result of drop back to tour_dataset to make it effective
tour_dataset = tour_dataset.drop(columns="CustomerID",axis=1)

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact', 'Occupation','Gender','ProductPitched','MaritalStatus', 'Designation'
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tour_dataset[numeric_features + categorical_features]

# Define target variable
y = tour_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

# Save to local CSVs
Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files_to_upload = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
repo_id_dataset = "SilviaMartin/Visitwithus" # Assuming this is the correct dataset repo ID

# FIX: Explicitly delete files from Hugging Face Hub before re-uploading
for file_path in files_to_upload:
    try:
        api.delete_file(
            path_in_repo=file_path,
            repo_id=repo_id_dataset,
            repo_type="dataset",
            commit_message=f"Delete {file_path} before re-upload"
        )
        print(f"Successfully deleted {file_path} from Hugging Face Hub.")
    except HfApi.HTTPError as e:
        # Ignore 404 errors (file not found), which means it didn't exist to delete
        if e.response.status_code == 404:
            print(f"{file_path} not found on Hugging Face Hub, skipping deletion.")
        else:
            print(f"Error deleting {file_path} from Hugging Face Hub: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while trying to delete {file_path}: {e}")

# Now upload the new files, ensuring a fresh commit
for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id_dataset,
        repo_type="dataset",
        commit_message=f"Update {file_path} with new data splits", # Add a clear commit message
    )
    print(f"Uploaded {file_path} to Hugging Face Hub.")
