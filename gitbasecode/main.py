from data_manager import load_dataset, load_train, load_val, load_test
from gitbase import load_model_pretrained, transforms, compute_metrics, defineTrainingArgs, dotrain, generateCaption, generateCaptionPretrained, dotrainWOFineTuning
from src_data.utils import write_json
from src_data.utils import get_data_version
from utils import get_save_file_path, set_data_version
import pandas as pd
import os
from PIL import Image
import io

from fastapi import APIRouter, FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from api import api_router

DATA_DIRECTORY = '../data'

#data_version = get_data_version(DATA_DIRECTORY)
#print(f'The next version name is: {data_version}')
#os.makedirs(f"{DATA_DIRECTORY}/{data_version}",exist_ok=True)
#set_data_version(data_version)

data_version = "data_v14"

print("Start")
    
selected_experiment = "Caption-No-More-Than-100-Tokens"
selected_category = 'No-Subfig'
n_train_images = 100
n_test_images = 50

# print(f"Selected Dataset: {selected_data}")
# print(f"Selected Experiment: {selected_experiment}")
# print(f"Selected Category: {selected_category}")
# print(f"Number of Training Images: {n_train_images}")
# print(f"Number of Testing Images: {n_test_images}")

meta_data = {
    "Selected Experiment": selected_experiment,
    "Selected Category": selected_category,
    "Number of Training Images": n_train_images,
    "Number of Testing Images": n_test_images
}
#write_json(meta_data,get_save_file_path("meta_data.json"))

SCICAP_META_DATA = f'{DATA_DIRECTORY}/captions_meta_data_19_may_24.xlsx'
#scicap_meta_data = pd.read_excel(SCICAP_META_DATA)
#print(f"Total metadata records:{scicap_meta_data.shape[0]}")

app = FastAPI(
    title="Image captioning for scientfic images"
)

api_router = APIRouter()
app.include_router(api_router)

@app.get("/")
def index(request: Request):
    print("in index")
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the API</h1>"
        "<div>"
        "Check the docs: <a href='/docs'>here</a>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)

@app.get("/trainold", status_code=200)
def train_old():
   trainWOFineTuning()


@app.get("/train", status_code=200)
def train():
   train()
   

@app.post("/infer", status_code=200)
async def infer(file:UploadFile = File(...)):
   image = Image.open(io.BytesIO(await file.read())).convert("RGB")
   infer(image)
   
# Set all CORS enabled origins
#if settings.BACKEND_CORS_ORIGINS:
#   app.add_middleware(
##        CORSMiddleware,
  #      allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
  #      allow_credentials=True,
  #      allow_methods=["*"],
  #      allow_headers=["*"],
  #  )

def train():
    print("Start")
    train_ds = load_train(1000)
    train_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/train")
    
    val_ds = load_val(500)
    val_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/val")
    
    test_ds = load_test(50)
    test_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/test")
    
    
    dotrain(train_ds, test_ds)
    

def trainWOFineTuning():
    print("Start")
    train_ds = load_train(1000)
    train_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/train")
    
    val_ds = load_val(500)
    val_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/val")
    
    test_ds = load_test(50)
    test_ds.save_to_disk(f"{DATA_DIRECTORY}/{data_version}/dataset/test")
    
    
    dotrainWOFineTuning(train_ds, test_ds)
    
def infer(image: Image):
    print("Start inference")
    #url = '../scicap_data/SciCap-No-Subfig-Img/test/1001.0317v2-Figure7-1.png'
    #absolute_path = os.path.abspath(url)
    #print(absolute_path)
    #image1 = Image.open(absolute_path)
    #url = '/home/ec2-user/environments/genai-cv-image-caption-scicap-data/scicap_data/SciCap-No-Subfig-Img/test/1001.0317v2-Figure7-1.png'
    #image1 = Image.open(url)
    generateCaptionPretrained(image)
    generateCaption(image)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004) 

    ## local host--> 127.0.0.0  
    ## host --> 0.0.0.0 allows all host

    