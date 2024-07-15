from data_manager import load_dataset, load_train, load_val
from gitbase import load_model_pretrained, transforms, compute_metrics, defineTrainingArgs, dotrain, generateCaption, generateCaptionPretrained
from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from api import api_router


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
    dataset_and_tokens_train = load_train()
    dataset_and_tokens_val = load_val()
    print(type(dataset_and_tokens_train[0]))
    
    print(type(dataset_and_tokens_val))
    dotrain(dataset_and_tokens_train, dataset_and_tokens_val)
    
def infer():
    print("Start inference")
    url = '/content/drive/MyDrive/Colab Notebooks/SciCap-No-Subfig-Img/test/2011.07019v1-Figure3-1.png'
    image1 = Image.open(url)
    generateCaptionPretrained(image1)
    generateCaption(image1)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.0", port=8004) 

    ## local host--> 127.0.0.0  
    ## host --> 0.0.0.0 allows all host

    