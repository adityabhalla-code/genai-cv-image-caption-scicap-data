import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root, dirparent = file.parent, file.parents[1], file.parents[2]

import json
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

api_router = APIRouter()


@api_router.get("/train", status_code=200)
def train() -> Any:
   train()
   

@api_router.get("/infer", status_code=200)
def infer() -> Any:
   infer()
   
   