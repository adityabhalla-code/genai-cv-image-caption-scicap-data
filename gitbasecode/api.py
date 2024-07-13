import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root, dirparent = file.parent, file.parents[1], file.parents[2]

import json
from typing import Any
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from gitbasecode.main_api import train

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.post("/train", status_code=200)
async def train() -> Any:
   train()
   

@api_router.post("/infer", status_code=200)
async def train() -> Any:
   infer()
   
   