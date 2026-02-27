from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ModelConfigCreate(BaseModel):
    name: str
    display_name: str
    path: str
    vector_dim: int = 1280
    description: Optional[str] = None
    is_default: bool = False


class ModelConfigUpdate(BaseModel):
    display_name: Optional[str] = None
    vector_dim: Optional[int] = None
    description: Optional[str] = None
    is_default: Optional[bool] = None


class ModelConfigOut(BaseModel):
    id: str
    name: str
    display_name: str
    path: str
    vector_dim: int
    description: Optional[str] = None
    is_default: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AvailableModelFile(BaseModel):
    filename: str
    path: str
    size_bytes: int
    registered: bool
