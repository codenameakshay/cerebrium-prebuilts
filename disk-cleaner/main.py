from typing import Optional
from pydantic import BaseModel
import os

class Item(BaseModel):
    dir: Optional[str] = None # directory to clear