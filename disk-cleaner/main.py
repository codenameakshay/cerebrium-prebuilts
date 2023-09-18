from typing import Optional
from pydantic import BaseModel
import os

class Item(BaseModel):
    dir: Optional[str] = None # directory to clear

def recursive_size(path):
    total_size = os.path.getsize(path)
    if os.path.isdir(path):
        for item in os.listdir(path):
            itempath = os.path.join(path, item)
            if os.path.isfile(itempath):
                total_size += os.path.getsize(itempath)
            elif os.path.isdir(itempath):
                total_size += recursive_size(itempath)
    return total_size


def predict(item, run_id, logger):
    item = Item(**item)
    results = []
    if item.dir is None:
        item.dir = "/persistent-storage/"
    for root, dirs, files in os.walk(item.dir):
        for file in files:
            path = os.path.join(root, file)
            try:
                os.remove(path)
                results.append(path)
            except Exception as e:
                results.append(f"Failed to delete {path} because {e}")

    # get the size of the persistent storage
    pvc_size = recursive_size("/persistent-storage/") / 1e9 # in GB
    return {"files-deleted": results, "pvc-size-GB": pvc_size} # return your results 