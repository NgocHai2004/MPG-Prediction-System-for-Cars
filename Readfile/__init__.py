import pandas as pd
from pathlib import Path

class Read_file:
    def __init__(self,path):
        self.path = path
    def read(self):
        return pd.read_csv(self.path)

