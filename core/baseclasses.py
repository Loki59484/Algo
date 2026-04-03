"""
Module containing Abstract Base Classes for structuring different features.
"""

# IMPORTING MODULES
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import sys

# Setting up paths to manage imports
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# IMPORTING CUSTOM MODULES
from core.datatypes import *

# SET-UP LOGGING
logger = logging.getLogger(__name__)

class Executor(ABC):
    
    @abstractmethod
    def buy_order(price: float, qty : int):
        

        return Order()