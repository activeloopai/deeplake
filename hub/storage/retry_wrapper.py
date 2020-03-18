from typing import *
import os, sys, io, traceback, time, random, json
from retrying import retry

from .base import Base


class RetryWrapper(Base):
    def __init__(self, internal: Base):
        self._internal = internal
    
    @retry(stop_max_attempt_number=3)
    def get(self, path): 
        return self._internal.get(path)
    
    @retry(stop_max_attempt_number=3)
    def put(self, path, content):
        self._internal.put(path, content)

    @retry(stop_max_attempt_number=3)
    def exists(self, path):
        return self._internal.exists(path) 
    
    @retry(stop_max_attempt_number=3)
    def delete(self, path):
        self._internal.delete(path)

    @retry(stop_max_attempt_number=3)
    def get_or_none(self, path):
        return self._internal.get_or_none(path)