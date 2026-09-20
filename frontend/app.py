import streamlit as st
import pandas as pd
import requests
import os

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:5000"
)
