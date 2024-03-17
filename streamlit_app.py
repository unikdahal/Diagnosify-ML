import streamlit as st
import subprocess

subprocess.Popen(["uvicorn", "app:app", "--reload"])
