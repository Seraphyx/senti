#!/usr/local/bin/python3
import numpy as np
import tensorflow as tf
from functools import wraps
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"
