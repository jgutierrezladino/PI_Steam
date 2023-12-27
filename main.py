from funciones import *
from fastapi import FastAPI
import pandas as pd
app = FastAPI()


@app.get('/')
def bienvenida():
    return {'API de consultas a una base de datos de Steam, /docs en el link para acceder a las funciones de consulta.'}