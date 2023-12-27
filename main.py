from funciones import *
from fastapi import FastAPI
import pandas as pd
app = FastAPI()
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)


# Ruta de bienvenida
@app.get('/')
def bienvenida():
    return {'API de consultas a una base de datos de Steam, /docs en el link para acceder a las funciones de consulta.'}

# Ruta para la función PlayTimeGenre
@app.get('/play_time_genre/{genre}')
def play_time_genre(genre: str):
    result = PlayTimeGenre(genre)
    return {'result': result}

# Ruta para la función UserForGenre
@app.get('/user_for_genre/{genre}')
def user_for_genre(genre: str):
    result = UserForGenre(genre)  
    return {'result': result}

# Ruta para la función UsersRecommend
@app.get('/users_recommend/{year}')
def users_recommend(year: int):
    result = UsersRecommend(year)
    return {'result': result}

# Ruta para la función UsersNotRecommend
@app.get('/users_not_recommend/{year}')
def users_not_recommend(year: int):
    result = UsersNotRecommend(year)
    return {'result': result}

# Ruta para la función sentiment_analysis
@app.get('/sentiment_analysis/{year}')
def sentiment_analysis_route(year: int):
    try:
        result = sentiment_analysis(year)  # Solo proporciona un argumento, que es 'year'
        return {'result': result}
    except Exception as e:
        return {'error': f"Error al procesar la solicitud: {str(e)}"}

