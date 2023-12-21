## importamos las librerias

import pandas as pd
import gzip
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

## cargamos los archivos.

# Leer el DataFrame df_games desde un archivo CSV
df_games = pd.read_csv('../DataSets/df_games_id.csv')
# Leer el DataFrame df_user desde un archivo CSV dentro de un archivo ZIP
with zipfile.ZipFile('../DataSets/df_user_id.zip', 'r') as zipf:
    with zipf.open('df_user_id.csv') as csv_file:
        df_user = pd.read_csv(csv_file)
# Leer el DataFrame df_reviews desde un archivo CSV
df_reviews = pd.read_csv('../DataSets/df_user_reviews.csv')

### Preprocesamiento de datos:

# Vamos a procesar la columna 'genres' para que sea más fácil de trabajar y eliminar duplicados.

# Crear un DataFrame con la relación entre juegos y géneros
df_game_genres = pd.DataFrame({'id': df_games['id'], 'genres': df_games['genres'], 'app_name': df_games['app_name']})
# Eliminar duplicados
df_game_genres = df_game_genres.drop_duplicates(subset=['id', 'genres'])

### Matriz de géneros:
# Construir una matriz que represente la relación entre los juegos y los géneros.

# Eliminar duplicados en df_game_genres
df_game_genres = df_game_genres.drop_duplicates(subset=['app_name', 'genres'])
# Crear la matriz de géneros
genres_matrix = df_game_genres.pivot(index='app_name', columns='genres', values='id').fillna(0)

### Similitud del coseno entre juegos:
#Calcular la similitud del coseno entre juegos basada en la matriz de géneros.

cosine_sim = cosine_similarity(genres_matrix, genres_matrix)

### Función para obtener recomendaciones de juegos:

# Función para obtener recomendaciones de juegos similares
def get_recommendations_game(game_id, cosine_sim=cosine_sim):
    # Obtener la fila correspondiente al juego
    game_row = genres_matrix.loc[game_id].values.reshape(1, -1)
    # Calcular la similitud del coseno entre el juego y todos los demás juegos
    sim_scores = cosine_similarity(game_row, genres_matrix.values)
    # Obtener los juegos más similares (excluyendo el juego de entrada)
    similar_games = sim_scores.argsort()[0][::-1][1:]
    # Tomar los primeros 5 juegos recomendados
    recommendations = genres_matrix.index[similar_games][:5].tolist()
    return recommendations