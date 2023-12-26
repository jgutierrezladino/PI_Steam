## Importamos las librerias necesarias para operar.

import pandas as pd
import gzip
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


## cargamos los archivos

# Leer el DataFrame df_games desde un archivo CSV
df_games = pd.read_csv('./DataSets/df_games_id.csv')
# Leer el DataFrame df_user desde un archivo CSV dentro de un archivo ZIP
with zipfile.ZipFile('./DataSets/df_user_id.zip', 'r') as zipf:
    with zipf.open('df_user_id.csv') as csv_file:
        df_user = pd.read_csv(csv_file)
# Leer el DataFrame df_reviews desde un archivo CSV
df_reviews = pd.read_csv('./DataSets/df_user_reviews.csv')

## Año con más horas jugadas por genero ##

import pandas as pd

def PlayTimeGenre(genre, percentage=0.1):
    # Tomar un 10% de los datos de df_games
    subset_df_games = df_games.sample(frac=percentage, random_state=42)
    # Filtrar juegos por género
    filtered_games = subset_df_games[subset_df_games['genres'].str.contains(genre, case=False, na=False)]
    # Unir DataFrames
    merged_df = pd.merge(df_user, filtered_games, left_on='item_id', right_on='id')
    # Agregar horas de juego por año
    grouped_df = merged_df.groupby('release_date')['playtime_forever'].sum().reset_index()
    # Identificar el año con más horas jugadas
    max_year = grouped_df.loc[grouped_df['playtime_forever'].idxmax()]['release_date']
    # Extraer solo el año
    year_only = max_year.split('-')[0]
    result_string = 'El año con más horas jugadas para el género: ' + genre + ' es: ' + str(year_only)
    return result_string

## Usuario con mas horas jugadas por genero. ##

def UserForGenre(genre, percentage=0.1):
    # Tomar un 10% de los datos de df_games
    subset_df_games = df_games.sample(frac=percentage, random_state=42)  # 
    # Filtrar juegos por género
    filtered_games = subset_df_games[subset_df_games['genres'].str.contains(genre, case=False, na=False)]
    # Unir DataFrames
    merged_df = pd.merge(df_user, filtered_games, left_on='item_id', right_on='id')
    # Convertir la columna 'release_date' a formato de fecha y extraer el año
    merged_df['release_date'] = pd.to_datetime(merged_df['release_date'], errors='coerce')
    merged_df = merged_df.dropna(subset=['release_date'])  # Eliminar filas con fechas nulas
    merged_df['release_year'] = merged_df['release_date'].dt.year
    # Agregar horas de juego por año y usuario
    grouped_df = merged_df.groupby(['user_id', 'release_year'])['playtime_forever'].sum().reset_index()
    # Identificar el usuario con más horas jugadas
    max_user = grouped_df.loc[grouped_df['playtime_forever'].idxmax()]['user_id']
    # Obtener la lista de acumulación de horas jugadas por año para ese usuario
    user_hours_by_year = grouped_df[grouped_df['user_id'] == max_user][['release_year', 'playtime_forever']]
    user_hours_list = [{"Año": int(year), "Horas": hours} for year, hours in user_hours_by_year.values]
    return {"Usuario con más horas jugadas para Género " + genre: max_user, "Horas jugadas": user_hours_list}

## Top tres más recomendadas. ##

def UsersRecommend(year, percentage=0.1):
    # Tomar un 10% de los datos de df_games
    subset_df_games = df_games.sample(frac=percentage, random_state=42)
    # Convertir la columna 'release_date' a tipo datatime
    subset_df_games['release_date'] = pd.to_datetime(subset_df_games['release_date'], errors='coerce')
    # Filtrar las reviews para el año dado con recomendación negativa y comentarios negativos
    filtered_reviews = df_reviews[
        (df_reviews['recommend'] == True) &
        (df_reviews['sentiment_analiysis'] != 0) &  
        (pd.to_datetime(df_games['release_date']).dt.year == year)]
    # Unir con el DataFrame de juegos para obtener los nombres de los juegos
    merged_reviews = pd.merge(filtered_reviews, df_games[['id', 'app_name']], left_on='item_id', right_on='id')
    # Contar la cantidad de veces que aparece cada juego
    top_games = merged_reviews['app_name'].value_counts().head(3)
    # Convertir el resultado a un formato de lista de diccionarios
    result = [{"Puesto numero {} más recomendado es".format(i+1): juego} for i, juego in enumerate(top_games.index)]
    return result

## Top tres menos recomendadas. ##

def UsersNotRecommend(year, percentage=0.1):
    # Tomar un 10% de los datos de df_games
    subset_df_games = df_games.sample(frac=percentage, random_state=42)  # Puedes ajustar el valor de random_state según tus necesidades
    # Convertir la columna 'release_date' a tipo datetime
    subset_df_games['release_date'] = pd.to_datetime(subset_df_games['release_date'], errors='coerce')
    # Filtrar las reviews para el año dado con recomendación negativa y comentarios negativos
    filtered_reviews = df_reviews[
        (df_reviews['recommend'] == False) &
        (df_reviews['sentiment_analiysis'] == 0) &  # 0 para comentarios negativos
        (pd.to_datetime(subset_df_games['release_date']).dt.year == year)]
    # Unir con el DataFrame de juegos para obtener los nombres de los juegos
    merged_reviews = pd.merge(filtered_reviews, subset_df_games[['id', 'app_name']], left_on='item_id', right_on='id')
    # Contar la cantidad de veces que aparece cada juego
    top_games = merged_reviews['app_name'].value_counts().head(3)
    # Convertir el resultado a un formato de lista de diccionarios
    result = [{"Puesto numero {} menos recomendado es".format(i+1): juego} for i, juego in enumerate(top_games.index)]
    return result

## Analisis de sentimiento por año.

def sentiment_analysis(year, percentage=0.1):
    # Tomar un 10% de los datos de df_games
    subset_df_games = df_games.sample(frac=percentage, random_state=42)  # Puedes ajustar el valor de random_state según tus necesidades
    # Unir las reseñas con la información de los juegos
    merged_reviews = pd.merge(df_reviews, subset_df_games[['id', 'release_date']], left_on='item_id', right_on='id')
    # Filtrar las reseñas para el año dado
    filtered_reviews = merged_reviews[
        (pd.to_datetime(merged_reviews['release_date'], errors='coerce').dt.year == year)]
    # Contar la cantidad de registros por categoría de sentimiento
    sentiment_counts = filtered_reviews['sentiment_analiysis'].value_counts()
    # Crear el diccionario de resultados
    result = {
        'Negative': sentiment_counts.get(0, 0),
        'Neutral': sentiment_counts.get(1, 0),
        'Positive': sentiment_counts.get(2, 0)}
    return result

### Preprocesamiento de datos: para Machine-Learning

# Tomar un 10% de los datos de df_games
subset_df_games = df_games.sample(frac=0.1, random_state=42)

# Vamos a procesar la columna 'genres' para que sea más fácil de trabajar y eliminar duplicados.

# Crear un DataFrame con la relación entre juegos y géneros
df_game_genres = pd.DataFrame({'id': df_games['id'], 'genres': df_games['genres'], 'app_name': df_games['app_name']})
# Eliminar duplicados
df_game_genres = df_game_genres.drop_duplicates(subset=['id', 'genres'])
# Eliminar duplicados en df_game_genres
df_game_genres = df_game_genres.drop_duplicates(subset=['app_name', 'genres'])
# Crear la matriz de géneros
genres_matrix = df_game_genres.pivot(index='app_name', columns='genres', values='id').fillna(0)

### Similitud del coseno entre juegos:
#Calcular la similitud del coseno entre juegos basada en la matriz de géneros.
cosine_sim = cosine_similarity(genres_matrix, genres_matrix)

### Función para obtener recomendaciones de juegos:

# División de datos
X_train, X_test = train_test_split(genres_matrix, test_size=0.2, random_state=42)
# Entrenamiento del modelo
knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
knn_model.fit(X_train)
# Función para obtener recomendaciones usando el modelo entrenado
def get_recommendations_knn(game_id, model=knn_model):
    game_row = genres_matrix.loc[game_id].values.reshape(1, -1)
    _, indices = model.kneighbors(game_row)
    recommendations = genres_matrix.index[indices[0][1:6]].tolist()
    return recommendations