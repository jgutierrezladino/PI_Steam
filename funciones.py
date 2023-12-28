## Importamos las librerias necesarias para operar.

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
import logging


## cargamos los archivos

# Leer el DataFrame df_games desde un archivo CSV
df_games = pd.read_csv('./DataSets/df_games_id.csv')
# Leer el DataFrame df_user desde un archivo CSV 
df_user = pd.read_csv('./DataSets/df_user_id.csv')
# Leer el DataFrame df_reviews desde un archivo CSV
df_reviews = pd.read_csv('./DataSets/df_user_reviews.csv')

## Año con más horas jugadas por genero ##

def PlayTimeGenre(genre):
    # Filtrar juegos por género
    filtered_games = df_games[df_games['genres'].str.contains(genre, case=False, na=False)]
    # Unir DataFrames
    merged_df = pd.merge(df_user, filtered_games, left_on='item_id', right_on='id')
    # Agregar horas de juego por año
    grouped_df = merged_df.groupby('release_date')['playtime_forever'].sum().reset_index()
    # Identificar el año con más horas jugadas
    max_year = grouped_df.loc[grouped_df['playtime_forever'].idxmax()]['release_date']
    # Extraer solo el año
    year_only = max_year.split('-')[0]
    result_string = 'El año con más horas jugadas para el género ' + genre + ' es: ' + str(year_only)
    return result_string

## Usuario con mas horas jugadas por genero. ##

def UserForGenre(genre):
    try:
        # Filtrar juegos por género
        filtered_games = df_games[df_games['genres'].str.contains(genre, case=False, na=False)]
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
        user_hours_by_year = grouped_df[grouped_df['user_id'] == max_user]
        # Crear una lista de diccionarios para cada año y las horas jugadas
        result = [{"Año": int(year), "Horas": int(hours)} for year, hours in user_hours_by_year[['release_year', 'playtime_forever']].values]
        return {"Usuario con más horas jugadas para Género " + genre: max_user, "Horas jugadas por año": result}
    except Exception as e:
        return {"error": f"Error en la función UserForGenre: {str(e)}"}

## Top tres más recomendadas. ##

def UsersRecommend(year):
    # Convertir la columna 'release_date' a tipo datetime
    df_games['release_date'] = pd.to_datetime(df_games['release_date'], errors='coerce')
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

def UsersNotRecommend(year):
    # Convertir la columna 'release_date' a tipo datetime
    df_games['release_date'] = pd.to_datetime(df_games['release_date'], errors='coerce')
    # Filtrar las reviews para el año dado con recomendación negativa y comentarios negativos
    filtered_reviews = df_reviews[
        (df_reviews['recommend'] == False) &
        (df_reviews['sentiment_analiysis'] == 0) &  # 0 para comentarios negativos
        (pd.to_datetime(df_games['release_date']).dt.year == year)]
    # Unir con el DataFrame de juegos para obtener los nombres de los juegos
    merged_reviews = pd.merge(filtered_reviews, df_games[['id', 'app_name']], left_on='item_id', right_on='id')
    # Contar la cantidad de veces que aparece cada juego
    top_games = merged_reviews['app_name'].value_counts().head(3)
    # Convertir el resultado a un formato de lista de diccionarios
    result = [{"Puesto numero {} menos recomendado es".format(i+1): juego} for i, juego in enumerate(top_games.index)]
    return result

## Analisis de sentimiento por año.

def sentiment_analysis(year):
    try:
        # Unir las reseñas con la información de los juegos
        merged_reviews = pd.merge(df_reviews, df_games[['id', 'release_date']], left_on='item_id', right_on='id')
        # Filtrar las reseñas para el año dado
        filtered_reviews = merged_reviews[
            (pd.to_datetime(merged_reviews['release_date'], errors='coerce').dt.year == year)]
        # Contar la cantidad de registros por categoría de sentimiento
        sentiment_counts = filtered_reviews['sentiment_analiysis'].value_counts()
        # Crear el diccionario de resultados
        result = {
            'Negative': int(sentiment_counts.get(0, 0)),
            'Neutral': int(sentiment_counts.get(1, 0)),
            'Positive': int(sentiment_counts.get(2, 0))}
        return result
    except Exception as e:
        return {"error": f"Error en la función sentiment_analysis: {str(e)}"}

### Preprocesamiento de datos para Machine-Learning
# Manejar datos faltantes y duplicados
df_games = df_games.dropna()  # Puedes ajustar esto según tus necesidades
df_games = df_games.drop_duplicates(subset=['id', 'app_name'])

# Codificación one-hot para géneros
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(df_games['genres'].str.split(',')), columns=mlb.classes_)

# Concatenar características al conjunto de datos original
df_games_encoded = pd.concat([df_games, genres_encoded], axis=1)

# Calcular la similitud del coseno
cosine_sim = cosine_similarity(genres_encoded, genres_encoded)

# División de datos
X_train, X_test = train_test_split(df_games_encoded, test_size=0.2, random_state=42)

# Manejar valores faltantes en X_train
X_train_features = X_train.drop(['id', 'app_name', 'genres', 'release_date'], axis=1)  # Eliminar columnas no necesarias

# Opción 1: Eliminar filas con NaN
# X_train_features = X_train_features.dropna()

# Opción 2: Imputar valores faltantes
imputer = SimpleImputer(strategy='mean')
X_train_features = imputer.fit_transform(X_train_features)

# Entrenamiento del modelo KNN
knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
knn_model.fit(X_train_features)

# Función para obtener recomendaciones usando el modelo KNN
def get_recommendations_knn(game_id, model=knn_model, df=df_games_encoded):
    game_row = df.drop(['id', 'app_name', 'genres', 'release_date'], axis=1).loc[game_id].values.reshape(1, -1)
    _, indices = model.kneighbors(game_row)
    recommendations = df['app_name'].iloc[indices[0][1:6]].tolist()
    return recommendations

# Función de recomendación
def recommend_games(game_name):
    try:
        # Verificar si hay alguna fila que coincide con el nombre del juego
        if df_games[df_games['app_name'] == game_name].empty:
            return {"error": f"No se encontró ningún juego con el nombre '{game_name}'"}
        # Obtener el índice del juego
        idx = df_games[df_games['app_name'] == game_name].index[0]
        # Obtener las recomendaciones usando el modelo KNN
        recommended_games = get_recommendations_knn(idx)
        return {'recomendaciones': list(recommended_games)}
    except Exception as e:
        return {"error": f"Error en la función recommend_games: {str(e)}"}
