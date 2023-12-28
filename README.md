<p align="center"><img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"></p>

<h1 align='center'> Proyecto Individual N°1</h1>

<h2 align='center'> Machine Learning Operations (MLOps)</h2>

<h2 align='center'>Bruno Zenobio, DATAFT16</h2>

---


- ## **`Links`**
    - [Carpeta con los dataset](./DataSets/)
    - [Proceso de ETL y EDA](./ETL%20y%20EDA/)
    - [API desplegada en Render](https://proyecto-individual-steam.onrender.com/docs)
    - [Link al video]()



---

# Introducción

En este proyecto llevaremos a cabo todo lo aprendido en nuestra cursada de Henry para el proyecto de Data Science, donde enfrentaremos diferentes desafios tanto personales como profesionales.

1. **Exploración y Transformación:** Se realizan operaciones necesarias para poder leer los archivos los cuales venian en un formato JSON comprimido, junto con una seria de columnas respectivamente anidadas, por lo que se hace uso del diccionario de datasets para identificar que relaciones tienen los archivos y como poder operarlos de la mejor manera.

2. **Preparación de Datos:** Se prepararán los datos para comprender las relaciones entre las variables y construir modelos sobre ellos. También se crearán funciones para consultas a los datos, consumibles a través de una API.

3. **Modelado:** Se desarrollarán modelos de Machine Learning para entender relaciones y predecir correlaciones entre variables (Item-Item).


## Diccionario de los Datos

<p align="center"><img src="./Imagenes/Diccionario de Datos STEAM.jpeg"></p>

---

# Desarrollo

### Exploración, Transformación y Carga (ETL)

A partir de los 3 dataset proporcionados (steam_games, user_reviews y user_items) referentes a la plataforma de Steam, en primera instancia se realizó el proceso de extraccion de los datos necesarios los cuales se resaltan en la anterior imagen.

#### `steam_games`

- Se cargo el archivo que venia en formato '.gz', se descomprimio y se cargo a un DataFrame para poder manejar el contenido.
- Se extrajo las columnas necesarias para nuestro proyecto.
- Se eliminaron las columnas totalmente nulas.
- Se extrajeron años de la columna release_date.
- Se desanido la columna 'genres' ya que traia mas de un genero para un mismo juego.
- Se exportó para tener el dataset limpio en un formato CSV para despues facilitar su lectura.

#### `user_reviews`

- Se cargo el archivo que venia en formato '.gz', se descomprimio y se cargo en un DataFrame para poder manejar los datos.
- Se eliminaron columnas innesesarias, para aligerar el proceso.
- Se desanido la columna 'reviews' que contenia diccionarios.
- Se desanidaron los diccionarios que tenia en la columna reviews anteriormente creados y se creo nuevas columnas para cada dato del diccionario.
- Se tomaron unicamente las columnas necesarias para nuestro proyecto.
- Se hizo un analisis de sentimiento para la columna 'review' que contenia comentarios de usuarios para darle un mejor manejo.
- Se exportó para tener el dataset limpio en un formato CSV para despues facilitar su lectura.

#### `user_items`

- Se realizó un explode ya que la columna de items era una lista de diccionarios.
- Se eliminaron filas con valores nulos en la columna de "items".
- Se exportó para tener el dataset limpio.


### Despliegue para la API

Se desarrollaron las siguientes funciones, a las cuales se podrá acceder desde la API en la página Render:

- **`play_time_genre(genre: str)`**: Retorna para un genero el año con mas horas jugadas.
- **` user_for_genre(genre: str)`**:Retorna el usuario con mas horas acumuladas para el genero dado.
- **`users_recommend(year: int)`**: Retorna para un año dado, el top 3 de juegos más recomendados.
- **`users_not_recommend(year: int)`**: Retorna para un año dado, el top 3 de juegos menos recomendados.
- **`sentiment_analysis_route(year: int)`**: Retorna para un año dado la cantidad de comantarios Negativos, Positivos y Neutros,
- **`recommend_games_route(game_name: str)`**: Dado el nombre de un juego, esta funsion retorna 5 juegos recomendados.
