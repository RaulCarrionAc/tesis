import streamlit as st
st.set_page_config(page_title="Proyecto de titulación", layout = "centered")

#imports
import base64
from subQuestionQueryEngine import ejecutar_consulta_completa #motor armado en el script
from PIL import Image
import pandas as pd
#import pydeck as pdk
from shapely import wkt
import io
#from google.oauth2.credentials import Credentials
#from google_auth_oauthlib.flow import InstalledAppFlow
#from googleapiclient.discovery import build
#from googleapiclient.http import MediaIoBaseUpload
from datetime import datetime
import torch
from functools import partial

def init_torch():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).to('cuda')
            torch.backends.cudnn.benchmark = True
        except Exception as e:
            st.warning(f"Advertencia CUDA: {e}")
            torch.set_default_tensor_type('torch.FloatTensor')
    torch.set_num_threads(1)

init_torch()

# Cache optimizado (versión segura)
@st.cache_resource
def load_resources():
    return {
        'query_func': partial(ejecutar_consulta_completa)
    }

# Inicialización
if 'app_ready' not in st.session_state:
    with st.spinner('Inicializando aplicación...'):
        st.session_state.resources = load_resources()
        st.session_state.app_ready = True

#SCOPES = ['https://www.googleapis.com/auth/drive.file']
image_path = r"../Imagenes"

# def autenticar():
#     import os
#     from google.oauth2.credentials import Credentials

#     os.makedirs("auth", exist_ok=True)
#     creds = None

#     if os.path.exists("auth/token.json"):
#         creds = Credentials.from_authorized_user_file("auth/token.json", SCOPES)
#     else:
#         st.error("No se puede usar la app porque no está configurada la autenticación de Google Drive.\n\n" +
#                  "Debes generar previamente el archivo `auth/token.json` haciendo login local una vez.")
#         st.stop()

#     return creds

# def subir_csv_a_drive(df: pd.DataFrame, nombre_archivo: str, folder_id: str = None):
#     creds = autenticar()
#     service = build('drive', 'v3', credentials=creds)

#     # Crear archivo CSV en memoria
#     buffer = io.BytesIO()
#     df.to_csv(buffer, index=False)
#     buffer.seek(0)

#     media = MediaIoBaseUpload(buffer, mimetype='text/csv', resumable=True)

#     archivo = {
#         'name': nombre_archivo,
#         'mimeType': 'text/csv'
#     }

#     if folder_id:
#         archivo['parents'] = [folder_id]

#     creado = service.files().create(body=archivo, media_body=media, fields='id').execute()
#     print(f"Archivo subido correctamente con ID: {creado.get('id')}")

# def ejecutar_consultas(pregunta: str)->str:
#     try:
#         response = subquestion_engine.query(pregunta)
#         return response.response if hasattr(response, "response") else str(response)
#     except Exception as e:
#         return f"Ocurrió un error durante la consulta: {str(e)}"

#inicializar estado
if "step" not in st.session_state:
    st.session_state.step = 4
    st.session_state.contador_preguntas = 0
    st.session_state.respuestas_chat = []

#paso 1: Introducción
if st.session_state.step == 1:
    st.header("Bienvenido")
    st.markdown("""
                Esta aplicación es parte del proyecto de tesis de **Raúl Carrión**, para optar al título de Ingeniero Civil en Geografía.
                
                El propósito de este sitio es evaluar el desempéño de un sistema de asistencia para la toma de decisiones en el ámbito inmobiliario, orientado especialmente a personas sin formación técnica.

                El sistema se basa en un modelo explicativo construido a partir de variables espaciales agrupadas en tres categorías:
                1. **Infraestructura**: Atributos físicos de los inmuebles, como superficie total, número de habitaciones, cantidad de baños y tipo de viviendas (unifamiliar, multifamiliar u otros).
                2. **Geodemográficos**: Factores poblacionales y socioeconómicos según la ubicación, como el ingreso medio o la tasa de población activa.
                3. **Localización**: Proximidad a puntos de interés o servicios relevantes, como la distancia a Central Park o Times Square.

                ---
                Para determinar si esta herramienta cumple con su objetivo de facilitar la toma de decisiones, se desarrolló un proceso que consiste en tres pasos:

                1. Se presentará una caracterización del área de estudio, en este caso New York a través de los coeficientes locales obtenidos de un modelo de regresión ponderada geográficamente.
                En base a esta caracterización se realizarán preguntas relacionadas al nivel de comprensión de estas variables para el usuario.
                2. Se procede con la presentación de la herramienta basada en modelos de lenguaje que cumplirá la finalidad de dar explicaciones amigables al usuario sobre los resultados del modelo de regresión ponderada geográficamente.
                3. Finalmente, se evaluará el desempeño de la herramienta mediante una experiencia de usuario.
                """)
    
    #imagen de la usach
    with open(image_path + "/geo.png", "rb") as file:
        img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <div style="text-align: center; magin-top: 30px;">
            <img src = "data:image/png;base64,{img_base64}" width = "300"/>
        </div>
        """,
        unsafe_allow_html=True
    )
    #boton para pasar de estado
    if st.button("Comenzar"):
        st.session_state.step = 2
        #st.experimental_rerun()

#paso 2: Caracterización
elif st.session_state.step == 2:
    st.header("Caracterización del área de estudio")
    st.markdown("En la siguiente figura se observa el área de estudio correspondiende a New York.")

    with open(image_path + "/AreaEstudio.png", "rb") as file:
        img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <div style="text-align: center; magin-top: 30px;">
            <img src = "data:image/png;base64,{img_base64}" width = "700" height = "600"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""De la imagen se puede apreciar que el área de estudio está compuesta por 5 distritos: Staten Island, Brooklyn, Manhattan, Queens, Bronx.
                Cada uno de estos distritos cuenta con características distintivas entre ellas. Esto se aprecia en las siguientes cartas temáticas construidas a partir de los coeficientes locales del modelo de regresión
                ponderado geográficamente mencionado anteriormente. En el que se logran apreciar las diferencias entre cada distrito.
                """)
    
    st.markdown("""
                A continuación, se presenta un mapa de distribución de los precios de viviendas. De este podemos ver que los precios oscilan entre los \$148.215 
                hasta los \$13.114.684 USD. Concentrandose la
                los precios más altos en el sector de Manhattan seguido por Brooklyn.
                """)
    
    with open(image_path + "/distribucionPrecios.png", "rb") as file:
        img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <div style="text-align: center; magin-top: 30px;">
            <img src = "data:image/png;base64,{img_base64}" width = "700" height = "600"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
                Como se aprecia en la imagen existe una clara diferencia en los precios dependiendo del sector donde se ubica la vivienda. Esto puede tener diversos motivos pero a continuación,
                se presentarán diversas variables que a través del modelo de regresión ponderada geográficamente se ha determinado la influencia de estas sobre los precios. 
                """)
    
    st.markdown("""
                La siguiente imagen muestra como varían los precios en base a la distancia de las viviendas con respecto a Central Park y Times Square.
                """ )
    
    with open(image_path + "/CoeficientesLocales_Distancias.jpg", "rb") as file:
        img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
            .full-width-img {{
                position: relative;
                width: 100vw !important;
                left: 50%;
                transform: translateX(-50%);
                margin: 0;
                padding: 0;
            }}
        </style>
        
        <div class="full-width-img">
            <img src="data:image/png;base64,{img_base64}" style="width:30%%; height:auto;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
                Podemos observar que Central Park tiene principalmente un impacto positivo en los precios de los inmuebles en gran parte de Brooklyn, aun cuando este barrio 
                se encuentra a una distancia considerable del parque. En cambio, en la zona de Manhattan donde Central Park está ubicado su influencia sobre los precios es generalmente negativa. 
                Esto se explica, en gran medida, porque al tratarse de un área muy turística (Central Park recibe 37,5 millones de visitantes anuales), los niveles de contaminación acústica se elevan y 
                reducen la calidad de vida. Además, puede haber tasas de delincuencia más altas en sus alrededores.

                En cuanto a Times Square, dentro de Manhattan se observa un efecto parecido: los precios tienden a bajar en sus inmediaciones. 
                Sin embargo, en los distritos colindantes sucede lo contrario: Queens se beneficia de su relativa cercanía y conectividad con Times Square, 
                mientras que Brooklyn y Staten Island registran efectos negativos por su mayor lejanía y, probablemente, por una red de transporte menos 
                directa hacia este punto turístico.
                """)
    
    st.markdown("""
                Desde el punto de vista demográfico tenemos la tasa de población activa que corresponde a la proporción de personas en edad de trabajar con respecto a la población que no.
                Es decir, población entre 15 y 65 años con respecto a poblacion menos de 15 y mayores a 65 años. y también tenemos el ingreso medio.
                """)
    
    with open(image_path + "/CoeficientesLocales_Demograficos.jpg", "rb") as file:
        img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
            .full-width-img {{
                position: relative;
                width: 100vw !important;
                left: 50%;
                transform: translateX(-50%);
                margin: 0;
                padding: 0;
            }}
        </style>
        
        <div class="full-width-img">
            <img src="data:image/png;base64,{img_base64}" style="width:60%%; height:auto;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
                Pese a que se aprecian diferentes agrupaciones en los coeficientes de la tasa de población activa, esta es cercana a 0 en todos en todo New York, por lo tanto, podemos indicar
                que no tiene una variabilidad local en el espacio indicando así Homogeneidad espacial, por otro lado, el ingreso medio si presenta Heterogeneidad espacial teniendo su una mayor influencia en el Bronx
                y Brooklyn, seguidos por Staten Island.
                """)
    
    with open(image_path + "/CoeficientesLocales_Infraestructura.jpg", "rb") as file:
        img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
            .full-width-img {{
                width: 100vw !important;
                margin: 0;
                padding: 0;
            }}
            .full-width-img img {{
                display: block;
                width: 70%;
                height: auto;
                margin: 0 auto;
            }}
        </style>
        
        <div class="full-width-img">
            <img src="data:image/png;base64,{img_base64}"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
                Podemos ver que la cantidad de baños en Manhattan tienen un mayor impacto en el precio con respecto al resto de la ciudad, esto principalmente debido a que Manhattan al ser una zona de lujo
                cada baño se asocia a mayores niveles de comodidad. En Staten Island, es una zona en la que se priorizan otras cualidades de la vivienda como podría serlo la superficie total del inmueble, tal como
                podemos ver en la influencia de esta variable con respecto al precio. Por otro lado, la cantidad de dormitorios presenta mayor influencia en el Bronx y Brooklyn, esto puede ser debido a que este sea 
                un barrio residencial con familias numerosas
                """)
    
    st.markdown("""
                Finalmente, tenemos la influencia que presenta sobre el precio dependiendo de que tipo de vivienda se evalúa, Viviendas Unifamiliares son aquellas diseñadas para uso exclusivo de una familia,
                Viviendas multifamiliares como dúplex y casas adosadas y Otros son aquellos terrenos y propiedades atípicas o especiales como por ejemplo terrenos en venta.
                """)
    
    with open(image_path + "/coeficientesLocales_CATvivienda.jpg", "rb") as file:
        img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode()
    st.markdown(
        f"""
        <style>
            .full-width-img {{
                position: relative;
                width: 100vw !important;
                left: 50%;
                transform: translateX(-50%);
                margin: 0;
                padding: 0;
            }}
        </style>
        
        <div class="full-width-img">
            <img src="data:image/png;base64,{img_base64}" style="width:60%%; height:autp;"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
                En general, podemos ver que las viviendas unifamiliares tienen un efecto positivo en toda la ciudad a excepción de un barrio de Brooklyn, pero en general Esa zona de Brooklyn presenta
                un efecto negativo para cualquier tipo de vivienda a diferencia de las demás zonas que tenemos efectos positivos en general en función del tipo de vivienda en venta.
                """)
#############################################################################################################################
# Paso 3: Nivel de conocimiento
elif st.session_state.step == 3:
    #st.header("Nivel de conocimiento")
    #st.markdown("Por ahora esta sección es una encuesta vacía (puedes agregar preguntas aquí más adelante).")

    #### preguntas
    st.header("Preguntas previas al uso de la herramienta")
    st.markdown("""
        Antes de usar la herramienta, por favor responda estas preguntas. Esto permitirá entender el nivel de conocimiento actual y tus expectativas.
        """)
    # Placeholder
    nivel_tecnico = st.radio(
    "¿Tienes conocimientos previos en modelos espaciales como GWR?",
    ["Sí, con experiencia profesional", "Sí, conocimientos básicos", "No"]
    )

    opciones = [1, 2, 3, 4, 5]
    etiquetas = ["Nada confiado", "Poco Confiado", "Medianamente Confiado", "Confiado", "Muy confiado"]

    confianza_inicial = st.select_slider(
        "¿En base a la caracterización presentada en el punto anterior, qué tanto confías en tú capacidad para tomar una decisión de inversión inmobiliaria?",
        options=opciones,
        value=3,
        format_func=lambda x: etiquetas[x-1] if etiquetas[x-1] else "",  # Oculta números intermedios
        help="Desliza para seleccionar tu nivel de confianza"
    )
    #esto oculta los labels de la barra
    st.markdown("""
    <style>
        div[data-baseweb="slider"] div[role="slider"] ~ div {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # decision_previa = st.slider(
    # "¿Qué tan capaz te sientes actualmente para tomar decisiones de inversión con base en datos geográficos?",
    # min_value=1, max_value=5, value=3
    # )

    factores_importantes = st.text_input(
    "¿Qué factores consideras más importantes al evaluar una propiedad inmobiliaria?"
    )

    if st.button("Enviar respuestas"):
        respuesta_df = pd.DataFrame([{
            "nivel_tecnico": nivel_tecnico,
            "confianza_inicial": confianza_inicial,
            #"decision_previa": decision_previa,
            "factores_importantes": factores_importantes
        }])
        
        ########################################################### ESTO REVISARLO DESPUES ########################################################################
        #nombre_archivo = f"encuesta_inicial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        #subir_csv_a_drive(respuesta_df, nombre_archivo, folder_id = "1LpJpssF7QnAv9oVPbHs3aPSA7iXtrAkC")
        #st.success("¡Tus respuestas fueron enviadas y guardadas en Google Drive!")
        #st.session_state.step = 3
        #st.experimental_rerun()

#paso 4: Herramienta llm
############################################ HERRAMIENTA LLM ####################################
elif st.session_state.step == 4:
    st.header("Herramienta Basada en Modelos de Lenguaje")
    st.markdown("Las respuestas pueden tomar algunos minutos, por favor ten paciencia.")

    #mostrar conversacion previa
    for entrada, salida in st.session_state.respuestas_chat:
        with st.chat_message("user"):
            st.markdown(entrada)
        with st.chat_message("assistant"):
            st.markdown(salida)

    #contador del limite de 5 preguntas
    if st.session_state.contador_preguntas < 5:
        pregunta = st.chat_input("Escribe tu consulta...")
        if pregunta:
            with st.chat_message("user"):
                st.markdown(pregunta)
            
            with st.chat_message("assistant"):
                with st.spinner("Analizando... Espere por favor."):
                    query = st.session_state.resources["query_func"]
                    out = query(pregunta)
                if isinstance(out, dict) and "respuesta" in out:
                    contenido = out["respuesta"]
                else:
                    contenido = out

                st.markdown(contenido)
            
            st.session_state.respuestas_chat.append((pregunta, contenido))
            st.session_state.contador_preguntas += 1
            st.experimental_rerun()

    