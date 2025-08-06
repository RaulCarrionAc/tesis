# BLOQUE DE IMPORTACIONES
import asyncio
import nest_asyncio
nest_asyncio.apply()

import chromadb
from llama_index.core.storage import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.question_gen.prompts import DEFAULT_SUB_QUESTION_PROMPT_TMPL
from llama_index.core.question_gen.output_parser import SubQuestionOutputParser
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.response_synthesizers import get_response_synthesizer

import pandas as pd
from sqlalchemy import (
    create_engine,
    select,
    MetaData,
    Table
)

from IPython.display import Markdown

from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)

from llama_index.core.schema import TextNode
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core import VectorStoreIndex, SQLDatabase
from llama_index.core.prompts import (
    RichPromptTemplate,
    PromptTemplate,
    PromptType,
    )

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, QueryBundle, Response
from llama_index.core.callbacks import CallbackManager
from typing import Optional
from functools import lru_cache
import torch
###

### funciones y clases ###
class CustomSQLTool(BaseQueryEngine):
    def __init__(self, retriever, sql_database, contexto, callback_manager: Optional[CallbackManager] = None):
        super().__init__(callback_manager=callback_manager or CallbackManager())
        self.retriever = retriever
        self.sql_database = sql_database
        self.contexto = contexto
        self._last_result = None

    def _query(self, query_bundle: QueryBundle) -> Response:
        """Implementación síncrona requerida por BaseQueryEngine"""
        return asyncio.run(self._aquery(query_bundle))

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """Implementación asíncrona requerida por BaseQueryEngine"""
        return await self._run_async_query(query_bundle.query_str)

    def _get_prompt_modules(self) -> dict:
        """Método abstracto requerido - puede devolver un dict vacío"""
        return {}
    async def _run_async_query(self, question: str) -> Response:
        template_sql_str = """
Información de contexto abajo.
---------------------------------------
{{ contexto }}
---------------------------------------
Dada la información de contexto, genera una consulta SQL en base a la consulta.                                                                                
Consulta: {{ consulta }}
Recuerda que estas usando SQLite y la expresion correcta es !=, no <>.
"""
        prompt_template = RichPromptTemplate(template_sql_str)
        full_prompt = prompt_template.format(contexto=self.contexto, consulta=question)
        query_bundle = QueryBundle(full_prompt)

        nodes = await asyncio.to_thread(lambda: self.retriever.retrieve(query_bundle))
        if not nodes:
            raise ValueError("No se recuperaron nodos desde el retriever")
        
        data_rows = [n.node.metadata for n in nodes]
        df = pd.DataFrame(data_rows)
        self._last_result = {"dataframe":df,"markdown":df.to_markdown(index = False), "csv":df.to_csv(index=False)}

        result_node = TextNode(text=self._last_result["markdown"], metadata={"dataframe": df})
        return Response(response=self._last_result["markdown"], source_nodes=[result_node])
    
    def query(self, query_bundle: QueryBundle) -> Response:
        """Método público que usa la implementación síncrona"""
        return self._query(query_bundle)
    
    def get_last_result(self):
        if self._last_result is None:
            raise ValueError("No se ha ejecutado ninguna consulta aun")
        return self._last_result

def obtener_variables(**kwargs) -> str:
    from unidecode import unidecode

    VARIABLES = {
        "banos": "BATH",
        "dormitorios": "BEDS",
        "tamano": "PROPERTYSQFT",
        "ingresos": "mi_avg",
        "central park": "d_cp",
        "times square": "d_ts",
        "multifamiliar": "CAT_VM",
        "unifamiliar": "CAT_VU",
        "otros": "CAT_Otros"
    }

    pregunta = kwargs.get("query_str", "").lower()
    pregunta = unidecode(pregunta)  # quita tildes

    print("Pregunta normalizada:", pregunta)
    variables_detectadas = []


    for alias, variable in VARIABLES.items():
        if alias in pregunta:
            print(f">>> Match: '{alias}' -> {variable}")
            variables_detectadas.append(variable)
    if not variables_detectadas:
        print(">>> Ninguna coindicencia encontrada.")
        #return ["VARIABLE_NO_ENCONTRADA"]
        return None
    
    return ", ".join(variables_detectadas)

@lru_cache(maxsize = 1)#cachea la instancia llm
def get_llm_model():
    instrucciones = """
    Sigue estas reglas:
    * responde en español
    """
    return Ollama(
        model = "mistral:latest",
        temperature = 0.3,
        request_timeout= 300,
        system_prompt = instrucciones,
    )

@lru_cache(maxsize = 1) #cachea el modelo de embedding
def get_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbedding(model_name = "hkunlp/instructor-large",device = device)


@lru_cache(maxsize = 1)
def get_sql_database():
    df = pd.read_csv(r"../data/data tabulada/resultados_gwr.csv", sep = ";")
    engine = create_engine("sqlite:///local_gwr.sqlite")
    df.to_sql("resultados_gwr",con=engine, index=False, if_exists="replace")
    return SQLDatabase(engine, include_tables=["resultados_gwr"])

@lru_cache(maxsize = 1)
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_storage")
    return chroma_client.get_or_create_collection(name="hf_nodes_without_metadata")

#prompts
sql_prompt = PromptTemplate(
    template="""
Eres un experto en análisis espacial aplicado a bases de datos de regresión ponderada geográficamente (GWR).
Tu objetivo es caracterizar el área de estudio, identificando los valores más altos y más bajos de una variable específica en cada distrito (`BoroName`), usando una consulta SQL sobre la tabla `resultados_gwr`.
Sigue estas reglas:

- La tabla `resultados_gwr` contiene resultados de un modelo GWR aplicado a viviendas.
- Los campos `BROKERTITLE` y `BoroName` son los únicos campos no numéricos.
- Nunca compares variables numéricas con strings.
- No uses `GROUP BY`.
- Usa filtros lógicos como `!= 0` para evitar valores nulos o irrelevantes.
- Muestra como resultado solo las siguientes columnas: `"PRICE"`, `"BROKERTITLE"`, `"BoroName"`, y la variable solicitada.
- Siempre aplica el siguiente patrón para mostrar los **5 valores más altos y 5 más bajos** de la variable solicitada, por cada distrito.
- Devuelve la consulta SQL sin comillas ni caracteres de escape (\n, \t, etc.), como si la fueras a ejecutar directamente en SQLite.

---

Subpregunta: {query_str}

---

Consulta SQL:

WITH ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY "BoroName" ORDER BY "<nombre de variable>" DESC) AS rn_desc, ROW_NUMBER() OVER (PARTITION BY "BoroName" ORDER BY "<nombre de variable>" ASC) AS rn_asc FROM resultados_gwr WHERE "<nombre de variable>" != 0) SELECT "PRICE_2", "BoroName", "<nombre de variable>", "Latitud", "Longitud" FROM ranked WHERE rn_desc <= 5 OR rn_asc <= 5 ORDER BY "BoroName", rn_desc;

---

Tabla: resultados_gwr
Columnas:
- PRICE: Precio de la vivienda en dólares.
- BROKERTITLE: Nombre del corredor o agente inmobiliario.
- intercept: Precio base cuando las demás características valen cero.
- BATH: Influencia o coeficiente local del número de baños en el precio.
- PROPERTYSQFT: Influencia o coeficiente local del tamaño de la propiedad en el precio.
- mi_avg: Influencia o coeficiente local del ingreso promedio en el precio.
- tpa: Influencia o coeficiente local de la Tasa de población activa en el precio.
- d_cp: Influencia o coeficiente local de la Distancia de la vivienda a Central Park.
- d_ts: Influencia o coeficiente local de la Distancia de la vivienda Distancia a Times Square.
- CAT_VM: Influencia o coeficiente local de vivienda multifamiliar en el precio.
- CAT_VU: Influencia o coeficiente local de vivienda sea unifamiliar.
- CAT_Otros: Influencia o coeficiente local de otros tipos de vivienda.
- residuals: Error de predicción del modelo.
- geometry: Ubicación geográfica de la propiedad.
- BoroName: Nombre del Distrito al que pertenece la vivienda.
---

Consulta SQL:

"""
)

output_template_str = """
La variable analizada en este caso fue **{{ variable }}**, y se evaluó su influencia en el precio de las viviendas en distintas zonas de Nueva York, utilizando resultados obtenidos mediante un modelo de regresión ponderada geográficamente (GWR).

---

### Guía de estilo para la redacción:

Tu objetivo es explicar, con claridad y naturalizando el texto para una persona que no comprenda nada acerca del tema. Explica cómo {{ variable }} afecta los precios de las viviendas en Nueva York según los datos entregados.


Sigue estas reglas para redactar una respuesta clara, completa y sin tecnicismos:

1. **Segmenta el análisis por distrito**: Evalúa cómo varía el impacto de {{ variable }} en Bronx, Queens, Staten Island, Manhattan y Brooklyn.
2. **Compara entre distritos**: Indica qué zonas muestran un efecto más fuerte o débil, y ofrece hipótesis razonables.
3. **Traduce las variables técnicas**: Usa lenguaje natural. Por ejemplo:
4.   **Nunca muestres nombres técnicos como BATH, d_cp, etc.**
5. **Usa un tono conversacional**: Como si lo explicaras a alguien sin conocimientos técnicos.
6. **Evita copiar frases de ejemplo o repetir ideas**.
7. **Incorpora un ejemplo o analogía cotidiana**, si ayuda a explicar el fenómeno.
8. **Cierra con un resumen que aporte insight**, no una repetición.
9. **Redacta al menos tres párrafos completos.**


### PUNTOS IMPORTANTES A CONSIDERAR

** Cambia el uso de los conceptos listados a continuación por los señalados con -> :**
* PRICE -> Precio
* BROKERTITLE -> Corredor de propiedad
* intercept -> intercepto
* BATH -> baños
* PROPERTYSQFT -> tamaño de la propiedad
* mi_avg -> ingrso medio
* tpa -> tasa de población activa
* d_cp -> distancia a central park
* d_ts -> distancia a times square
* CAT_VM -> vivienda multifamiliar
* CAT_VU -> vivienda unifamiliar
* CAT_Otros -> otro tipo de vivienda
---


### Información útil de referencia:

{{ contexto }}

---

### Datos analizados y respuestas parciales:

{{ context_str }}
"""

#template de salida para el synthesizer
summary_prompt = RichPromptTemplate(
    template_str=output_template_str,
    template_var_mappings={
        "contexto": "contexto",
        "variable": "variable",
        "context_str": "context_str"
    }
)


def extract_sql(raw_sql: str)->str:
    """Toma la salida cruda del modelo, que puede venir con prefijos o texto explicativo, y devuelve solo la consulta SQL"""
    print("[extract_sql] raw_sql:")
    print(raw_sql)
    print("─────────────────────────────────────")
    lines = raw_sql.splitlines()
    start_idx = next( (i for i, l in enumerate(lines) if l.strip().upper().startswith(("WITH","SELECT","INSERT"))), None )
    print(f"[extract_sql] start_idx: {start_idx}")
    if start_idx is None:
        cleaned = raw_sql.strip()
        print("[extract_sql] No se detectó SQL. Devolviendo raw completo:")
        print(cleaned)
        return cleaned
    
    cleaned_lines = [line.strip() for line in lines[start_idx:]]
    cleaned = "\n".join(cleaned_lines).strip()
    print("[extract_sql] SQL extraído:")
    print(cleaned)
    print("─────────────────────────────────────")
    return cleaned

### CONFIGURACION DEL SISTEMA ###
Settings.llm = get_llm_model()
Settings.embed_model = get_embedding_model()

response_synthesizer = get_response_synthesizer(
    response_mode = "tree_summarize",
    summary_template = summary_prompt,
    use_async = True,
    llm = get_llm_model()
)

#template para el llm naturalizador
texto_entrada ="""
Tu regla principal es naturalizar el siguiente texto, naturalizar implica escribir de una forma fluida y comprensible, sin usar nombres relativos a alguna tabla sino que mencionarlo de forma informal
---

    {text}

---

Reescribe lo anterior de forma narrativa, sencilla y sin tecnicismos. 

Además la consulta del usuario muestra los siguientes datos referentes a los coeficientes locales basado en su consulta con los cuales debes realizar un análisis propio.
Con dichos datos realiza un análisis basado en los distritos y recuerda **no usar los nombres de las variables como tal, sino que naturalizar sus nombres**
datos:

{datos}

"""

template_entrada = PromptTemplate(template = texto_entrada)

systemPrompt_naturalizador = """
Actúa como un narrador. Tú tarea será naturalizar los textos de forma narrativa es decir, facilitar por medio de la narración la lectura de lo que se te entrega.
Debes suponer que el lector no comprende el tema que se está tratando por lo tanto, debes facilitar esta tarea.
Es muy importante que los nombres técnicos como BATH, d_cp, d_ts, PROPERTYSQFT no los escribas textualmente.
"""

@lru_cache(maxsize=1)
def get_tools():
    sql_database = get_sql_database()
    chroma_collection = get_chroma_collection()

    # Configuración del vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index_pdf = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=get_embedding_model(),
        storage_context=storage_context
    )

    # SQL Retriever
    sql_retriever = NLSQLRetriever(
        sql_database=sql_database,
        tables=["resultados_gwr"],
        llm=get_llm_model(),
        sql_prompt=sql_prompt,
        verbose=True,
        return_raw=True,
    )
    
    # SQL Tool - Asegúrate de usar QueryEngineTool correctamente
    sql_tool = QueryEngineTool(
        query_engine=CustomSQLTool(sql_retriever, sql_database, contexto=sql_prompt),
        metadata=ToolMetadata(
        name = "sql_tool",
        description = "Consultas numéricas sobre la tabla 'resultados_gwr'."
        )    
    )
    
    contexto_tool = QueryEngineTool(
        query_engine=index_pdf.as_query_engine(
            similarity_top_k=4,
            llm=get_llm_model(),
            return_source=True,
            response_mode="compact",
        ),
        metadata = ToolMetadata(
        name = "contexto_tool",
        description = "Información complementaria del contexto."
        )   
    )
    
    return [sql_tool, contexto_tool]
###

def crear_motor_consulta(pregunta: str):
    vars_detectadas = obtener_variables(query_str = pregunta)
    prompt = prompt_variable if vars_detectadas else prompt_general
    #crear el generador de preguntas
    question_gen = LLMQuestionGenerator(llm = get_llm_model(), prompt=prompt)

    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=get_tools(),
        question_gen = question_gen,
        response_synthesizer = response_synthesizer,
        use_async = True,
        verbose = True
        )

@lru_cache(maxsize = 1)
def get_llm_naturalizador():
    return Ollama(
        model = "llama3.2:1b",
        temperature = 0.6,
        system_prompt = "Actúa como un narrador. Naturaliza los textos de una forma no técnica.",
        request_timeout = 300, 
    )

def ejecutar_consulta_completa(pregunta: str):
    """Ejecuta todo el flujo de consulta y devuelve la respuesta procesada"""
    try:
        # Obtiene herramientas cacheadas
        query_engine_tools = get_tools()
        
        # Verifica que las herramientas tengan la estructura correcta
        if not all(hasattr(tool.metadata, "name") for tool in query_engine_tools):
            raise ValueError("Las herramientas no tienen la estructura esperada")
        
        # Motor específico para esta pregunta
        motor = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            question_gen=LLMQuestionGenerator(
                llm=get_llm_model(),
                prompt=prompt_variable if obtener_variables(query_str=pregunta) else prompt_general
            ),
            response_synthesizer=get_response_synthesizer(
                response_mode="tree_summarize",
                summary_template=summary_prompt,
                use_async=True,
                llm=get_llm_model()
            ),
            use_async=True,
            verbose=True
        )
        
        # Ejecutar consulta
        respuesta_cruda = motor.query(pregunta)
        
        # Obtener resultados SQL de manera segura
        try:
            sql_tool = next((tool for tool in query_engine_tools if tool.metadata.name == "sql_tool"), None)
            if sql_tool:
                resultados_sql = sql_tool.query_engine.get_last_result()
            else:
                resultados_sql = {"csv": "No se encontraron datos SQL"}
        except Exception as e:
            resultados_sql = {"csv": f"Error al obtener datos SQL: {str(e)}"}
        # raw_sql = None
        # if hasattr(respuesta_cruda, "raw_sql"):
        #     raw_sql = respuesta_cruda.raw_sql
        # else:
        #     raw_sql = respuesta_cruda.response
        # clean_sql = extract_sql(raw_sql)
        # try:
        #     sql_tool = next(tool for tool in query_engine_tools if tool.metadata.name == "sql_tool")
        #     response_sql = sql_tool.query_engine.query(QueryBundle(clean_sql))
        #     resultados_sql = sql_tool.query_engine.get_last_result()
        # except Exception as e:
        #     resultados_sql = {"csv": f"Error al ejecutar SQL: {str(e)}"}

        # Procesar respuesta para hacerla más amigable
        respuesta_final = get_llm_naturalizador().predict(
            template_entrada,
            text=respuesta_cruda.response if hasattr(respuesta_cruda, 'response') else str(respuesta_cruda),
            datos=resultados_sql["csv"]
        )
        
        return {"respuesta": respuesta_final}
    
    except Exception as e:
        return {"respuesta": f"Error al procesar la consulta: {str(e)}"}

## CONTEXTOS Y TEMPLATES ##
contexto = (
    "Esta tabla muestra los resultados de los coeficientes locales obtenidos por un modelo GWR (Geographically Weighted Regression) "
    "aplicado a datos de propiedades en Nueva York. Los coeficientes locales reflejan cómo las características de las propiedades "
    "afectan el precio en diferentes ubicaciones geográficas. Esto incluye características como el número de dormitorios (BEDS), "
    "el tamaño de la propiedad (PROPERTYSQFT), y la distancia a puntos de interés como Central Park (d_cp) y Times Square (d_ts).\n\n"
    
    "Es importante entender que los coeficientes locales **no** representan los valores reales de las características, sino la **fuerza** "
    "del impacto que cada característica tiene sobre el precio en una ubicación específica. Un coeficiente local alto indica que esa "
    "característica tiene una gran influencia en el precio en esa área, pero no necesariamente que los valores de esa característica sean altos.\n\n"

    "Recuerda que un coeficiente local alto para 'BEDS' no implica que las propiedades en esa área tengan muchas habitaciones, sino que el "
    "número de dormitorios es un factor importante para determinar su precio en esa área.\n\n"
    
    "Los residuos ('residuals') no deben ser utilizados para evaluar el impacto de características, ya que solo reflejan "
    "el error de predicción y no la relación directa entre variables."
    "El campo 'INTERCEPT' representa el precio base de las viviendas cuando todas las demás características tienen valor cero.\n"

    "Descripcion de los campos:\n"
    "* PRICE: Precio de la vivienda en dólares. Es la variable objetivo del modelo GWR, que se ve influenciada por los coeficientes locales.\n"
    "* BROKERTITLE: Nombre del corredor o agente de bienes raíces que gestiona la propiedad en venta.\n"
    "* intercept: El valor base del precio de la vivienda cuando todas las demás características son cero. Este coeficiente refleja el precio medio base en una ubicación específica.\n"
    "* BATH: Fuerza del impacto del número de baños sobre el precio en una ubicación específica. No refleja la cantidad exacta de baños.\n"
    "* PROPERTYSQFT: Fuerza del impacto del tamaño de la propiedad (en pies cuadrados) sobre el precio. No refleja el tamaño exacto de las propiedades.\n"
    "* mi_avg: Fuerza del impacto del ingreso promedio de los residentes en el área sobre el precio. No indica el ingreso real.\n"
    "* tpa: tasa de población activa.\n"
    "* distancia a central park: esta variable se llama d_cp en la tabla sql.\n"
    "* distancia a times square: esta variable se llama d_ts en la tabla sql.\n"
    "* CAT_VM: Fuerza del impacto de que la vivienda sea multifamiliar sobre el precio.\n"
    "* CAT_VU: Fuerza del impacto de que la vivienda sea unifamiliar sobre el precio.\n"
    "* CAT_Otros: Fuerza del impacto de que la vivienda no sea ni multifamiliar ni unifamiliar sobre el precio.\n"
    "* residuals: Residuos del modelo GWR, que indican las diferencias entre los valores observados y los valores predichos para cada observación.\n"
    "* geometry: ubicación en (latitud, longitud) de la vivienda.\n"
    "* BoroName: Nombre del distrito al que pertenece la vivienda"
)

template_str_variable = """
Tu tarea es descomponer preguntas en múltiples subpreguntas más simples.

La pregunta que analizaremos está relacionada con la variable **{{ variable }}**. 

Descompón la siguiente pregunta **{{ question }}** en subpreguntas más simples relacionadas exclusivamente con {{ variable }}...

Herramientas disponibles:
- name: sql_tool ; description: Consultas numéricas sobre la tabla 'resultados_gwr'.
- name: contexto_tool ; description: Descripciones textuales del modelo y variables, basadas en documentos complementarios.

Reglas para descomponer:
- Cada subpregunta debe centrarse en un concepto específico de la pregunta original.
- Las subpreguntas deben poder responderse de forma independiente.
- No hagas preguntas de tipo sí/no a menos que sean esenciales.
- Usa términos presentes en la pregunta original (por ejemplo, si dice "Central Park", debes usar `d_cp`).
- Asigna `sql_tool` solo si la subpregunta puede resolverse mediante una consulta a la tabla SQL.
- Solo debes generar subpreguntas relacionadas con variables mencionadas explícitamente en la pregunta original.  
- No incluyas subpreguntas sobre variables no mencionadas o no relacionadas con el tema principal.
- Los ejemplos son solo ilustrativos. Evalúa siempre las variables directamente mencionadas en la pregunta original.

---

Ejemplo ilustrativo (solo válido si la variable es {{ variable }}):

[
  {"sub_question": "¿Cómo influye la {{ variable }} en los precios?", "tool_name": "sql_tool"},
  {"sub_question": "¿Qué representa {{ variable }} dentro del contexto urbano en este análisis?", "tool_name": "contexto_tool"},
  {"sub_question": "¿Qué implican los valores positivos y negativos del coeficiente de {{ variable }} en distintas zonas de la ciudad?", "tool_name": "contexto_tool"}
  {"sub_question": "..."},
  {"sub_question": "..."},
  ...
]

NO HAGAS ESTO (Ejemplos inválidos si la variable es {{ variable }}:
[
  {"sub_question": "¿Qué representa PROPERTYSQFT en el análisis?", "tool_name": "contexto_tool"},
  {"sub_question": "¿Cómo influye mi_avg sobre el precio?", "tool_name": "contexto_tool"},
]

REGLA ESTRICTA: 
**Solo genera subpregunta relacionada con la variable {{ variable }}**.

---

Tu salida debe ser **únicamente una lista JSON válida**, como esta:

[
  {"sub_question": "...", "tool_name": "sql_tool"},
  {"sub_question": "...", "tool_name": "contexto_tool"}
]

---

Descripcion de los variables:
    
* PRICE: Precio de la vivienda en dólares. Es la variable objetivo del modelo GWR, que se ve influenciada por los coeficientes locales.  
* BROKERTITLE: Nombre del corredor o agente de bienes raíces que gestiona la propiedad en venta.      
* intercept: El valor base del precio de la vivienda cuando todas las demás características son cero. Este coeficiente refleja el precio medio base en una ubicación específica.       
* BATH: Fuerza del impacto del número de baños sobre el precio en una ubicación específica. No refleja la cantidad exacta de baños.     
* PROPERTYSQFT: Fuerza del impacto del tamaño de la propiedad (en pies cuadrados) sobre el precio. No refleja el tamaño exacto de las propiedades.
* mi_avg: Fuerza del impacto del ingreso promedio de los residentes en el área sobre el precio. No indica el ingreso real.
* tpa: tasa de población activa.
* distancia a central park: esta variable se llama d_cp en la tabla sql. 
* distancia a times square: esta variable se llama d_ts en la tabla sql. 
* CAT_VM: Fuerza del impacto de que la vivienda sea multifamiliar sobre el precio.   
* CAT_VU: Fuerza del impacto de que la vivienda sea unifamiliar sobre el precio.   
* CAT_Otros: Fuerza del impacto de que la vivienda no sea ni multifamiliar ni unifamiliar sobre el precio.    
* residuals: Residuos del modelo GWR, que indican las diferencias entre los valores observados y los valores predichos para cada observación.  
* geometry: ubicación en (latitud, longitud) de la vivienda.
* BoroName: Nombre del distrito al que pertenece la vivienda
"""

template_str_general = """
Tu tarea es descomponer preguntas generales relacionadas con el modelo GWR en subpreguntas más simples.

Debes realizar un analisis de las siguientes variables del modelo:

* PRICE: Precio de la vivienda en dólares. Es la variable objetivo del modelo GWR, que se ve influenciada por los coeficientes locales.     
* intercept: El valor base del precio de la vivienda cuando todas las demás características son cero. Este coeficiente refleja el precio medio base en una ubicación específica.       
* BATH: Fuerza del impacto del número de baños sobre el precio en una ubicación específica. No refleja la cantidad exacta de baños.     
* PROPERTYSQFT: Fuerza del impacto del tamaño de la propiedad (en pies cuadrados) sobre el precio. No refleja el tamaño exacto de las propiedades.
* mi_avg: Fuerza del impacto del ingreso promedio de los residentes en el área sobre el precio. No indica el ingreso real.
* tpa: tasa de población activa.
* distancia a central park: esta variable se llama d_cp en la tabla sql. 
* distancia a times square: esta variable se llama d_ts en la tabla sql. 

---

Herramientas disponibles:
- name: sql_tool ; description: Consultas numéricas sobre la tabla 'resultados_gwr'.
- name: contexto_tool ; description: Descripciones textuales del modelo y variables, basadas en documentos complementarios.

Reglas para descomponer:
- No te enfoques en una sola variable.
- Considera incluir subpreguntas sobre la interpretación de coeficientes, zonas de mayor impacto, errores del modelo, etc.

---

Ejemplo ilustrativo:

[
  {"sub_question": "¿Cuál es el coeficiente local de {{ variable}} en el modelo GWR?", "tool_name": "sql_tool"},
  {"sub_question": "¿Qué representa {{ variable }} dentro del contexto urbano en este análisis?", "tool_name": "contexto_tool"},
  {"sub_question": "¿Cómo se interpreta un coeficiente negativo {{ variable }} en relación al precio de las viviendas?", "tool_name": "contexto_tool"}
  {"sub_question": "..."},
  {"sub_question": "..."},
  ...
]

Tu salida debe ser **únicamente una lista JSON válida**, como esta:

[
  {"sub_question": "...", "tool_name": "sql_tool"},
  {"sub_question": "...", "tool_name": "contexto_tool"}
]

"""

#PROMPTS
template_var_mappings = {"context": "contexto", "question":"question"}
function_mappings = {"variable": obtener_variables}

output_parser = SubQuestionOutputParser()

prompt_variable = RichPromptTemplate(
    template_str = template_str_variable,
    #template_var_mappings = template_var_mappings,
    template_var_mappings = {"question": "question","variable": "variable"},
    #function_mappings = function_mappings,
    function_mappings={"variable": obtener_variables},
    output_parser = output_parser,
    prompt_type=PromptType.SUB_QUESTION
)

prompt_general = RichPromptTemplate(
    template_str = template_str_general,
    template_var_mappings={"context": "contexto","question": "question"},
    output_parser = output_parser,
    prompt_type=PromptType.SUB_QUESTION
)



###


# if __name__ == "__main__":
#     # Configuración básica para pruebas
#     import time
#     from pprint import pprint
    
#     print("=== INICIO DE PRUEBAS ===")
#     tools = get_tools()
#     for tool in tools:
#         print(f"Tool name: {tool.metadata.name}")
#         print(f"Description: {tool.metadata.description}")
#         print(f"Query engine type: {type(tool.query_engine)}")

#     # 1. Prueba de funciones básicas
#     print("\n1. Probando obtener_variables():")
#     test_preguntas = [
#         "¿Cómo afectan los baños al precio?",
#         "Qué influencia tiene la distancia a Times Square",
#         "Analiza el impacto de viviendas multifamiliares"
#     ]
    
#     for pregunta in test_preguntas:
#         vars_detectadas = obtener_variables(query_str=pregunta)
#         print(f"Pregunta: '{pregunta}' -> Variables detectadas: {vars_detectadas}")
    
#     # 2. Prueba de consulta completa
#     print("\n2. Probando ejecutar_consulta_completa():")
#     test_consultas = [
#         "¿Cómo influye el número de baños en el precio?",
#         "Explica el efecto de la distancia a Central Park en Manhattan",
#         "Compara el impacto de viviendas unifamiliares vs multifamiliares en Brooklyn"
#     ]
    
#     for consulta in test_consultas:
#         print(f"\nConsulta: '{consulta}'")
#         start_time = time.time()
        
#         try:
#             resultado = ejecutar_consulta_completa(consulta)
#             elapsed = time.time() - start_time
            
#             print(f"\nResultado ({elapsed:.2f}s):")
#             print("="*50)
#             print(resultado["respuesta"])
#             print("="*50)
            
#             # Verificar si se obtuvieron datos SQL
#             if "sql_tool" in [t.metadata.name for t in get_tools()]:
#                 sql_tool = next(t for t in get_tools() if t.metadata.name == "sql_tool")
#                 try:
#                     datos = sql_tool.query_engine.get_last_result()
#                     print("\nDatos SQL obtenidos:")
#                     print(datos["dataframe"].head(3))
#                 except Exception as e:
#                     print(f"\nError al obtener datos SQL: {str(e)}")
                    
#         except Exception as e:
#             print(f"\nError en consulta: {str(e)}")
    
#     print("\n=== PRUEBAS COMPLETADAS ===")
    
