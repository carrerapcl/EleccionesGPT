import os

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

import openai
openai.api_key = os.getenv('OPENAI_KEY')

# COnfigure LLM service
llm = OpenAI(temperature=0.5, model="gpt-3.5-turbo", max_tokens=-1)
service_context = ServiceContext.from_defaults(llm=llm)

# Load data (PDF files)
pp_docs = SimpleDirectoryReader(input_files=["Programas/programa_electoral_pp_23j_feijoo_2023.pdf"]).load_data()
psoe_docs = SimpleDirectoryReader(input_files=["Programas/PROGRAMA_ELECTORAL-GENERALES-2023.pdf"]).load_data()
sumar_docs = SimpleDirectoryReader(input_files=["Programas/SUMAR_Un-Programa-para-ti.pdf"]).load_data()
vox_docs = SimpleDirectoryReader(input_files=["Programas/Programa-VOX-2023.pdf"]).load_data()

# Build indices
pp_index = VectorStoreIndex.from_documents(pp_docs)
psoe_index = VectorStoreIndex.from_documents(psoe_docs)
sumar_index = VectorStoreIndex.from_documents(sumar_docs)
vox_index = VectorStoreIndex.from_documents(vox_docs)

# Build query engines
pp_engine = pp_index.as_query_engine(similarity_top_k=3)
psoe_engine = psoe_index.as_query_engine(similarity_top_k=3)
sumar_engine = sumar_index.as_query_engine(similarity_top_k=3)
vox_engine = vox_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=pp_engine,
        metadata=ToolMetadata(
            name="programa_pp",
            description="Proporciona informacion sobre el programa electoral del partido politico PP para las elecciones generales de 2023",
        ),
    ),
    QueryEngineTool(
        query_engine=psoe_engine,
        metadata=ToolMetadata(
            name="programa_psoe",
            description="Proporciona informacion sobre el programa electoral del partido politico PSOE para las elecciones generales de 2023",
        ),
    ),
    QueryEngineTool(
        query_engine=sumar_engine,
        metadata=ToolMetadata(
            name="programa_sumar",
            description="Proporciona informacion sobre el programa electoral del partido politico SUMAR para las elecciones generales de 2023",
        ),
    ),
    QueryEngineTool(
        query_engine=vox_engine,
        metadata=ToolMetadata(
            name="programa_vox",
            description="Proporciona informacion sobre el programa electoral del partido politico VOX para las elecciones generales de 2023",
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)

# Run queries
# response = s_engine.query(
#     "Cuales son las propuestas de los diferentes partidos en materia de energia nuclear?"
# )
response = s_engine.query(
    "Si soy progresista en lo social, y liberal en lo economico... cual es el partido que mejor se adapta a mis ideales?"
)

print(response)
