import os
from pathlib import Path

import faiss
import gradio as gr
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext
)
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import (
    completion_to_prompt,
    messages_to_prompt
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import FaissVectorStore
from llama_index.storage import StorageContext


_MODELS_DIR = Path('models')
_DOCS_DIR = Path('data')
_LLM_PATH = _MODELS_DIR / 'llama-2-7b-chat.Q4_K_M.gguf'
_EMBED_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
_EMBED_DIM = 384
_GRADIO_SERVER_NAME = '0.0.0.0'
_GRADIO_SERVER_PORT = 7860
os.environ['NLTK_DATA'] = str(_MODELS_DIR)


def main():
    llm = LlamaCPP(
        model_path=str(_LLM_PATH),
        model_kwargs={'n_gpu_layers': -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt
    )
    embed_model = HuggingFaceEmbedding(
        model_name=_EMBED_MODEL_NAME,
        cache_folder=_MODELS_DIR
    )
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )
    documents = SimpleDirectoryReader(_DOCS_DIR).load_data()
    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(_EMBED_DIM))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        service_context=service_context,
        storage_context=storage_context
    )
    query_engine = index.as_query_engine(streaming=True)

    _USE_AUGMENTATION = False  # initially do not use retrieval augmentation

    def _gen_response(query, history):
        tokens = ''
        nonlocal _USE_AUGMENTATION
        if _USE_AUGMENTATION:  # use retrieval augmentation
            for token in query_engine.query(query).response_gen:
                tokens += token
                yield tokens
        else:  # vanilla LLM response
            for token in llm.stream_complete(query):
                tokens += token.delta
                yield tokens
    
    def _on_select(evt: gr.SelectData):
        nonlocal _USE_AUGMENTATION
        _USE_AUGMENTATION = evt.selected 
    
    with gr.Blocks() as interface:
        gr.ChatInterface(fn=_gen_response)
        with gr.Blocks():
            gr.Checkbox(label='Augment with documents').select(_on_select)
            gr.Textbox(
                value='\n'.join(str(i) for i in _DOCS_DIR.glob('*')),
                label='Documents'
            )
        interface.queue()
        interface.launch(
            server_name=_GRADIO_SERVER_NAME,
            server_port=_GRADIO_SERVER_PORT
        )

if __name__ == '__main__':
    main()
