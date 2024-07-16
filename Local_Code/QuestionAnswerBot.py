import os
import time

import torch
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from sentence_transformers import SentenceTransformer
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    
    load_index_from_storage,


)

from transformers import pipeline

from llama_index.core.schema import Document# import Document
from llama_index.core import VectorStoreIndex

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.service_context import ServiceContext, set_global_service_context

os.environ["PINECONE_API_KEY"] = "d5f546cf-87c9-4a93-8ca1-af1030d5c37c"
PINECONE_API_KEY = "d5f546cf-87c9-4a93-8ca1-af1030d5c37c"
LLAMA_CLOUD_API_KEY = "llx-Jce36FVDag4H86sqG3OfuaII5O0mB3OwMo4HcrTGtLgdbYyY"
os.environ["LLAMA_CLOUD_API_KEY"]="llx-Jce36FVDag4H86sqG3OfuaII5O0mB3OwMo4HcrTGtLgdbYyY"

system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
# This will wrap the default prompts that are internal to llama-index
from llama_index.core.prompts.prompts import SimpleInputPrompt
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


pc = Pinecone(
        api_key="d5f546cf-87c9-4a93-8ca1-af1030d5c37c",
        environment = "us-east-1"
)


# Download Embedding Models

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
model.save('.\\sentence-transformers',)

#Load the embedding model
local_model_path = ".\\sentence-transformers"
Settings.embed_model = HuggingFaceEmbedding(model_name=local_model_path)

embed_model = HuggingFaceEmbedding(
    model_name=local_model_path
)

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    
    model_kwargs={"torch_dtype": torch.bfloat16}
)

Settings.llm=llm
'''
class LocalOPT(HuggingFaceLLM):

    #model_name = "facebook/opt-iml-1.3b" 
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  #2.20GB
    #model_name= "meta-llama/Meta-Llama-3-8B"
    # 2.63 GB Model
    pipeline = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype":torch.bfloat16}#, "cache_dir":"D:\\AIML\\models"}
    )


    def _call(self,prompt:str,stop=None) ->str:
        response= self.pipeline(prompt, max_new_tokens=256)[0]["generated_text"]
        return response[len(prompt) :]

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self):
        return "custom"

'''

def create_index():
    print("Creating index")
    
    parser2 = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
        result_type="text",  # "markdown" and "text" are available
        verbose=False,
        language='en'
    )
    documents = parser2.load_data("Assignment_Support_Document.pdf")
    print("PDF COntent Length ",{len(documents)})

        #Load the embedding model
    local_model_path = "D:\\AIML\\Local_code\\sentence-transformers"
    

    embed_model = HuggingFaceEmbedding(
        model_name=local_model_path
    )
    Settings.embed_model =embed_model
    #Creating Nodes 
    # Initialize the parser
    parser = SentenceSplitter.from_defaults(chunk_size=512, chunk_overlap=20)

    import pprint
    # Parse documents into nodes or chunks
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Number of nodes created: {len(nodes)}")
    #pprint.pprint([nodes[i] for i in range(3)])
    pc = Pinecone(  
        api_key="d5f546cf-87c9-4a93-8ca1-af1030d5c37c",
        environment = "us-east-1"
    )

    # Now do stuff
    if 'nagp' not in pc.list_indexes().names():
        print("Index not present on Pinecone, creating one..")
        pc.create_index(
            name='nagp',
            dimension=768,#1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    # construct vector store and customize storage context
    storage_context = StorageContext.from_defaults(
        vector_store=PineconeVectorStore(pc.Index("nagp",host="https://nagp-qtm90cl.svc.aped-4627-b74a.pinecone.io"))
    )

    # Filter out empty nodes
    filtered_nodes = [node for node in nodes if node.get_content() != ""]
      
    print(f"No. of Non EMpty Nodes: {len(filtered_nodes)}")
    # Create an index from the filetered nodes
    index = VectorStoreIndex(filtered_nodes, storage_context=storage_context)
    ## save index to disk. Will be stored in ./storage by default
    index.set_index_id("nagp")
    # save index to disk. Will be stored in ./storage by default
    index.storage_context.persist()
    print("Done creating index",index)
    return index



def execute_query(index):

    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query("Estimated Tax receipts in 2024-2025")
    return response
    

if __name__ == "__main__":
   
    if not os.path.exists("./storage/index_store.json"): #()
        print("No Local cache of model found, downloading from huggingface")
        index = create_index()
        print("Executing Query Now")
        response = execute_query(index)
        print(response)
    else:
        print("Loading from local cache of model")
        # store embeddings in pinecone index
        vector_store = PineconeVectorStore(pinecone_index=pc.Index("nagp",host="https://nagp-qtm90cl.svc.aped-4627-b74a.pinecone.io"))

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


        print("Executing Query Now")
        start_time = time.time()
        response = execute_query(index)
        time_elapsed = time.time() - start_time
        print(response)
        print("total time taken for response in seconds ",time_elapsed)
       





    




