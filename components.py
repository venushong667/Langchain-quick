from typing import Union
import torch
from torch import bfloat16

import transformers
from transformers import StoppingCriteria, StoppingCriteriaList

from langchain import Wikipedia
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.agents.agent import AgentExecutor, Agent
from langchain.agents.react.base import DocstoreExplorer
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

from agents import ReActDocstoreAgent


PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times until you think the question is fully answered)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Previous chat history: {chat_history}
Question: {input}
Thought:{agent_scratchpad}"""

def init_components(model_id, cache_dir=None, hf_auth=None):
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        cache_dir=cache_dir,
        load_in_8bit=True,
        device_map={"":0},
        use_auth_token=hf_auth,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        use_auth_token=hf_auth
    )

    print("Initalized model and tokenizer.")

    return model, tokenizer

def init_pipeline(model, tokenizer):
    stop_list = ['\nHuman:', '\n```\n']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to("cuda") for x in stop_token_ids]
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        # stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)

    # checking again that everything is working fine
    # output = llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse.")
    # print("Dummy output: ", output)

    print("Initalized pipeline.")

    return llm

def create_documents():
    
    web_links = [
        "https://www.databricks.com/","https://help.databricks.com","https://databricks.com/try-databricks","https://help.databricks.com/s/","https://docs.databricks.com","https://kb.databricks.com/","http://docs.databricks.com/getting-started/index.html","http://docs.databricks.com/introduction/index.html","http://docs.databricks.com/getting-started/tutorials/index.html","http://docs.databricks.com/release-notes/index.html","http://docs.databricks.com/ingestion/index.html","http://docs.databricks.com/exploratory-data-analysis/index.html","http://docs.databricks.com/data-preparation/index.html","http://docs.databricks.com/data-sharing/index.html","http://docs.databricks.com/marketplace/index.html","http://docs.databricks.com/workspace-index.html","http://docs.databricks.com/machine-learning/index.html","http://docs.databricks.com/sql/index.html","http://docs.databricks.com/delta/index.html","http://docs.databricks.com/dev-tools/index.html","http://docs.databricks.com/integrations/index.html","http://docs.databricks.com/administration-guide/index.html","http://docs.databricks.com/security/index.html","http://docs.databricks.com/data-governance/index.html","http://docs.databricks.com/lakehouse-architecture/index.html","http://docs.databricks.com/reference/api.html","http://docs.databricks.com/resources/index.html","http://docs.databricks.com/whats-coming.html","http://docs.databricks.com/archive/index.html","http://docs.databricks.com/lakehouse/index.html","http://docs.databricks.com/getting-started/quick-start.html","http://docs.databricks.com/getting-started/etl-quick-start.html","http://docs.databricks.com/getting-started/lakehouse-e2e.html","http://docs.databricks.com/getting-started/free-training.html","http://docs.databricks.com/sql/language-manual/index.html","http://docs.databricks.com/error-messages/index.html","http://www.apache.org/","https://databricks.com/privacy-policy","https://databricks.com/terms-of-use"
    ] 

    loader = WebBaseLoader(web_links)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    print("Initalized documents.")

    return documents, all_splits


def init_vector_store(vector_store_type, embed_model_id, docs_splits):
    embeddings = HuggingFaceEmbeddings(model_name=embed_model_id)

    # storing embeddings in the vector store
    vector_store_dict = {
        "FAISS": FAISS
    }
    vector_class = vector_store_dict.get(vector_store_type)
    if not vector_class:
        raise ValueError("Vector store not supported.")
    
    store = vector_class.from_documents(docs_splits, embeddings)

    print("Initalized vector store.")

    return store

def init_doc_store(doc_type="wikipedia"):
    doc_store_dict = {
        "wikipedia": Wikipedia
    }

    doc_class = doc_store_dict.get(doc_type)
    if not doc_class:
        raise ValueError(f"Doc store '{doc_type}' not supported.")
    
    doc_store = DocstoreExplorer(doc_class())

    tools = [
        Tool(
            name="Search",
            func=doc_store.search,
            description="useful for when you need to ask with search",
        ),
        Tool(
            name="Lookup",
            func=doc_store.lookup,
            description="useful for when you need to ask with lookup",
        ),
    ]
    return doc_store, tools

def init_chain(chain_type, llm, vector_store_type=None, embedding_model_id=None,doc_store_type=None):
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    chain_dict = {
        "conversation-retrieval": ConversationalRetrievalChain,
        "conversation": ConversationChain,
        "zero-shot-react-agent": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        "conversation-react-agent": AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        "react-docstore": ReActDocstoreAgent
    }
    chain_class = chain_dict.get(chain_type)
    if not chain_class:
        raise ValueError("Chain Type not supported.")
    print("Initialized chain type: ", chain_class)
    
    if vector_store_type:
        documents, docs_splits = create_documents()
        vector_store = init_vector_store(vector_store, embedding_model_id, docs_splits)
    
    if doc_store_type:
        doc_stor, tools = init_doc_store(doc_store_type)

    if chain_class in [t for t in AgentType] + [ReActDocstoreAgent]:
        chain = init_agent(chain_class, llm, tools=tools)
    else:
        if vector_store:
            chain = chain_class.from_llm(
                llm,
                vector_store.as_retriever(),
                return_source_documents=True
            )
        else:
            # chain = ConversationChain(
            #     prompt=PROMPT,
            #     llm=llm,
            #     verbose=True,
            #     memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
            # )
            
            chain = ConversationChain(
                llm=llm,
                verbose=True
            )
    print("Initalized chain.")

    return chain

def init_agent(agent_type: Union[AgentType, Agent], llm, tools=["serpapi", "llm-math"], verbose=True):
    # memory = ConversationBufferMemory(memory_key="chat_history")
    # agent_kwargs={"prefix": PREFIX, "suffix": SUFFIX, "format_instructions": FORMAT_INSTRUCTIONS},
    # memory=memory,
    if Agent in agent_type.__mro__:
        agent_obj = agent_type.from_llm_and_tools(
            llm, tools
        )
        agent = AgentExecutor.from_agent_and_tools(
            agent=agent_obj, tools=tools, verbose=True
        )
    else:
        agent = initialize_agent(
            tools,
            llm,
            agent=agent_type,
            verbose=verbose
        )
    
    return agent