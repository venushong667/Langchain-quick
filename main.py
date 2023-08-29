import os
import sys

from dataclasses import dataclass, field
from dotenv import load_dotenv
from transformers import HfArgumentParser
from components import create_documents, init_chain, init_components, init_pipeline, init_vector_store

from utils import read_json, build_llama2_prompt

load_dotenv()

@dataclass
class ConfigArguments:
    hf_auth: str = field(
        default=None, metadata={"help": "Huggingface Auth Token to access models."}
    )
    main_model_id: str = field(
        default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "Main model to perform conversation."}
    )
    embedding_model_id: str = field(
        default="sentence-transformers/all-mpnet-base-v2", metadata={"help": "Embedding model to embed documents into vector store."}
    )
    vector_store: str = field(
        default="FAISS", metadata={"help": "Vector storage type."}
    )
    chain_type: str = field(
        default="conversation-retrieval", metadata={"help": "LLM pipeline chain type."}
    )
    cache_dir: str = field(
        default="./cache", metadata={"help": "Model cache directory path."}
    )

def inference(chain):
    chat_history = []
    # query = "What is Data lakehouse architecture in Databricks?"
    while True:
        query = input("Question: ").strip()
        print("Query: ", query)
        print("Generating....")

        # result = chain({"question": query, "chat_history": chat_history})
        # answer = result['answer'].strip()
        
        # chat_history.append((query, answer))
        # print(result)
        input_key = getattr(chain, "input_key", chain.input_keys[0])
        
        result = chain({input_key: query})

        print("--------------------------------------------")
        print("Response: ", result)
        print("--------------------------------------------")

def inference_model(model, tokenizer):
#     prompt = """[INST] <<SYS>>
# The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
# <</SYS>>

# {input} [/INST]"""
    chat_history = [
        {"role": "system", "content": "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."}
    ]
    user_role = "user"
    model_role = "assistant"
    while True:
        query = input("Question: ").strip()
        chat_history.append({"role": user_role, "content": query})
        model_input = build_llama2_prompt(chat_history)
        print(model_input)
        input_ids = tokenizer(
            model_input,
            return_tensors="pt",
            add_special_tokens=False
        ).input_ids.to("cuda")
        output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            max_new_tokens=512
        )
        output_seq = tokenizer.decode(output.sequences[0])
        answer = output_seq.split("[/INST] ")[-1]
        print(answer)
        chat_history.append({"role": model_role, "content": answer})

def main():
    parser = HfArgumentParser(ConfigArguments)
    if len(sys.argv) > 2:
        config_args = parser.parse_args_into_dataclasses()[0]
    else:
        config_file_path = os.path.abspath(sys.argv[1]) if len(sys.argv) == 2 and sys.argv[1].endswith(".json") else "./config.json"
        config_args = parser.parse_json_file(json_file=config_file_path)[0]
    config_args.hf_auth = config_args.hf_auth or os.getenv("HUGGING_FACE_HUB_TOKEN")
    print(config_args)
    print("Initalizing....")

    model, tokenizer = init_components(
        config_args.main_model_id,
        config_args.cache_dir,
        config_args.hf_auth
    )
    
    # enable evaluation mode to allow model inference
    model.eval()

    llm = init_pipeline(model, tokenizer)

    retrieval = config_args.chain_type.split("-")[-1] == "retrieval"
    vector_store = None
    if retrieval:
        documents, docs_splits = create_documents()
        vector_store = init_vector_store(config_args.vector_store, config_args.embedding_model_id, docs_splits)
    
    chain = init_chain(config_args.chain_type, llm, vector_store)
    
    inference(chain)
    # inference_model(model, tokenizer)

if __name__ == "__main__":
    main()