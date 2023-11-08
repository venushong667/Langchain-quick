import requests
import re
from bs4 import BeautifulSoup
from llama_index import (
    PromptTemplate,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    KnowledgeGraphIndex,
    StorageContext,
    Document
)
from llama_index.schema import MetadataMode
from llama_index.node_parser import SimpleNodeParser
from llama_index.graph_stores import SimpleGraphStore
from llama_index.llms import HuggingFaceLLM

def get_bbc_stories():
    page = requests.get('https://www.bbc.com/news')

    soup = BeautifulSoup(page.content, "html.parser")
    link_elements = soup.select("a[href]")
    urls = [link_elem['href'] for link_elem in link_elements]

    divs = soup.select('div')
    stories = [div for div in divs if div.get('data-entityid')]
    for story in stories:
        # icon_elems = story.select('.gel-icon')
        icon_elems = story.select('.gs-u-vh')
        for elem in icon_elems:
            elem.decompose()
    return stories

def get_news():
    stories = get_bbc_stories()
    news = []

    for story in stories:
        entityid = story.get('data-entityid')
        if 'top-stories' not in entityid:
            continue

        url = story.select_one("a[href]").get('href')

        location = None
        loc_elem = story.select_one('.nw-o-link--no-visited-state')
        if loc_elem:
            location = loc_elem.text
            loc_elem.decompose()

        datetime = 'Live'
        hours_ago = None
        timestamp = None
        dt_elem = story.select_one('.nw-c-timestamp time')
        if dt_elem:
            datetime = dt_elem.get('datetime')
            hours_ago = dt_elem.text
            timestamp = dt_elem.get('data-seconds')
            # ipdb.set_trace()
            dt_elem.decompose()

        new = {
            'text': story.getText(' | '),
            'url': url,
            'from': location,
            'datetime': datetime,
            'timestamp': timestamp,
            'hours_ago': hours_ago
        }
        news.append(new)
    
    return news

def init_storage_context():
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    return storage_context

def init_service_context():
    query_wrapper_prompt = PromptTemplate(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{query_str}\n\n### Response:"
    )
    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.1, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device_map="auto",
        tokenizer_kwargs={"max_length": 2048},
        model_kwargs={
            'cache_dir': './cache',
            'load_in_8bit': True
        }
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16}
    )
    service_context = ServiceContext.from_defaults(chunk_size=512, llm=llm)
    return service_context, llm

def init_index(storage_context, service_context):
    index = KnowledgeGraphIndex(
        [],
        service_context=service_context,
        storage_context=storage_context
    )

    return index

def convert_documents(data):
    documents = []
    for d in data:
        document = Document(
            text=d['text'],
            metadata={
                'url': d['url'],
                'from': d['from'],
                'datetime': d['datetime'],
                'timestamp': d['timestamp'],
                'hours_ago': d['hours_ago']
            },
            excluded_llm_metadata_keys=['url', 'timestamp', 'datetime'],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
        documents.append(document)
        
    return documents

# def extract_triplets(nodes):


def extract_nodes(index, data, llm: HuggingFaceLLM):
    nodes = None
    node_parser = SimpleNodeParser.from_defaults()
    documents = convert_documents(data)

    nodes = node_parser.get_nodes_from_documents(documents)
    IE_prompt = """[INST]
    Based on relation extraction, extract entity pairs with the appropriate relation type from given context in format (entity1, relation_type, entity2). 
    Context: """
    node_tups = []
    for node in nodes:
        inputs = IE_prompt + node.get_text() + " [/INST]"
        res = llm.complete(inputs).text
        import ipdb; ipdb.set_trace()
        relations, explain = res.split('Note: ')
        relations = [rel for rel in relations.split('\n')if rel]
        node_tups.append((node, relations))

    # node_0_tups = [
    #     ("author", "worked on", "writing"),
    #     ("author", "worked on", "programming"),
    # ]
    # for tup in node_0_tups:
    #     index.upsert_triplet_and_node(tup, nodes[0])
    # extract_triplets()

    return nodes

if __name__ == "__main__":
    data = get_news()
    storage_context = init_storage_context()
    service_context, llm = init_service_context()
    index = init_index(storage_context, service_context)
    nodes = extract_nodes(index, data, llm)
    
