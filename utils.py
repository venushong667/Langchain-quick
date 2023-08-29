import json

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def build_llama2_prompt(messages):
    prompt_seq = [
        {"role": "system", "content": "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."},
        {"role": "user", "content": "{input}"},
    ]
    startPrompt = "<s>[INST] "
    endPrompt = " [/INST]"

    def mapping(message):
        role = message["role"]
        content = message["content"]
        if role == 'user':
            return content.strip() + endPrompt
        elif role == 'assistant':
            EOS = "</s>" if "</s>" not in content else ""
            return f" {content}{EOS}<s>[INST] "
        elif role == 'function':
            raise ValueError('Llama 2 does not support function calls.')
        elif role == 'system':
            return f"<<SYS>>\n{content}\n<</SYS>>\n\n"
        else:
            raise ValueError(f"Invalid message role: {role}")
    
    conversation = list(map(mapping, messages))

    return startPrompt + ''.join(conversation)