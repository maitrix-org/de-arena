import torch
import json
import sys

if torch.cuda.is_available():
    print(f"Total CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices are available.")

existing_model_paths = {
    'gpt-4o-mini-2024-07-18' : "OPENAI", # 1 finish old
    'gpt4-1106' : "OPENAI",
    'gpt3.5-turbo-0125' : "OPENAI", # 2 finish old
    # "o1-preview" : "OPENAI", 
    # "o1-mini" : "OPENAI", 
    "ChatGPT-4o-latest (2024-09-03)" : "OPENAI", # 4
    "gpt-4o-2024-08-06" : "OPENAI", # 3 running old
    "gpt-4-turbo-2024-04-09" : "OPENAI",
    "gpt-4o-2024-05-13" : "OPENAI",

    "claude-3.5-sonnet" : "Claude", 
    # "claude-3.5-sonnet-20241022" : "Claude",
    "claude-3-opus" : "Claude",
    "claude-3-sonnet" : "Claude",
    "claude-3-haiku" : "Claude", 
    "claude-2.0" : "Claude",
    "claude-2.1" : "Claude",

    # "gemini-1.5-flash-001" : "gemini", 
    # "gemini-1.5-pro-001" : "gemini",
    # "gemini-1.0-pro-001" : "gemini",

    # "llama-3.2-1b-it" : "/data/shared/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14",
    # "llama-3.2-3b-it" : "/data/shared/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/392a143b624368100f77a3eafaa4a2468ba50a72",
    # "llama-3.2-11b-vision-it" : "/data/shared/huggingface/hub/models--meta-llama--Llama-3.2-11B-Vision-Instruct/snapshots/075e8feb24b6a50981f6fdc161622f741a8760b1",
    # "llama-3.2-90b-vision-it" : "/data/shared/huggingface/hub/models--meta-llama--Llama-3.2-90B-Vision-Instruct/snapshots/ec79ddbe5c61f433eb855f4007bb6cb158ad3b35",


    # 'athene-70b' : "/data/shared/huggingface/hub/models--Nexusflow--Athene-70B/snapshots/4b070bdb1c5fb02de52fe948da853b6980c75a41",

    "gemma-1.1-7b-it" : "/data/shared/huggingface/hub/models--google--gemma-1.1-7b-it/snapshots/065a528791af6f57f013e8e42b7276992b45ef71",
    "gemma-2-27b-it" : "/data/shared/huggingface/hub/models--google--gemma-2-27b-it/snapshots/2d74922e8a2961565b71fd5373081e9ecbf99c08",
    # "gemma-2-9b-it-simpo" : "/data/shared/huggingface/hub/models--princeton-nlp--gemma-2-9b-it-SimPO/snapshots/8c87091f412e3aa6f74f66bd86c57fb81cbc3fde",
    "google-gemma-2-9b-it" : "/data/shared/huggingface/hub/models--google--gemma-2-9b-it/snapshots/4efc01a1a58107f8c7f68027f5d8e475dfc34a6f",
    "gemma-2-2b-it" : "/data/shared/huggingface/hub/models--google--gemma-2-2b-it/snapshots/299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
    "gemma-1.1-2b-it" : "/data/shared/huggingface/hub/models--google--gemma-1.1-2b-it/snapshots/d750f5eceb83e978c09e2b3597c2a8784e381022",
    "gemma-2b-it" : "/data/shared/huggingface/hub/models--google--gemma-2b-it/snapshots/de144fb2268dee1066f515465df532c05e699d48",
    "gemma-7b-it" : "/data/shared/huggingface/hub/models--google--gemma-7b-it/snapshots/18329f019fb74ca4b24f97371785268543d687d2",
    
    # "yi-34b-chat" : "/data/shared/huggingface/hub/models--01-ai--Yi-34B-Chat/snapshots/493781d21ad8992f4875668eff44d5af58f4e96b",
    "yi-1.5-34b-chat" : "/data/shared/huggingface/hub/models--01-ai--Yi-1.5-34B-Chat/snapshots/fa4ffba162f20948bf77c2a30eca952bf0812b7f",

    "mistral-7b-instruct-2" : "/data/shared/huggingface/hub/mistral-inst-7B-v0.2",
    "mistral-7b-instruct-1" : "/data/shared/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.1/snapshots/73068f3702d050a2fd5aa2ca1e612e5036429398",
    "mistral-8x7b-instruct-v0.1" : "/data/shared/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/1e637f2d7cb0a9d6fb1922f305cb784995190a83",
    
    "llama2-13b-chat" : "/data/shared/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/snapshots/a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8",
    "llama2-7b-chat" : "/data/shared/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
    # "llama3-8b-instruct" : "/data/shared/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/a8977699a3d0820e80129fb3c93c20fbd9972c41",

    "command-r-(04-2024)" : "/data/shared/huggingface/hub/models--CohereForAI--c4ai-command-r-v01/snapshots/16881ccde1c68bbc7041280e6a66637bc46bfe88",
    "command-r-(08-2024)" : "/data/shared/huggingface/hub/models--CohereForAI--c4ai-command-r-08-2024/snapshots/f8d837566c7bfe038870477e83f97f14e341cca6",

    "qwen1.5-14b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-14B-Chat/snapshots/9492b22871f43e975435455f5c616c77fe7a50ec",
    "qwen1.5-32b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-32B-Chat/snapshots/0997b012af6ddd5465d40465a8415535b2f06cfc",
    "qwen2-72b-instruct": "/data/shared/huggingface/hub/models--Qwen--Qwen2-72B-Instruct/models--Qwen--Qwen2-72B-Instruct/snapshots/1af63c698f59c4235668ec9c1395468cb7cd7e79",
    "qwen1.5-4b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-4B-Chat/snapshots/a7a4d4945d28bac955554c9abd2f74a71ebbf22f",
    "qwen1.5-72b-chat" : "/data/shared/huggingface/hub/models--Qwen--Qwen1.5-72B-Chat/snapshots/d341a6f2cb937e7a830ecbe3ab7b87215bc3a6b0",

    "openchat-3.5" : "/data/shared/huggingface/hub/models--openchat--openchat_3.5/snapshots/c8ac81548666d3f8742b00048cbd42f48513ba62",
    "openchat-3.5-0106" : "/data/shared/huggingface/hub/models--openchat--openchat-3.5-0106/snapshots/f3b79c43f12da94b56565c5fc5a65d40e696c876",

    "zephyr-7b-beta": "/data/shared/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/b70e0c9a2d9e14bd1e812d3c398e5f313e93b473",

    "vicuna-7b" : "/data/shared/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
    "vicuna-13b" : "/data/shared/huggingface/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2",
    "vicuna-33b" : "/data/shared/huggingface/hub/models--lmsys--vicuna-33b-v1.3/snapshots/ef8d6becf883fb3ce52e3706885f761819477ab4",

    # "meta-llama-3.1-70b-instruct" : "/data/shared/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393",
    # "meta-llama-3.1-8b-instruct" : "/data/shared/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693",
    # "llama-3-70b-instruct" : "/data/shared/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c",
    # "nemotron-70b" : "/data/shared/huggingface/hub/models--nvidia--Llama-3.1-Nemotron-70B-Instruct-HF/snapshots/b919e5d07ce15f31ea741f2be99a00a33c3b427b",

    # "jamba-1.5-mini" : "/data/shared/huggingface/hub/models--ai21labs--AI21-Jamba-1.5-Mini/snapshots/7cbad0fc5121cb91c6363bacb815fe3aae2ec4f9",
    "starling-lm-7b-beta" : "/data/shared/huggingface/hub/models--Nexusflow--Starling-LM-7B-beta/snapshots/39a4d501472dfede947ca5f4c5af0c1896d0361b",
    "starling-lm-7b-alpha" : "/data/shared/huggingface/hub/models--berkeley-nest--Starling-LM-7B-alpha/snapshots/1dddf3b95bc1391f6307299eb1c162c194bde9bd",
    "koala-13b" : "/data/shared/huggingface/hub/models--TheBloke--koala-13B-HF/snapshots/b20f96a0171ce4c0fa27d6048215ebe710521587",
    "openassistant-pythia-12b" : "/data/shared/huggingface/hub/models--OpenAssistant--oasst-sft-1-pythia-12b/snapshots/293df535fe7711a5726987fc2f17dfc87de452a1",

}

gt_scores = {
    "o1-preview" : 1355,
    "o1-mini" : 1313,
    "claude-3.5-sonnet" : 1268,
    "claude-3-opus" : 1248,
    "claude-3-sonnet" : 1201,
    "claude-3-haiku" : 1179,
    "claude-2.0" : 1132,
    "claude-2.1" : 1118,
    "ChatGPT-4o-latest (2024-09-03)" : 1339,
    "gpt-4o-mini-2024-07-18" : 1274,
    "gpt-4o-2024-08-06" : 1264,
    "gpt-4-Turbo-2024-04-09" : 1257,
    "gpt-4o-2024-05-13" : 1285,
    'gpt4-1106' : 1250,
    'gpt3.5-turbo-0125' : 1106,
    "athene-70b": 1250,
    "jamba-1.5-mini": 1176,
    "gemma-2b-it" : 990,
    "gemma-7b-it" : 1038,
    "gemma-1.1-7b-it" : 1084,
    "gemma-2-27b-it" : 1219,
    "google-gemma-2-9b-it" : 1189,
    "gemma-2-2b-it": 1138,
    "gemma-1.1-2b-it" : 1021,
    "yi-34b-chat" : 1111,
    "yi-1.5-34b-chat" : 1157,
    "mistral-7b-instruct-1" : 1008,
    "mistral-7b-instruct-2" : 1072,
    "mistral-8x7b-instruct-v0.1" : 1114,
    "meta-llama-3.1-8b-instruct" : 1173,
    "llama2-13b-chat" : 1063,
    "llama3-8b-instruct" : 1152,
    "llama-2-7b-chat" : 1037,
    "meta-llama-3.1-70b-instruct" : 1247,
    "llama-3-70b-instruct" : 1206,
    "command-r-(04-2024)" : 1149,
    "command-r-(08-2024)" : 1179,
    "qwen1.5-4B-chat" : 989,
    "qwen1.5-14b-chat" : 1109,
    "qwen1.5-32b-chat" : 1125,
    "qwen1.5-72b-chat" : 1148,
    "qwen2-72b-instruct": 1187,
    "openchat-3.5" : 1077,
    "openchat-3.5-0106" : 1092,
    "vicuna-13b" : 1042,
    "vicuna-33b" : 1091,
    "vicuna-7b" : 1005,
    "zephyr-7b-alpha" : 1041,
    "zephyr-7b-beta" : 1053,
    "tulu-2-dpo-70b" : 1099,
    "starling-lm-7b-alpha" : 1088,
    "starling-lm-7b-beta" : 1119,
    "codellama-34b-instruct" : 1043,
    "gemma-2-9b-it-simpo" : 1216,
    "openassistant-pythia-12b" : 893,
    "gemini-1.5-flash-001" : 1227,
    "koala-13b" : 965,
    'gpt-4-turbo-2024-04-09' :  1257,
    "llama2-7b-chat" : 1037,
    'gemini-1.5-pro-001' : 1260,
    'gemini-1.0-pro-001' : 1132,
    'qwen1.5-4b-chat' : 989,
    'llama-3.2-3b-it' : 1103,
    'yi-lightning' : 1287,
    'glm-4-plus' : 1274,
    'nemotron-70b' : 0,
    "claude-3.5-sonnet-20241022": 0,
    'qwen2.5-1.5b' : 0,
    'smollm2-1.7b' : 0,
    'llama-3.2-1b-it' : 0,
    'ministral-8b-it' : 0,
    'llama-3.1-tulu-3-8b' : 0,
    'llama-3.1-tulu-3-70b' : 0,
    'qwq-32b-preview' : 0,
    'meta-llama-3.3-70b-instruct' : 0,
}
