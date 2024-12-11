from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

device = torch.device("cpu")

def chex(msg):
    model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B",torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
    streamer = TextStreamer(tokenizer)
    messages = [
        {"role": "user", "content": msg}
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=200, streamer=streamer, do_sample=True)

    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0].split("[/INST]")[1].replace("</s>","")
