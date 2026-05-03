from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pydantic import BaseModel

class Item(BaseModel):
    theme: str
    meter: str
    qafiyah: str

app = FastAPI()
# Load model directly

tokenizer = AutoTokenizer.from_pretrained("../models/generation_models/ashaar_finetuned_final_8")
model = AutoModelForCausalLM.from_pretrained("../models/generation_models/ashaar_finetuned_final_8")
theme_to_token = json.load(open("../pre_trained_models/ashaar/tokens/theme_tokens.json", "r"))
token_to_theme = {t:m for m,t in theme_to_token.items()}
meter_to_token = json.load(open("../pre_trained_models/ashaar/tokens/meter_tokens.json", "r"))
token_to_meter = {t:m for m,t in meter_to_token.items()}

@app.post("/generate/")
async def read_root(item: Item):
    # theme = "قصيدة غزل"
    # meter = "الطويل"
    # qafiyah = "ر"
    theme = item.theme
    meter = item.meter
    qafiyah = item.qafiyah
    print(theme)
    prompt = f"{meter_to_token[meter]} {qafiyah} {theme_to_token[theme]}"
    # print(prompt)
    encoded_input = tokenizer(prompt, return_tensors='pt')
    output = model.generate(**encoded_input, max_length = 156,temperature=0.5, top_p = 3, do_sample=True)
    # print(encoded_input)
    result = ""
    prev_token = ""

    for i, beam in enumerate(output[:, len(encoded_input.input_ids[0]):]):
        for token in beam:
            # print(decoded)
            decoded = tokenizer.decode(token)
            if 'meter' in decoded or 'theme' in decoded:
                break
            if decoded == "<|endoftext|>":
                break
            if decoded == "<|vsep|>":
                result += "\n\n"
            elif decoded in ["<|bsep|>", "</|bsep|>"]:
                result += "\n"
            elif decoded in ['<|psep|>', '</|psep|>']:
                pass
            else:
                result += decoded
            prev_token = decoded
        else:
            break
    # print(theme+" "+ f"من بحر {meter} مع قافية بحر ({qafiyah})" + "\n" +result)
    return {"output": str(result)}