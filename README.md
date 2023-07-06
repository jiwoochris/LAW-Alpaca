# LAW-Alpaca

AI 법률 어드바이저 모델 개발

## Pretrained model
[beomi/polyglot-ko-12.8b-safetensors](https://huggingface.co/beomi/polyglot-ko-12.8b-safetensors)

Stanford Alpaca 모델을 학습한 방식과 동일한 방식으로 학습을 진행한, 한국어를 이해하는 Alpaca 모델 KoAlpaca 사용

## Data
[생활 법령](https://www.easylaw.go.kr/CSP/Main.laf) 100문 100답 데이터 2,195개를 스크랩 하여 LLM 학습을 위한 대화 형식의 json 파일로 만들어놓았습니다.
![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/f9d81285-3a2f-445f-895e-f3f5c2ef9ee5)


[huggingface dataset](https://huggingface.co/datasets/juicyjung/easylaw_kr)에도 올려놓았습니다.

```python
from datasets import load_dataset

dataset = load_dataset("juicyjung/easylaw_kr")
```

## Copyright Policy

생활법령 데이터 저작권 정책 : https://www.easylaw.go.kr/CSP/InfoCopyright.laf

![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/7704897e-4775-401c-a075-526b9c9fd211)

## Resources

- **학습 코드 (Colab)**: 학습 코드는 [여기](https://colab.research.google.com/drive/1OjyOK1JGg10QKYjEWsHchX1CiWiEH_si?usp=sharing)에서 확인할 수 있습니다.
- **Model**: Hugging Face에도 [adapter_model](https://huggingface.co/juicyjung/ko_law_alpaca-12.8b) 파일 업로드 해놓았습니다.


## Usage

To use this model, you can load it using the Hugging Face Transformers library as follows:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

peft_model_id = "juicyjung/ko_law_alpaca-12.8b"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()
```

You can then use the model to generate responses to legal queries as follows:

```python
def gen(x):
    q = f"### 질문: {x}\n\n### 답변:"
    # print(q)
    gened = model.generate(
        **tokenizer(
            q, 
            return_tensors='pt', 
            return_token_type_ids=False
        ).to('cuda'), 
        max_new_tokens=50,
        early_stopping=True,
        do_sample=True,
        eos_token_id=2,
    )
    print(tokenizer.decode(gened[0]))


gen('월세방을 얻어 자취를 하던 중 군 입영통지서를 받았습니다. 아직 임대차 계약기간이 남았는데 보증금을 돌려받을 수 있을까요?')
```

## Contributions

이 프로젝트에 대한 Contributions는 언제나 환영입니다. 특히 데이터 힘들게 수집했으니 많은 후속 연구 부탁드립니다!!
문제가 발견되거나 제안사항이 있으면 이 repository에 issue를 열어 주세요.

이유는 잘 모르겠는데 모델 학습이 수월하게 되지 않습니다...

- **학습 코드 (Colab)**: 학습 코드는 [여기](https://colab.research.google.com/drive/1OjyOK1JGg10QKYjEWsHchX1CiWiEH_si?usp=sharing)에서 확인할 수 있습니다.

여기서 코드 보시고 수정해야할 부분 있으면 많은 조언 부탁드립니다 ㅠㅠ
