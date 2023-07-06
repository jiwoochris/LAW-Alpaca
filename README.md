![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/c1b4612c-7099-4cd7-8044-444f9f31c710)

(고수분들 조언 부탁드립니다 🙏 [Help Click](https://github.com/juicyjung/LAW-Alpaca#help))

# LAW-Alpaca

AI 법률 어드바이저 모델 개발

KoAlpaca 모델에 법률 데이터를 학습시켜 (LoRA finetuning) 법률 자문을 해줄 수 있는 언어모델을 개발한다.

![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/8f6550d7-35a3-45af-98c8-1704c530368d)
![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/3078da85-5eb0-4448-9d29-5f5baf1383bc)
![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/9897cbdc-f995-459d-b6a7-8f8a4dcced85)



## Pretrained model
[beomi/polyglot-ko-12.8b-safetensors](https://huggingface.co/beomi/polyglot-ko-12.8b-safetensors)

Stanford Alpaca 모델을 학습한 방식과 동일한 방식으로 학습을 진행한, 한국어를 이해하는 Alpaca 모델 KoAlpaca 사용

## Data
[생활 법령](https://www.easylaw.go.kr/CSP/Main.laf) 100문 100답 데이터 2,195개를 스크랩 하여 LLM 학습을 위한 대화 형식의 json 파일로 만들어놓았습니다.
![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/f9d81285-3a2f-445f-895e-f3f5c2ef9ee5)


[huggingface dataset](https://huggingface.co/datasets/juicyjung/easylaw_kr)에도 올려놓았습니다.

datasets library에서 이 dataset을 바로 불러올 수 있습니다 :

```python
from datasets import load_dataset

dataset = load_dataset("juicyjung/easylaw_kr")
```

## Copyright Policy

생활법령 데이터 저작권 정책 : https://www.easylaw.go.kr/CSP/InfoCopyright.laf

"누구에게나 개방되어있으며, 영리 목적을 포함하여 모든 자유로운 활동이 보장됩니다."

![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/7704897e-4775-401c-a075-526b9c9fd211)

## Resources

- **학습 코드 (Colab)**: 학습 코드는 [여기](https://colab.research.google.com/drive/1OjyOK1JGg10QKYjEWsHchX1CiWiEH_si?usp=sharing)에서 확인할 수 있습니다.
- **Model**: Hugging Face에도 [adapter_model](https://huggingface.co/juicyjung/ko_law_alpaca-12.8b) 파일 업로드 해놓았습니다.


## Usage

PEFT library에서 이 모델을 불러올 수 있습니다 :

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

법률 질문에 대한 답변을 생성하기 위해 다음과 같은 코드를 사용합니다 :

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

이 프로젝트에 대한 Contributions는 언제나 환영입니다. 특히 [데이터](https://github.com/juicyjung/LAW-Alpaca/blob/main/%EC%83%9D%ED%99%9C%EB%B2%95%EB%A0%B9.json) 힘들게 수집했으니 많은 후속 연구 부탁드립니다!!

문제가 발견되거나 제안사항이 있으면 이 repository에 issue를 열어 주세요.

### Help

이유는 잘 모르겠는데 모델 학습이 수월하게 되지 않습니다... 코랩 요금제도 너무 비싸서 계속 테스트 해보기도 버겁고 학부생 수준에서 쉽지 않네요...

1. Training Loss가 어느순간부터 0으로만 나오는 현상 (LoraConfig의 r을 16 이상으로 올리거나 TrainingArguments epoch을 6 이상으로 늘릴 때 주로 발생)
2. 이상한 말을 막 내뱉는 현상 (아래 예시)
![image](https://github.com/juicyjung/LAW-Alpaca/assets/83687471/c18ed7ce-4188-4ce3-b637-7d65d96c22fa)
- [생활법령.json](https://github.com/juicyjung/LAW-Alpaca/blob/main/%EC%83%9D%ED%99%9C%EB%B2%95%EB%A0%B9.json)에는 들어가있지 않지만 "안녕 너는 누구야?" -> "안녕하세요 저는 법률자문을 도와주는 ai 챗봇 LAW Alpaca라고 해요" 와 같은 데이터도 같이 학습시켰음에도 불구하고 다른 얘기를 한다... 학습이 된건지 만건지..
- 그래도 "아파트 아래층 사람이 발코니에서 담배를 피워 간접흡연으로 피해를 받고 있는데요. 세대 내부 발코니도 아파트 금연구역으로 지정할 수 있나요?"와 같은 법률 질문은 LoRA 학습하기 전과 비교해 학습 데이터와 비슷한 답변을 내놓긴 합니다.

**학습 코드 (Colab)**: 학습 코드는 [여기](https://colab.research.google.com/drive/1OjyOK1JGg10QKYjEWsHchX1CiWiEH_si?usp=sharing)에서 확인할 수 있습니다.

여기서 코드 보시고 수정해야할 부분 있으면 많은 조언 부탁드립니다 ㅠㅠ
