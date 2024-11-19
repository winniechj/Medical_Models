# Medical_Models

### Source

- Medical LLM - BioMistral: https://huggingface.co/BioMistral/BioMistral-7B

- Identify Chest disease - Chexnet: https://github.com/HabanaAI/Gaudi-solutions/blob/a823f23e49cf8d2d58b9c012cf911681d698f0cc/healthcare/chexnet.ipynb

### BioMistral on Gaudi HPU

Goto Optimum-Habana: https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation

Run: 
```
python run_generation.py --model_name_or_path BioMistral/BioMistral-7B --bf16 --use_hpu_graphs --use_kv_cache --batch_size 1 --max_new_tokens 128 --attn_softmax_bf16 --limit_hpu_graphs --reuse_cache --trim_logits
```

### BioMistral on CPU
Run: `python BioMistral_cpu.py`

### Chexnet

chexnet.py Source: https://github.com/dsmertin/Gaudi-solutions/blob/656b2ac81cf665b6cda9288d358f51eabfef36db/healthcare/chexnet.py

Step 1: download source file

Step 2: Download data at https://nihcc.app.box.com/v/ChestXray-NIHCC
download and extract `images` folder / `Data_Entry_2017_v2020.csv` / `train_val_list.txt` / `test_list.txt` 

Step 3: `mkdir output`

Step 4 Inference or Training:
- Inference on Gaudi: `python chexnet.py --inference --iterations 1000 --data_dir images --output_dir output --hpu`
- Inference on CPU: `python chexnet_cpu.py --inference --iterations 1000 --data_dir ./ --output_dir output `
- Training(Finetune): `python chexnet.py --training --data_dir images`
