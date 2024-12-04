# Healthcare Medical Models

## Tutorial:

### BioMistral on Gaudi HPU
1. Clone optimum-habana repo: `git clone https://github.com/huggingface/optimum-habana.git`
2. Change directory: `cd optimum-habana/examples/text-generation`
3. Install dependency: `pip install -r requirements.txt`
Run: 
```
python run_generation.py --model_name_or_path BioMistral/BioMistral-7B --bf16 --use_hpu_graphs --use_kv_cache --batch_size 1 --max_new_tokens 128 --attn_softmax_bf16 --limit_hpu_graphs --reuse_cache --trim_logits
```

### BioMistral on CPU
Run: `python BioMistral_cpu.py` in this repo.

### CheXNet on Gaudi

1. Download data at https://nihcc.app.box.com/v/ChestXray-NIHCC
download and extract `images` folder / `Data_Entry_2017_v2020.csv` / `train_val_list.txt` / `test_list.txt` 
2: `mkdir output`
3: Start Inferencing or Training:
- Inference on Gaudi: `python chexnet.py --inference --iterations 1000 --data_dir images --output_dir output --hpu`
- Training(Finetune): `python chexnet.py --training --data_dir images`

### CheXNet on CPU
- Inference: Run `chexnet.py` in this repo. `python chexnet_cpu.py --inference --model_path chexnet_finetuned.pth --iterations 1000 --data_dir ./ --output_dir output `
- Finetune: `python chexnet.py --training --epochs 10 --data_dir ./ --output_dir output `
  
### Source
- Medical LLM - BioMistral: https://huggingface.co/BioMistral/BioMistral-7B
- Identify Chest disease - Chexnet: https://github.com/HabanaAI/Gaudi-solutions/blob/a823f23e49cf8d2d58b9c012cf911681d698f0cc/healthcare/chexnet.ipynb
