# Medical_Models

BioMistral source: https://huggingface.co/BioMistral/BioMistral-7B
Chexnet Source: https://github.com/HabanaAI/Gaudi-solutions/blob/a823f23e49cf8d2d58b9c012cf911681d698f0cc/healthcare/chexnet.ipynb

### Chexnet on Gaudi HPU

Source: https://github.com/dsmertin/Gaudi-solutions/blob/656b2ac81cf665b6cda9288d358f51eabfef36db/healthcare/chexnet.py

Step 1: download source file
Step 2: Download data at https://nihcc.app.box.com/v/ChestXray-NIHCC
download and extract `images` folder / `Data_Entry_2017_v2020.csv` / `train_val_list.txt` / `test_list.txt` 
Step 3: `mkdir output`

Step 4 Inference or Training:
- Inference: `python chexnet.py --inference --data_dir images --output_dir output --hpu`
- Training(Finetune): `python chexnet.py --training --data_dir images`