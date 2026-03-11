# ECE285
To run the code, you will need to use: 
uv run SRDiff_main/data_gen/CT.py --config SRDiff_main/configs/CT.yaml

uv run SRDiff_main/tasks/trainer.py --config ./SRDiff_main/configs/rrdb/CT_pretrain.yaml --exp_name rrdb_ctz

uv run SRDiff_main/tasks/trainer.py --config ./SRDiff_main/configs/diffsr_ct.yaml --infer --exp_name ctX





[DISCLAIMERS]
The file paths might be different, so you would need to change them to run the above code.
The UV file does not include PyTorch libraries due to dependency and compatibility issues with CUDA. Such PyTorch libraries are included through pip install using cu130, and not through UV.
