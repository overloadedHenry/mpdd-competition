@echo off
python eval.py --root_path "E:\acmm_mpdd\MPDD-data\MPDD-Young\Testing" ^
--json_path "E:\acmm_mpdd\MPDD-data\MPDD-Young\Testing\labels\Testing_files_modified.json" ^
--ps_feature_path "E:\acmm_mpdd\ours\descriptions_embeddings_with_ids_bge.npy" ^
--params_path "E:\acmm_mpdd\ours\model_params\Trans_mamba_class2.pt" ^
--task_label_num 2 
pause
