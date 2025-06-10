
@echo off
python train.py --root_path "E:\acmm_mpdd\MPDD-data\MPDD-Young\Training" ^
--json_path "E:\acmm_mpdd\MPDD-data\MPDD-Young\Training\labels\Training_Validation_files.json" ^
--test_root_path  "E:\acmm_mpdd\Testing" ^
--test_json_path "E:\acmm_mpdd\Testing\labels\Testing_files_modified.json" ^
--ps_feature_path "E:\acmm_mpdd\ours\descriptions_embeddings_with_ids_bge_zh_ds.npy" ^
--test_ps_feature_path "E:\acmm_mpdd\Testing\labels\descriptions_embeddings_with_ids_bge_zh_ds_test.npy" ^
--task_label_num 2 ^
--model "mamba"
pause
