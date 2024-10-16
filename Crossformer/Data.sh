'''
-------------------------------------------------------------------------------------------------
Run the following code to train and test:
python main_crossformer.py --data Data --in_len 64 --out_len 4 --seg_len 6 --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.9e-5 --itr 5 --batch_size 256
-------------------------------------------------------------------------------------------------

-------------------------------------------------------------------------------------------------
Run the following code to evaluate:
python eval_crossformer.py --checkpoint_root ./checkpoints --setting_name Crossformer_Data__d_model512__n_heads16__d_ff1024__batch_size128
-------------------------------------------------------------------------------------------------
'''

