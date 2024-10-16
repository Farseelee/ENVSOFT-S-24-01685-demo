############################predict length 48####################################
python main_crossformer.py --data Data \
--in_len 168 --out_len 24 --seg_len 6 \
--d_model 64 --d_ff 128 --n_heads 2 \
--learning_rate 5e-4  --lradj fixed --itr 5


'''
-------------------------------------------------------------------------------------------------
训练、测试运行：
python main_crossformer.py --data Data --in_len 64 --out_len 4 --seg_len 6 --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.9e-5 --itr 5 --batch_size 128
-------------------------------------------------------------------------------------------------
Epoch: 10, Steps: 77 | Train Loss: 0.2293403 Vali Loss: 0.0439593 Test Loss: 0.0947184
EarlyStopping counter: 2 out of 3
Updating learning rate to 3.125e-07
Epoch: 11 cost time: 28.19685411453247
Epoch: 11, Steps: 77 | Train Loss: 0.2267093 Vali Loss: 0.0440189 Test Loss: 0.0950091
EarlyStopping counter: 3 out of 3
Early stopping
>>>>>>>testing : Crossformer_Data__d_model512__n_heads16__d_ff1024__batch_size128<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
test 2828
mse:0.08872933685779572, mae:0.1640925258398056, sse:0.08872933685779572, sst:0.9953812956809998, r2_score:0.9108589440584183

-------------------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------------------
评估运行：
python eval_crossformer.py --checkpoint_root ./checkpoints --setting_name Crossformer_Data__d_model512__n_heads16__d_ff1024__batch_size128
-------------------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------------------
sobol指数
-------------------------------------------------------------------------------------------------
python eval_crossformer.py --checkpoint_root ./checkpoints --setting_name Crossformer_Data__d_model512__n_heads16__d_ff1024__batch_size32__learning_rate1e-06



Crossformer_Data__d_model512__n_heads4__d_ff1024__batch_size128__learning_rate1.1843488737836e-05

-------------------------------------------------------------------------------------------------
参数space：
-------------------------------------------------------------------------------------------------
d_model:[128, 256, 512, 1024]
n_heads:[4, 8, 16, 32]
learning_rate:(1e-6, 1e-4, "log-uniform：参数的取值不是线性分布的，而是对数尺度上的均匀分布")
batch_size:[16, 32, 64, 128, 256]
-------------------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------------------
高斯噪声
-------------------------------------------------------------------------------------------------
python main_crossformer.py --data Data_with_noise1 --root_path './noise datasets/' --data_path 'Data_with_noise1.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'
mse:0.09398263692855835, mae:0.16004441678524017, sse:0.09398263692855835, sst:1.087000846862793, r2_score:0.9135394990444183

python main_crossformer.py --data Data_with_noise2 --root_path './noise datasets/' --data_path 'Data_with_noise2.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'
mse:0.09643908590078354, mae:0.16799640655517578, sse:0.09643908590078354, sst:1.144734263420105, r2_score:0.9157541692256927

python main_crossformer.py --data Data_with_noise3 --root_path './noise datasets/' --data_path 'Data_with_noise3.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'
mse:0.09517811238765717, mae:0.16318349540233612, sse:0.09517811238765717, sst:1.1026290655136108, r2_score:0.9136807546019554

python main_crossformer.py --data Data_with_noise4 --root_path './noise datasets/' --data_path 'Data_with_noise4.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'
mse:0.09616875648498535, mae:0.1622229516506195, sse:0.09616875648498535, sst:1.0924484729766846, r2_score:0.9119695276021957

python main_crossformer.py --data Data_with_noise5 --root_path './noise datasets/' --data_path 'Data_with_noise5.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'
mse:0.0969153344631195, mae:0.16632592678070068, sse:0.0969153344631195, sst:1.1280932426452637, r2_score:0.9140892550349236

python main_crossformer.py --data Data_with_noise6 --root_path './noise datasets/' --data_path 'Data_with_noise6.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'


python main_crossformer.py --data Data_with_noise7 --root_path './noise datasets/' --data_path 'Data_with_noise7.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'


python main_crossformer.py --data Data_with_noise8 --root_path './noise datasets/' --data_path 'Data_with_noise8.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'


python main_crossformer.py --data Data_with_noise9 --root_path './noise datasets/' --data_path 'Data_with_noise9.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'


python main_crossformer.py --data Data_with_noise10 --root_path './noise datasets/' --data_path 'Data_with_noise10.csv' --d_model 512 --d_ff 1024 --n_heads 16 --learning_rate 3.592e-5 --itr 1 --batch_size 128 --checkpoints './noise checkpoint/'


-------------------------------------------------------------------------------------------------


-------------------------------------------------------------------------------------------------
贝叶斯优化结果
-------------------------------------------------------------------------------------------------
最佳参数: [512, 16, 3.591913120571863e-05, 256]
最佳 R² 分数: 0.9309

python eval_crossformer.py --checkpoint_root ./checkpoints --setting_name Crossformer_Data__d_model512__n_heads16__d_ff1024__batch_size256__learning_rate3.591913120571863e-05

最优参数组合验证集评估结果：
mse:0.12118282914161682, mae:0.07425940781831741, rmse:0.18690542876720428
mape:0.35574430227279663, mspe:77.00701904296875, sse:0.07425940781831741
sst:1.0187511444091797, r2_score:0.9271074086427689
-------------------------------------------------------------------------------------------------

'''

