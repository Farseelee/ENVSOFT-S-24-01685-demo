import subprocess
import os
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args

# 定义超参数的范围
space = [
    Categorical([128, 256, 512, 1024], name='d_model'),
    Categorical([4, 8, 16, 32], name='n_heads'),
    Real(1e-6, 1e-4, "log-uniform", name='learning_rate'),
    Categorical([16, 32, 64, 128, 256], name='batch_size')
]

# 目标函数
@use_named_args(space)
def objective(**params):
    d_ff = 2 * params['d_model']
    command = [
        "python", "main_crossformer.py", "--data", "Data",
        "--in_len", "168", "--out_len", "24", "--seg_len", "6",
        "--d_model", str(params['d_model']), "--d_ff", str(d_ff),
        "--n_heads", str(params['n_heads']), "--learning_rate", str(params['learning_rate']),
        "--itr", "1", "--batch_size", str(params['batch_size'])
    ]
    print("Running command:", " ".join(command))

    # 运行命令并让输出直接显示在终端，同时捕获输出
    result = subprocess.run(command, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    if result.returncode != 0:
        print("Error running command:", result.stderr)
        return float('inf')  # 表示失败

    # 解析输出来获取 R² 分数
    output = result.stdout
    try:
        r2_line = next(line for line in output.splitlines() if "r2_score" in line)
        r2_score = float(r2_line.split('r2_score')[1].split(',')[0])
        print(f"R2 Score for current parameters: {r2_score}")
    except StopIteration:
        print("R2 Score not found for current parameters.")
    except Exception as e:
        print(f"An error occurred while parsing R2 score: {e}")

    return -r2_score  # 贝叶斯优化需要最小化目标函数

# 执行贝叶斯优化
result = gp_minimize(objective, space, n_calls=50, random_state=0)

# 打印最佳结果
print("最佳参数: %s" % result.x)
print("最佳 R² 分数: %.4f" % -result.fun)
