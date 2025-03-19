#ai.py
import time
import re
import os
import importlib.util
from openai import OpenAI

BASE_DIR = "user_functions"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

#############################################################################
# 1. 全局（或持久化）维护“用户 -> {因子名 -> 函数文件}”的映射字典
#############################################################################
# 可以根据你的实际需求把它做成数据库存储或别的方式，这里用内存演示
user_factor_map = {}  
# 结构示例：
# {
#    "user_123": {
#        "因子A": "generate_signal_20250318_101512.py",
#        "因子B": "generate_signal_20250318_101613.py"
#    },
#    "user_456": {
#        "my_factor_1": "generate_signal_20250318_102045.py"
#    }
# }

def register_user_factor(user_id: str, factor_name: str, func_file_name: str):
    """
    将当前用户的“因子名称”与“函数文件名”关联起来，保存到全局 user_factor_map 中。
    :param user_id:     用户 ID
    :param factor_name: 因子名称（字符串）
    :param func_file_name: 保存到本地时所用的 .py 文件名，比如 'generate_signal_20250318_101512.py'
    """
    if user_id not in user_factor_map:
        user_factor_map[user_id] = {}
    user_factor_map[user_id][factor_name] = func_file_name
    print(f"[register_user_factor] 用户 {user_id}, 因子 '{factor_name}' 映射到文件 '{func_file_name}'")

def list_factors_of_user(user_id: str):
    """
    列出某个用户已注册的所有因子名称。
    """
    if user_id not in user_factor_map:
        return []
    return list(user_factor_map[user_id].keys())

#############################################################################
# 2. 你的 AI 生成代码逻辑
#############################################################################

def get_generated_code(prompt_str, retries=3, delay=5):
    """
    让 AI 返回一个【带时间戳】的函数名，并生成相应代码。
    （AI 会在函数名的末尾加 YYYYMMDD_HHMMSS 时间戳）
    """
    system_instruction = "你是专业的证券分析师和机器学习专家。"

    user_prompt = f"""
    我需要基于日线行情构建量化交易数据特征用于机器学习。请生成一个 Python 函数，函数名可自行决定，但名称的末尾必须带上当前日期时间戳（格式形如 YYYYMMDD_HHMMSS）。
    函数接收一个 pandas DataFrame 参数 df（包含列：Open, High, Low, Close, Volume, Amount，索引为日期），
    请根据以下内容为df构建特征列：
    
    - {prompt_str}
    
    函数要求：
    1. 新增的特征列列名必须带上当前日期时间戳，使用当前时间戳写死，不在函数内更新；
    2. 对于逐行赋值，请使用 .at[] 或 .loc[]，禁止使用 .iloc[]；
    3. 函数执行完毕后返回修改后的 df；
    4. 仅允许使用 pandas 和 numpy 库及，不得使用其他库；
    5. 只需要返回纯代码，不需要额外解释。
    """

    try:
        client = OpenAI(
            api_key="sk-1e63e70de8e5442594186ee9cf8e9ee6", 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        for attempt in range(retries):
            try:
                print(f"请求生成代码，尝试 {attempt + 1}/{retries}...")
                completion = client.chat.completions.create(
                    model="deepseek-v3",  # 或者你所使用的模型
                    messages=[
                        {'role': 'system', 'content': system_instruction},
                        {'role': 'user', 'content': user_prompt}
                    ]
                )
                print("代码生成成功")
                code_str = completion.choices[0].message.content
                # 去除 Markdown 包裹
                code_str = code_str.strip("```python").strip("```").strip()
                return code_str  # 直接返回AI生成的代码
            except Exception as e:
                print(f"请求失败，错误信息: {e}")
                if attempt < retries - 1:
                    print(f"重试中...{attempt + 1}/{retries}")
                    time.sleep(delay)
                else:
                    print("已达到最大重试次数。")
                    return None
    except Exception as e:
        print(f"调用 OpenAI 失败：{e}")
        return None
def save_user_function(user_id: str, code_str: str):
    """
    将 AI 生成的函数（其函数名自带时间戳）存成 .py 文件。
    返回 (file_path, func_name)
    """
    # 正则匹配函数名
    func_name_match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', code_str)
    if not func_name_match:
        raise ValueError("无法从生成的代码中匹配到函数名，请检查 AI 的输出是否符合约定。")
    func_name = func_name_match.group(1)  # 提取函数名称
    
    # 以函数名作为文件名
    user_dir = os.path.join(BASE_DIR, str(user_id))
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    file_name = f"{func_name}.py"
    file_path = os.path.join(user_dir, file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code_str)
    
    print(f"[save_user_function] 已将 AI 生成的函数 {func_name} 保存至: {file_path}")
    return file_path, func_name

#############################################################################
# 3. 动态加载 & 多因子顺序调用
#############################################################################

def load_user_function(user_id: str, func_file_name: str):
    """
    根据文件名动态加载对应的 Python 模块，并返回其中的函数对象。
    默认函数名与文件名一致（去掉 .py 后缀）。
    """
    file_path = os.path.join(BASE_DIR, str(user_id), func_file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"无法找到文件: {file_path}")
    
    spec = importlib.util.spec_from_file_location("user_factor_mod", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 再次用正则或和你的存储规则去拿“def 函数名”，
    func_name = os.path.splitext(func_file_name)[0]  # 假定一致
    if not hasattr(module, func_name):
        raise AttributeError(f"模块中未找到函数 {func_name} 的定义。")

    return getattr(module, func_name)

def apply_factors_in_sequence(user_id: str, factor_names: list, df):
    """
    根据因子名称列表，依次加载并执行每个因子对应的函数。
    前一个函数输出的 df，作为后一个函数的输入 df。
    最后返回最终 df。
    """
    if user_id not in user_factor_map:
        raise ValueError(f"用户 {user_id} 在 user_factor_map 中不存在任何因子映射。")

    for factor_name in factor_names:
        if factor_name not in user_factor_map[user_id]:
            raise ValueError(f"用户 {user_id} 未注册因子 '{factor_name}'")

        func_file_name = user_factor_map[user_id][factor_name]
        func_obj = load_user_function(user_id, func_file_name)

        print(f"[apply_factors_in_sequence] 执行因子: {factor_name}, 对应函数文件: {func_file_name}")
        df = func_obj(df)  # 调用函数，得到更新后的 df

    return df

import numpy as np
import pandas as pd
import os
import tushare as ts

ts.set_token('c5c5700a6f4678a1837ad234f2e9ea2a573a26b914b47fa2dbb38aff')
pro = ts.pro_api()


#############################################################################
# 4. 测试流程示例
#############################################################################

if __name__ == "__main__":
    # ========== 1. 假设我们生成并注册因子 A ==========
    prompt_str = "MA5"

    code_result = get_generated_code(prompt_str)
    if code_result:
        file_path, func_name = save_user_function(user_id="12345", code_str=code_result)

        # 假设用户自定义的因子名称是 "因子A"
        factor_name = "因子A"
        file_name_only = os.path.basename(file_path)  # 只取文件名
        # 注册到全局映射
        register_user_factor(user_id="12345", factor_name=factor_name, func_file_name=file_name_only)

    # ========== 2. 假设再生成并注册因子 B ==========
    prompt_str = "MA10"

    code_result_2 = get_generated_code(prompt_str)
    if code_result_2:
        file_path_2, func_name_2 = save_user_function(user_id="12345", code_str=code_result_2)

        factor_name_2 = "因子B"
        file_name_only_2 = os.path.basename(file_path_2)
        register_user_factor(user_id="12345", factor_name=factor_name_2, func_file_name=file_name_only_2)

    # ========== 3. 依次执行多个因子 ==========
    import pandas as pd
    ts.set_token('c5c5700a6f4678a1837ad234f2e9ea2a573a26b914b47fa2dbb38aff')
    pro = ts.pro_api()
    def read_day_from_tushare(symbol_code, symbol_type='stock'):
        """
        使用 Tushare API 获取股票或指数的全部日线行情数据。
        参数:
        - symbol_code: 股票或指数代码 (如 "000001.SZ" 或 "000300.SH")
        - symbol_type: 'stock' 或 'index' (不区分大小写)
        返回:
        - 包含日期、开高低收、成交量等列的DataFrame
        """
        symbol_type = symbol_type.lower()
        print(f"传递给 read_day_from_tushare 的 symbol_type: {symbol_type} (类型: {type(symbol_type)})")  # 调试输出
        print(f"尝试通过 Tushare 获取{symbol_type}数据: {symbol_code}")
        
        # 添加断言，确保 symbol_type 是 'stock' 或 'index'
        assert symbol_type in ['stock', 'index'], "symbol_type 必须是 'stock' 或 'index'"
        
        try:
            if symbol_type == 'stock':
                # 获取股票日线数据
                df = pro.daily(ts_code=symbol_code, start_date='20000101', end_date='20251231')
                if df.empty:
                    print("Tushare 返回的股票数据为空。")
                    return pd.DataFrame()
                
                # 转换日期格式并排序
                df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                df = df.sort_values('date')
                
                # 重命名和选择需要的列
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'vol': 'Volume',
                    'amount': 'Amount',
                    'trade_date': 'TradeDate'
                })
                df.set_index('date', inplace=True)
                
                # 选择需要的列
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
                available_columns = [col for col in required_columns if col in df.columns]
                df = df[available_columns]
            
            elif symbol_type == 'index':
                # 获取指数日线数据，使用 index_daily 接口
                df = pro.index_daily(ts_code=symbol_code, start_date='20000101', end_date='20251231')
                if df.empty:
                    print("Tushare 返回的指数数据为空。")
                    return pd.DataFrame()
                
                # 转换日期格式并排序
                df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                df = df.sort_values('date')
                
                # 重命名和选择需要的列
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'vol': 'Volume',
                    'amount': 'Amount',
                    'trade_date': 'TradeDate'
                })
                df.set_index('date', inplace=True)
                
                # 选择需要的列，处理可能缺失的字段
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'TradeDate']
                available_columns = [col for col in required_columns if col in df.columns]
                df = df[available_columns]
            
            print(f"通过 Tushare 获取了 {len(df)} 条记录。")
            print(f"数据框的列：{df.columns.tolist()}")
            print(f"数据框前5行：\n{df.head()}")
            return df
        except AssertionError as ae:
            print(f"断言错误：{ae}")
            return pd.DataFrame()
        except Exception as e:
            print(f"通过 Tushare 获取数据失败：{e}")
            return pd.DataFrame()
    
    df_example = read_day_from_tushare(symbol_code='601555.SH', symbol_type='stock')

    # 拟执行顺序：先因子A，再因子B
    factor_chain = ["因子A", "因子B"]

    df_final = apply_factors_in_sequence(user_id="12345", factor_names=factor_chain, df=df_example)
    print("最终 df:\n", df_final)

    
