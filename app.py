import streamlit as st 
from datetime import datetime
import pandas as pd
import numpy as np
import tushare as ts
import pickle
import io
from itertools import product
import torch
import torch.nn as nn
import os
from models import set_seed
from preprocess import preprocess_data
from train import train_model
from predict import predict_new_data
from tushare_function import read_day_from_tushare, select_time
from plot_candlestick import plot_candlestick
from incremental_train import incremental_train_for_label
from ai import get_generated_code
from CSS import inject_orientation_script,load_custom_css
import re
import json
import time
# 设置随机种子
set_seed(42)

# 修改页面配置
st.set_page_config(
    page_title="东吴秀享AI超额收益系统", 
    layout="wide",
    initial_sidebar_state="auto"
)

# -------------------- 初始化 session_state -------------------- #
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'best_models' not in st.session_state:
    st.session_state.best_models = None

if 'peak_models_list' not in st.session_state:
    st.session_state.peak_models_list = []
if 'trough_models_list' not in st.session_state:
    st.session_state.trough_models_list = []

if 'train_df_preprocessed' not in st.session_state:
    st.session_state.train_df_preprocessed = None
if 'train_all_features' not in st.session_state:
    st.session_state.train_all_features = None

# 预测 / 回测 结果（未模型微调）
if 'final_result' not in st.session_state:
    st.session_state.final_result = None
if 'final_bt' not in st.session_state:
    st.session_state.final_bt = {}

# ★ 新增：模型微调后的预测 / 回测结果，用于对比
if 'inc_final_result' not in st.session_state:
    st.session_state.inc_final_result = None
if 'inc_final_bt' not in st.session_state:
    st.session_state.inc_final_bt = {}

# ★ 新增：存储预测集原始 DataFrame（模型微调后需要再次预测）
if 'new_df_raw' not in st.session_state:
    st.session_state.new_df_raw = None



def main_product():
    inject_orientation_script()
    st.title("东吴秀享AI超额收益系统")

    # ========== 侧边栏参数设置 ========== 
    with st.sidebar:
        st.header("参数设置")
        with st.expander("数据设置", expanded=True):
            data_source = st.selectbox("选择数据来源", ["指数", "股票"])
            symbol_code = st.text_input(f"{data_source}代码", "000001.SH")
            N = st.number_input("窗口长度 N", min_value=5, max_value=100, value=30)
        with st.expander("模型设置", expanded=True):
            classifier_name = st.selectbox("选择模型", ["Transformer", "深度学习"], index=1)
            if classifier_name == "深度学习":
                classifier_name = "MLP"
            mixture_depth = st.slider("因子混合深度", 1, 3, 1)
            oversample_method = st.selectbox(
                "类别不均衡处理", 
                ["过采样", "类别权重", 'ADASYN', 'Borderline-SMOTE', 'SMOTEENN', 'SMOTETomek',"时间感知过采样"]
            )
            if oversample_method == "过采样":
                oversample_method = "SMOTE"
            if oversample_method == "类别权重":
                oversample_method = "Class Weights"
            if oversample_method == "时间感知过采样":
                oversample_method = "Time-Aware"
            use_best_combo = True
        with st.expander("特征设置", expanded=True):
            auto_feature = st.checkbox("自动特征选择", True)
            n_features_selected = st.number_input(
                "选择特征数量", 
                min_value=5, max_value=100, value=20, 
                disabled=auto_feature
            )

    load_custom_css()

    # ========== 四个选项卡 ========== 
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["训练模型", "预测", "模型微调", "上传模型预测", "因子管理"])
    # =======================================
    #    Tab1: 训练模型
    # =======================================
    with tab1:
        st.subheader("训练参数")
        col1, col2 = st.columns(2)
        with col1:
            train_start = st.date_input("训练开始日期", datetime(2000, 1, 1))
        with col2:
            train_end = st.date_input("训练结束日期", datetime(2020, 12, 31))

        num_rounds = 10  # 这里写死为 10 轮
        if st.button("开始训练"):
            try:
                with st.spinner("数据预处理中..."):
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data = read_day_from_tushare(symbol_code, symbol_type)
                    
                    raw_data, all_features_train = preprocess_data(
                        raw_data, N, mixture_depth, mark_labels=True
                    )
                    df_preprocessed_train = select_time(raw_data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
                with st.spinner(f"开始多轮训练，共 {num_rounds} 次..."):
                    st.session_state.peak_models_list.clear()
                    st.session_state.trough_models_list.clear()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i in range(num_rounds):
                        progress_val = (i + 1) / num_rounds
                        status_text.text(f"正在训练第 {i+1}/{num_rounds} 个模型...")
                        progress_bar.progress(progress_val)

                        (peak_model, peak_scaler, peak_selector, peak_selected_features,
                         all_features_peak, peak_best_score, peak_metrics, peak_threshold,
                         trough_model, trough_scaler, trough_selector, trough_selected_features,
                         all_features_trough, trough_best_score, trough_metrics, trough_threshold
                        ) = train_model(
                            df_preprocessed_train,
                            N,
                            all_features_train,
                            classifier_name,
                            mixture_depth,
                            n_features_selected if not auto_feature else 'auto',
                            oversample_method
                        )
                        st.session_state.peak_models_list.append(
                            (peak_model, peak_scaler, peak_selector, peak_selected_features, peak_threshold)
                        )
                        st.session_state.trough_models_list.append(
                            (trough_model, trough_scaler, trough_selector, trough_selected_features, trough_threshold)
                        )

                    progress_bar.progress(1.0)
                    status_text.text("多轮训练完成！")

                # 将最后一次训练的模型存入 session_state
                st.session_state.models = {
                    'peak_model': peak_model,
                    'peak_scaler': peak_scaler,
                    'peak_selector': peak_selector,
                    'peak_selected_features': peak_selected_features,
                    'peak_threshold': peak_threshold,
                    'trough_model': trough_model,
                    'trough_scaler': trough_scaler,
                    'trough_selector': trough_selector,
                    'trough_selected_features': trough_selected_features,
                    'trough_threshold': trough_threshold,
                    'N': N,
                    'mixture_depth': mixture_depth
                }
                st.session_state.train_df_preprocessed = df_preprocessed_train
                st.session_state.train_all_features = all_features_train
                st.session_state.trained = True

                st.success(f"多轮训练全部完成！共训练 {num_rounds} 组峰/谷模型。")

                # 训练可视化
                peaks = df_preprocessed_train[df_preprocessed_train['Peak'] == 1]
                troughs = df_preprocessed_train[df_preprocessed_train['Trough'] == 1]
                fig = plot_candlestick(
                    df_preprocessed_train,
                    symbol_code,
                    train_start.strftime("%Y%m%d"),
                    train_end.strftime("%Y%m%d"),
                    peaks=peaks,
                    troughs=troughs
                )
                st.plotly_chart(fig, use_container_width=True, key="chart1")
            except Exception as e:
                st.error(f"训练失败: {str(e)}")

        # 训练集可视化（仅展示）
        try:
            st.markdown("<h2 style='font-size:20px;'>训练集可视化</h2>", unsafe_allow_html=True)
            symbol_type = 'index' if data_source == '指数' else 'stock'
            raw_data = read_day_from_tushare(symbol_code, symbol_type)
            
            raw_data, _ = preprocess_data(
                raw_data, N, mixture_depth, mark_labels=True
            )
            df_preprocessed_vis = select_time(raw_data, train_start.strftime("%Y%m%d"), train_end.strftime("%Y%m%d"))
            peaks_vis = df_preprocessed_vis[df_preprocessed_vis['Peak'] == 1]
            troughs_vis = df_preprocessed_vis[df_preprocessed_vis['Trough'] == 1]
            fig_vis = plot_candlestick(
                df_preprocessed_vis,
                symbol_code,
                train_start.strftime("%Y%m%d"),
                train_end.strftime("%Y%m%d"),
                peaks=peaks_vis,
                troughs=troughs_vis
            )
            st.plotly_chart(fig_vis, use_container_width=True, key="chart2")
        except Exception as e:
            st.warning(f"可视化失败: {e}")


    # =======================================
    #   Tab2: 预测 + 回测
    # =======================================
    with tab2:
        if not st.session_state.get('trained', False):
            st.warning("请先完成模型训练")
        else:
            st.subheader("预测参数")
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                pred_start = st.date_input("预测开始日期", datetime(2021, 1, 1))
            with col_date2:
                pred_end = st.date_input("预测结束日期", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab2")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab2"
                    )
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab2")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab2"
                    )
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal = st.checkbox("调整买卖信号", value=False, help="阳线买，阴线卖，高点需创X日新高", key="enable_change_signal_tab2")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab2"
                    )

            if st.button("开始预测"):
                try:
                    if st.session_state.train_df_preprocessed is None or st.session_state.train_all_features is None:
                        st.error("无法获取训练集数据，请先在Tab1完成训练。")
                        return

                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data = read_day_from_tushare(symbol_code, symbol_type)
                    raw_data, _ = preprocess_data(
                        raw_data, N, mixture_depth, mark_labels=False
                    )
                    new_df_raw = select_time(raw_data, pred_start.strftime("%Y%m%d"), pred_end.strftime("%Y%m%d"))

                    # ★ 存到 session_state，供模型微调使用
                    st.session_state.new_df_raw = new_df_raw

                    enable_chase_val = enable_chase
                    enable_stop_loss_val = enable_stop_loss
                    enable_change_signal_val = enable_change_signal
                    n_buy_val = n_buy
                    n_sell_val = n_sell
                    n_newhigh_val = n_newhigh

                    peak_models = st.session_state.peak_models_list
                    trough_models = st.session_state.trough_models_list

                    best_excess = -np.inf
                    best_models = None
                    final_result, final_bt, final_trades_df = None, {}, pd.DataFrame()
                    use_best_combo_val = use_best_combo

                    # ---------- 多组合搜索 ----------
                    if use_best_combo_val:
                        model_combinations = list(product(peak_models, trough_models))
                        total_combos = len(model_combinations)
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for idx, (peak_m, trough_m) in enumerate(model_combinations):
                            combo_progress = (idx + 1) / total_combos
                            status_text.text(f"正在进行第 {idx+1}/{total_combos} 轮预测...")
                            progress_bar.progress(combo_progress)

                            pm, ps, psel, pfeats, pth = peak_m
                            tm, ts, tsel, tfeats, tth = trough_m
                            try:
                                _, bt_result, _ = predict_new_data(
                                    new_df_raw,
                                    pm, ps, psel, pfeats, pth,
                                    tm, ts, tsel, tfeats, tth,
                                    st.session_state.models['N'],
                                    st.session_state.models['mixture_depth'],
                                    window_size=10,
                                    eval_mode=True,
                                    N_buy=n_buy_val,
                                    N_sell=n_sell_val,
                                    N_newhigh=n_newhigh_val,
                                    enable_chase=enable_chase_val,
                                    enable_stop_loss=enable_stop_loss_val,
                                    enable_change_signal=enable_change_signal_val,
                                )
                                current_excess = bt_result.get('超额收益率', -np.inf)
                                if current_excess > best_excess:
                                    best_excess = current_excess
                                    best_models = {
                                        'peak_model': pm,
                                        'peak_scaler': ps,
                                        'peak_selector': psel,
                                        'peak_selected_features': pfeats,
                                        'peak_threshold': pth,
                                        'trough_model': tm,
                                        'trough_scaler': ts,
                                        'trough_selector': tsel,
                                        'trough_selected_features': tfeats,
                                        'trough_threshold': tth
                                    }
                            except:
                                continue

                        progress_bar.empty()
                        status_text.empty()

                        if best_models is None:
                            raise ValueError("所有组合均测试失败，无法完成预测。")

                        final_result, final_bt, final_trades_df = predict_new_data(
                            new_df_raw,
                            best_models['peak_model'],
                            best_models['peak_scaler'],
                            best_models['peak_selector'],
                            best_models['peak_selected_features'],
                            best_models['peak_threshold'],
                            best_models['trough_model'],
                            best_models['trough_scaler'],
                            best_models['trough_selector'],
                            best_models['trough_selected_features'],
                            best_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！最佳模型超额收益率: {best_excess * 100:.2f}%")

                    # ---------- 单模型预测 ----------
                    else:
                        single_models = st.session_state.models
                        _, bt_result_temp, _ = predict_new_data(
                            new_df_raw,
                            single_models['peak_model'],
                            single_models['peak_scaler'],
                            single_models['peak_selector'],
                            single_models['peak_selected_features'],
                            single_models['peak_threshold'],
                            single_models['trough_model'],
                            single_models['trough_scaler'],
                            single_models['trough_selector'],
                            single_models['trough_selected_features'],
                            single_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=True,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        best_excess = bt_result_temp.get('超额收益率', -np.inf)
                        final_result, final_bt, final_trades_df = predict_new_data(
                            new_df_raw,
                            single_models['peak_model'],
                            single_models['peak_scaler'],
                            single_models['peak_selector'],
                            single_models['peak_selected_features'],
                            single_models['peak_threshold'],
                            single_models['trough_model'],
                            single_models['trough_scaler'],
                            single_models['trough_selector'],
                            single_models['trough_selected_features'],
                            single_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=n_buy_val,
                            N_sell=n_sell_val,
                            N_newhigh=n_newhigh_val,
                            enable_chase=enable_chase_val,
                            enable_stop_loss=enable_stop_loss_val,
                            enable_change_signal=enable_change_signal_val,
                        )
                        st.success(f"预测完成！(单模型) 超额收益率: {best_excess*100:.2f}%")

                    # ---------- 显示回测结果 ----------
                    st.subheader("回测结果")
                    metrics = [
                        ('累计收益率',   final_bt.get('累计收益率', 0)),
                        ('超额收益率',   final_bt.get('超额收益率', 0)),
                        ('胜率',         final_bt.get('胜率', 0)),
                        ('交易笔数',     final_bt.get('交易笔数', 0)),
                        ('最大回撤',     final_bt.get('最大回撤', 0)),
                        ('夏普比率',   '{:.4f}'.format(final_bt.get('年化夏普比率', 0)))
                    ]
                    first_line = metrics[:3]
                    cols_1 = st.columns(3)
                    for col, (name, value) in zip(cols_1, first_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")
                    second_line = metrics[3:]
                    cols_2 = st.columns(3)
                    for col, (name, value) in zip(cols_2, second_line):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    # ---------- 显示图表 ----------
                    peaks_pred = final_result[final_result['Peak_Prediction'] == 1]
                    troughs_pred = final_result[final_result['Trough_Prediction'] == 1]
                    fig = plot_candlestick(
                        final_result,
                        symbol_code,
                        pred_start.strftime("%Y%m%d"),
                        pred_end.strftime("%Y%m%d"),
                        peaks_pred,
                        troughs_pred,
                        prediction=True
                    )
                    st.plotly_chart(fig, use_container_width=True, key="chart3")

                    # ---------- 显示交易详情 ----------
                    col_left, col_right = st.columns(2)
                    final_result = final_result.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left:
                        st.subheader("预测明细")
                        st.dataframe(final_result[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    final_trades_df = final_trades_df.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df.empty:
                        final_trades_df['盈亏'] = final_trades_df['盈亏'] * 100
                        final_trades_df['买入日'] = final_trades_df['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df['卖出日'] = final_trades_df['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right:
                        st.subheader("交易记录")
                        if not final_trades_df.empty:
                            st.dataframe(
                                final_trades_df[['买入日', '买入原因', '买入价', '卖出日', '卖出原因', '卖出价', '持仓日', '盈亏']].style.format({'盈亏': '{:.2f}%'}))
                        else:
                            st.write("暂无交易记录")

                    # ---------- 保存到 session_state ----------
                    st.session_state.final_result = final_result
                    st.session_state.final_bt = final_bt
                    st.session_state.pred_start = pred_start
                    st.session_state.pred_end = pred_end
                    st.session_state.n_buy_val = n_buy_val
                    st.session_state.n_sell_val = n_sell_val
                    st.session_state.n_newhigh_val = n_newhigh_val
                    st.session_state.enable_chase_val = enable_chase_val
                    st.session_state.enable_stop_loss_val = enable_stop_loss_val
                    st.session_state.enable_change_signal_val = enable_change_signal_val

                except Exception as e:
                    st.error(f"预测失败: {str(e)}")


    # =======================================
    #   Tab3: 模型微调 （新标签页）
    # =======================================
    with tab3:
        st.subheader("模型微调（微调已有模型）")
        if st.session_state.final_result is None or st.session_state.new_df_raw is None:
            st.warning("请先在 [预测] 标签页完成一次预测，才能进行模型微调。")
        else:
            # 1) 模型微调日期（默认与预测区间一致）
            inc_col1, inc_col2 = st.columns(2)
            with inc_col1:
                inc_start_date = st.date_input(
                    "模型微调起始日期", 
                    st.session_state.get('pred_start', datetime(2021,1,1))
                )
            with inc_col2:
                inc_end_date = st.date_input(
                    "模型微调结束日期", 
                    st.session_state.get('pred_end', datetime.now())
                )

            # 2) 学习率选择
            lr_dict = {"极低 (1e-6)": 1e-6, "低 (1e-5)": 1e-5, "中 (1e-4)": 1e-4, "高 (1e-3)": 1e-3}
            lr_choice = st.selectbox("学习率", list(lr_dict.keys()), index=1)
            inc_lr = lr_dict[lr_choice]

            # 3) 训练轮数 (滑条, 默认值更少)
            inc_epochs = st.slider("最大训练轮数", 5, 100, 20)

            # 4) 设置冻结层选项
            if classifier_name == "MLP":
                freeze_options = {
                    "不冻结任何层": "none",
                    "只冻结第一层 (fc1)": "first_layer",
                    "只冻结第二层 (fc2)": "second_layer", 
                    "冻结所有层": "all",
                    "部分冻结第一层": "partial"
                }
            else:  # Transformer
                freeze_options = {
                    "不冻结任何层": "none",
                    "冻结输入层": "first_layer",
                    "冻结编码器层 (除最后一层)": "encoder_layers",
                    "冻结输出层": "output_layer",
                    "冻结所有层": "all"
                }
            
            freeze_choice = st.selectbox("冻结策略", list(freeze_options.keys()), index=0)
            freeze_option = freeze_options[freeze_choice]

            # 5) 是否启用混合训练 & 旧数据与新数据比例
            mix_enabled = st.checkbox("启用混合训练", value=True)
            inc_mix_ratio = 0.2  # 默认值稍高一点
            if mix_enabled:
                inc_mix_ratio = st.slider("旧数据与新数据比例", 0.1, 2.0, 0.2, step=0.1)

            # 6) 早停设置
            early_stopping = st.checkbox("启用早停", value=True)
            col_val1, col_val2 = st.columns(2)
            with col_val1:
                val_size = st.slider("验证集比例", 0.1, 0.5, 0.2, step=0.05, 
                                    disabled=not early_stopping)
            with col_val2:
                patience = st.slider("早停耐心值", 1, 10, 3, step=1,
                                disabled=not early_stopping)

            # 7) 点击按钮开始模型微调
            if st.button("执行模型微调"):
                try:
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data_full = read_day_from_tushare(symbol_code, symbol_type)

                    # ① 获取全量数据 + 自动打标签
                    df_preprocessed_all, _ = preprocess_data(
                        raw_data_full,
                        N,
                        mixture_depth,
                        mark_labels=True
                    )

                    # ② 截取微调区间
                    add_df = select_time(
                        df_preprocessed_all,
                        inc_start_date.strftime("%Y%m%d"),
                        inc_end_date.strftime("%Y%m%d")
                    )

                    # ③ 分别对 peak_model / trough_model 做 partial_fit，并显示训练进度
                    st.write("正在对峰模型进行微调训练...")
                    peak_prog = st.progress(0)
                    
                    # 微调峰模型
                    updated_peak_model, peak_val_acc, peak_epochs = incremental_train_for_label(
                        model=st.session_state.models['peak_model'],
                        scaler=st.session_state.models['peak_scaler'],
                        selected_features=st.session_state.models['peak_selected_features'],
                        df_new=add_df,  
                        label_column='Peak',
                        classifier_name=classifier_name,
                        window_size=10,
                        oversample_method=oversample_method,
                        new_lr=inc_lr,      
                        new_epochs=inc_epochs,      
                        freeze_option=freeze_option,
                        old_df=st.session_state.train_df_preprocessed if mix_enabled else None,
                        mix_ratio=inc_mix_ratio,
                        progress_bar=peak_prog,
                        early_stopping=early_stopping,
                        val_size=val_size,
                        patience=patience
                    )
                    
                    # 显示峰模型微调结果
                    st.success(f"峰模型微调完成! 最佳验证准确率: {peak_val_acc:.4f}，实际训练轮数: {peak_epochs}/{inc_epochs}")

                    st.write("正在对谷模型进行微调训练...")
                    trough_prog = st.progress(0)
                    
                    # 微调谷模型
                    updated_trough_model, trough_val_acc, trough_epochs = incremental_train_for_label(
                        model=st.session_state.models['trough_model'],
                        scaler=st.session_state.models['trough_scaler'],
                        selected_features=st.session_state.models['trough_selected_features'],
                        df_new=add_df,
                        label_column='Trough',
                        classifier_name=classifier_name,
                        window_size=10,
                        oversample_method=oversample_method,
                        new_lr=inc_lr,
                        new_epochs=inc_epochs,
                        freeze_option=freeze_option,
                        old_df=st.session_state.train_df_preprocessed if mix_enabled else None,
                        mix_ratio=inc_mix_ratio,
                        progress_bar=trough_prog,
                        early_stopping=early_stopping,
                        val_size=val_size,
                        patience=patience
                    )
                    
                    # 显示谷模型微调结果
                    st.success(f"谷模型微调完成! 最佳验证准确率: {trough_val_acc:.4f}，实际训练轮数: {trough_epochs}/{inc_epochs}")

                    # ④ 更新 session_state 中的模型
                    st.session_state.models['peak_model'] = updated_peak_model
                    st.session_state.models['trough_model'] = updated_trough_model

                    # 记录本次微调的参数
                    st.session_state.finetune_params = {
                        'lr': inc_lr,
                        'epochs': inc_epochs,
                        'freeze_option': freeze_option,
                        'mix_ratio': inc_mix_ratio if mix_enabled else 0,
                        'peak_val_acc': peak_val_acc,
                        'peak_epochs': peak_epochs,
                        'trough_val_acc': trough_val_acc,
                        'trough_epochs': trough_epochs
                    }

                    st.info("模型微调完成！下面对比微调前后的回测结果...")

                    # ⑤ 用微调后的模型再次预测 (针对 "预测" 时保存下来的 new_df_raw)
                    #    保持原先预测时的策略参数
                    refreshed_new_df = st.session_state.new_df_raw
                    if refreshed_new_df is None:
                        st.warning("未发现预测集数据，请先完成预测再查看对比结果。")
                        return

                    # ---- 重跑预测，得到模型微调后的结果 ----
                    if use_best_combo:
                        # 如果之前是多组合策略，这里同样把最优阈值、特征等拿来
                        best_models_inc = {
                            'peak_model': st.session_state.models['peak_model'],
                            'peak_scaler': st.session_state.models['peak_scaler'],
                            'peak_selector': st.session_state.models['peak_selector'],
                            'peak_selected_features': st.session_state.models['peak_selected_features'],
                            'peak_threshold': st.session_state.models['peak_threshold'],
                            'trough_model': st.session_state.models['trough_model'],
                            'trough_scaler': st.session_state.models['trough_scaler'],
                            'trough_selector': st.session_state.models['trough_selector'],
                            'trough_selected_features': st.session_state.models['trough_selected_features'],
                            'trough_threshold': st.session_state.models['trough_threshold']
                        }
                        inc_final_result, inc_final_bt, inc_final_trades_df = predict_new_data(
                            refreshed_new_df,
                            best_models_inc['peak_model'],
                            best_models_inc['peak_scaler'],
                            best_models_inc['peak_selector'],
                            best_models_inc['peak_selected_features'],
                            best_models_inc['peak_threshold'],
                            best_models_inc['trough_model'],
                            best_models_inc['trough_scaler'],
                            best_models_inc['trough_selector'],
                            best_models_inc['trough_selected_features'],
                            best_models_inc['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=st.session_state.n_buy_val,
                            N_sell=st.session_state.n_sell_val,
                            N_newhigh=st.session_state.n_newhigh_val,
                            enable_chase=st.session_state.enable_chase_val,
                            enable_stop_loss=st.session_state.enable_stop_loss_val,
                            enable_change_signal=st.session_state.enable_change_signal_val,
                        )
                    else:
                        single_models = st.session_state.models
                        inc_final_result, inc_final_bt, inc_final_trades_df = predict_new_data(
                            refreshed_new_df,
                            single_models['peak_model'],
                            single_models['peak_scaler'],
                            single_models['peak_selector'],
                            single_models['peak_selected_features'],
                            single_models['peak_threshold'],
                            single_models['trough_model'],
                            single_models['trough_scaler'],
                            single_models['trough_selector'],
                            single_models['trough_selected_features'],
                            single_models['trough_threshold'],
                            st.session_state.models['N'],
                            st.session_state.models['mixture_depth'],
                            window_size=10,
                            eval_mode=False,
                            N_buy=st.session_state.n_buy_val,
                            N_sell=st.session_state.n_sell_val,
                            N_newhigh=st.session_state.n_newhigh_val,
                            enable_chase=st.session_state.enable_chase_val,
                            enable_stop_loss=st.session_state.enable_stop_loss_val,
                            enable_change_signal=st.session_state.enable_change_signal_val,
                        )

                    # ---- 保存微调结果到 session_state ----
                    st.session_state.inc_final_result = inc_final_result
                    st.session_state.inc_final_bt = inc_final_bt

                    # ---- 对比模型微调前后的回测 ----
                    st.markdown("### 对比：未模型微调 vs 模型微调后")
                    orig_bt = st.session_state.final_bt
                    inc_bt = st.session_state.inc_final_bt

                    # 原模型 vs 新模型 主要指标对比
                    col_before, col_after, col_diff = st.columns(3)
                    with col_before:
                        st.write("**微调前**")
                        st.metric("累计收益率", f"{orig_bt.get('累计收益率', 0)*100:.2f}%")
                        st.metric("超额收益率", f"{orig_bt.get('超额收益率', 0)*100:.2f}%")
                        st.metric("胜率", f"{orig_bt.get('胜率', 0)*100:.2f}%")
                        st.metric("最大回撤", f"{orig_bt.get('最大回撤', 0)*100:.2f}%")
                        st.metric("交易笔数", f"{orig_bt.get('交易笔数', 0)}")
                    
                    with col_after:
                        st.write("**微调后**")
                        st.metric("累计收益率", f"{inc_bt.get('累计收益率', 0)*100:.2f}%")
                        st.metric("超额收益率", f"{inc_bt.get('超额收益率', 0)*100:.2f}%")
                        st.metric("胜率", f"{inc_bt.get('胜率', 0)*100:.2f}%")
                        st.metric("最大回撤", f"{inc_bt.get('最大回撤', 0)*100:.2f}%")
                        st.metric("交易笔数", f"{inc_bt.get('交易笔数', 0)}")
                    
                    with col_diff:
                        st.write("**变化量**")
                        st.metric("累计收益率变化", 
                                f"{(inc_bt.get('累计收益率', 0) - orig_bt.get('累计收益率', 0))*100:.2f}%",
                                delta_color="normal")
                        st.metric("超额收益率变化", 
                                f"{(inc_bt.get('超额收益率', 0) - orig_bt.get('超额收益率', 0))*100:.2f}%",
                                delta_color="normal")
                        st.metric("胜率变化", 
                                f"{(inc_bt.get('胜率', 0) - orig_bt.get('胜率', 0))*100:.2f}%",
                                delta_color="normal")
                        st.metric("最大回撤变化", 
                                f"{(inc_bt.get('最大回撤', 0) - orig_bt.get('最大回撤', 0))*100:.2f}%",
                                delta_color="inverse")  # 回撤是越小越好
                        st.metric("交易笔数变化", 
                                f"{inc_bt.get('交易笔数', 0) - orig_bt.get('交易笔数', 0)}",
                                delta_color="normal")

                    # 微调参数摘要
                    st.subheader("本次微调参数摘要")
                    ft_params = st.session_state.finetune_params
                    params_df = pd.DataFrame({
                        '参数': ['学习率', '最大训练轮数', '冻结策略', '混合数据比例', 
                            '峰模型验证准确率', '峰模型实际训练轮数',
                            '谷模型验证准确率', '谷模型实际训练轮数'],
                        '值': [
                            f"{ft_params['lr']:.1e}", 
                            str(ft_params['epochs']),
                            freeze_choice,
                            f"{ft_params['mix_ratio']:.1f}",
                            f"{ft_params['peak_val_acc']:.4f}",
                            f"{ft_params['peak_epochs']}/{ft_params['epochs']}",
                            f"{ft_params['trough_val_acc']:.4f}",
                            f"{ft_params['trough_epochs']}/{ft_params['epochs']}"
                        ]
                    })
                    st.dataframe(params_df, use_container_width=True)

                    st.subheader("模型微调后图表")
                    peaks_pred_inc = inc_final_result[inc_final_result['Peak_Prediction'] == 1]
                    troughs_pred_inc = inc_final_result[inc_final_result['Trough_Prediction'] == 1]
                    fig_updated = plot_candlestick(
                        inc_final_result,
                        symbol_code,
                        st.session_state.pred_start.strftime("%Y%m%d"),
                        st.session_state.pred_end.strftime("%Y%m%d"),
                        peaks_pred_inc,
                        troughs_pred_inc,
                        prediction=True
                    )
                    st.plotly_chart(fig_updated, use_container_width=True, key="chart_updated")

                except Exception as e:
                    st.error(f"模型微调过程中出错: {str(e)}")
                    st.exception(e) 


    # =======================================
    #   Tab4: 上传模型文件，独立预测
    # =======================================
    with tab4:
        st.subheader("上传模型文件（.pkl）并预测")
        st.markdown("在此页面可以上传之前已保存的最佳模型或单模型文件，直接进行预测。")
        uploaded_file = st.file_uploader("选择本地模型文件进行预测：", type=["pkl"])
        if uploaded_file is not None:
            with st.spinner("正在加载模型..."):
                best_models_loaded = pickle.load(uploaded_file)
                st.session_state.best_models = best_models_loaded
                st.session_state.trained = True
            st.success("已成功加载本地模型，可进行预测！")

        if not st.session_state.trained or (st.session_state.best_models is None):
            st.warning("请先上传模型文件，或前往其他页面训练并保存模型。")
        else:
            st.markdown("### 预测参数")
            col_date1_up, col_date2_up = st.columns(2)
            with col_date1_up:
                pred_start_up = st.date_input("预测开始日期(上传模型Tab)", datetime(2021, 1, 1))
            with col_date2_up:
                pred_end_up = st.date_input("预测结束日期(上传模型Tab)", datetime.now())

            with st.expander("策略选择", expanded=False):
                load_custom_css()
                strategy_row1 = st.columns([2, 2, 5])
                with strategy_row1[0]:
                    enable_chase_up = st.checkbox("启用追涨策略", value=False, help="卖出多少天后启用追涨", key="enable_chase_tab4")
                with strategy_row1[1]:
                    st.markdown('<div class="strategy-label">追涨长度</div>', unsafe_allow_html=True)
                with strategy_row1[2]:
                    n_buy_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_chase_up),
                        help="卖出多少天后启用追涨",
                        label_visibility="collapsed",
                        key="n_buy_tab4"
                    )
                strategy_row2 = st.columns([2, 2, 5])
                with strategy_row2[0]:
                    enable_stop_loss_up = st.checkbox("启用止损策略", value=False, help="持仓多少天后启用止损", key="enable_stop_loss_tab4")
                with strategy_row2[1]:
                    st.markdown('<div class="strategy-label">止损长度</div>', unsafe_allow_html=True)
                with strategy_row2[2]:
                    n_sell_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=60,
                        value=10,
                        disabled=(not enable_stop_loss_up),
                        help="持仓多少天后启用止损",
                        label_visibility="collapsed",
                        key="n_sell_tab4"
                    )
                strategy_row3 = st.columns([2, 2, 5])
                with strategy_row3[0]:
                    enable_change_signal_up = st.checkbox("调整买卖信号", value=False, help="高点需创X日新高", key="enable_change_signal_tab4")
                with strategy_row3[1]:
                    st.markdown('<div class="strategy-label">高点需创X日新高</div>', unsafe_allow_html=True)
                with strategy_row3[2]:
                    n_newhigh_up = st.number_input(
                        "",
                        min_value=1,
                        max_value=120,
                        value=60,
                        disabled=(not enable_change_signal_up),
                        help="要求价格在多少日内创出新高",
                        label_visibility="collapsed",
                        key="n_newhigh_tab4"
                    )

            # --------- 上传模型后的预测 --------- 
            if st.button("开始预测(上传模型Tab)"):
                try:
                    best_models = st.session_state.best_models
                    symbol_type = 'index' if data_source == '指数' else 'stock'
                    raw_data_up = read_day_from_tushare(symbol_code, symbol_type)
                    raw_data_up, _ = preprocess_data(
                        raw_data_up, N, mixture_depth, mark_labels=False
                    )
                    new_df_up = select_time(raw_data_up, pred_start_up.strftime("%Y%m%d"), pred_end_up.strftime("%Y%m%d"))

                    # 取出 N, mixture_depth，若之前训练时保存了这两个信息，可以直接读
                    N_val = st.session_state.models.get('N', 30)
                    mixture_val = st.session_state.models.get('mixture_depth', 1)

                    final_result_up, final_bt_up, final_trades_df_up = predict_new_data(
                        new_df_up,
                        best_models['peak_model'],
                        best_models['peak_scaler'],
                        best_models['peak_selector'],
                        best_models['peak_selected_features'],
                        best_models['peak_threshold'],
                        best_models['trough_model'],
                        best_models['trough_scaler'],
                        best_models['trough_selector'],
                        best_models['trough_selected_features'],
                        best_models['trough_threshold'],
                        N_val,
                        mixture_val,
                        window_size=10,
                        eval_mode=False,
                        N_buy=n_buy_up,
                        N_sell=n_sell_up,
                        N_newhigh=n_newhigh_up,
                        enable_chase=enable_chase_up,
                        enable_stop_loss=enable_stop_loss_up,
                        enable_change_signal=enable_change_signal_up,
                    )
                    st.success("预测完成！（使用已上传模型）")

                    st.subheader("回测结果")
                    metrics_up = [
                        ('累计收益率',   final_bt_up.get('累计收益率', 0)),
                        ('超额收益率',   final_bt_up.get('超额收益率', 0)),
                        ('胜率',         final_bt_up.get('胜率', 0)),
                        ('交易笔数',     final_bt_up.get('交易笔数', 0)),
                        ('最大回撤',     final_bt_up.get('最大回撤', 0)),
                        ('夏普比率',     final_bt_up.get('年化夏普比率', 0)),
                    ]
                    first_line_up = metrics_up[:3]
                    cols_1_up = st.columns(3)
                    for col, (name, value) in zip(cols_1_up, first_line_up):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")
                    second_line_up = metrics_up[3:]
                    cols_2_up = st.columns(3)
                    for col, (name, value) in zip(cols_2_up, second_line_up):
                        if isinstance(value, float):
                            col.metric(name, f"{value*100:.2f}%")
                        else:
                            col.metric(name, f"{value}")

                    peaks_pred_up = final_result_up[final_result_up['Peak_Prediction'] == 1]
                    troughs_pred_up = final_result_up[final_result_up['Trough_Prediction'] == 1]
                    fig_up = plot_candlestick(
                        final_result_up,
                        symbol_code,
                        pred_start_up.strftime("%Y%m%d"),
                        pred_end_up.strftime("%Y%m%d"),
                        peaks_pred_up,
                        troughs_pred_up,
                        prediction=True
                    )
                    st.plotly_chart(fig_up, use_container_width=True, key="chart_upload_tab")

                    col_left_up, col_right_up = st.columns(2)
                    final_result_up = final_result_up.rename(columns={
                        'TradeDate': '交易日期',
                        'Peak_Prediction': '高点标注',
                        'Peak_Probability': '高点概率',
                        'Trough_Prediction': '低点标注',
                        'Trough_Probability': '低点概率'
                    })
                    with col_left_up:
                        st.subheader("预测明细")
                        st.dataframe(final_result_up[['交易日期', '高点标注', '高点概率', '低点标注', '低点概率']])

                    final_trades_df_up = final_trades_df_up.rename(columns={
                        "entry_date": '买入日',
                        "signal_type_buy": '买入原因',
                        "entry_price": '买入价',
                        "exit_date": '卖出日',
                        "signal_type_sell": '卖出原因',
                        "exit_price": '卖出价',
                        "hold_days": '持仓日',
                        "return": '盈亏'
                    })
                    if not final_trades_df_up.empty:
                        final_trades_df_up['盈亏'] = final_trades_df_up['盈亏'] * 100
                        final_trades_df_up['买入日'] = final_trades_df_up['买入日'].dt.strftime('%Y-%m-%d')
                        final_trades_df_up['卖出日'] = final_trades_df_up['卖出日'].dt.strftime('%Y-%m-%d')

                    with col_right_up:
                        st.subheader("交易记录")
                        if not final_trades_df_up.empty:
                            st.dataframe(
                                final_trades_df_up[['买入日', '买入原因', '买入价', '卖出日', '卖出原因', '卖出价', '持仓日', '盈亏']].style.format({'盈亏': '{:.2f}%'})
                            )
                        else:
                            st.write("暂无交易记录")
                except Exception as e:
                    st.error(f"预测失败: {str(e)}")
    # =======================================
    #   Tab5: 因子管理
    # =======================================
    # =======================================
    #   Tab5: 因子管理
    # =======================================
    # =======================================
#   Tab5: 因子管理
    # =======================================
    with tab5:
        st.title("因子管理系统")

        # ---------------- 常量 & 全局设置 ----------------
        USER_FACTOR_MAP_FILE = "user_factor_map.json"
        BASE_DIR = "user_functions"
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)

        # ==================== 1. 缓存示例：AI 调用等耗时操作 ====================
        @st.cache_data(show_spinner=False)
        def get_generated_code_cached(prompt_str: str) -> str:
            """
            如果 get_generated_code(...) 内部很耗时（如调用外部API），
            可以缓存其结果。仅当 prompt_str 不变时，才会走缓存。
            """
            return get_generated_code(prompt_str)

        # ==================== 2. 读/写 user_factor_map (不缓存) ====================
        def load_user_factor_map() -> dict:
            if not os.path.exists(USER_FACTOR_MAP_FILE):
                return {}
            with open(USER_FACTOR_MAP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)

        def save_user_factor_map(data: dict):
            with open(USER_FACTOR_MAP_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        # ==================== 3. 注册 & 展示 用户自定义因子 ====================
        def register_user_factor(user_id: str, factor_name: str, file_name: str, func_name: str):
            tmp_data = load_user_factor_map()  # 先读取最新数据
            if user_id not in tmp_data:
                tmp_data[user_id] = {}
            tmp_data[user_id][factor_name] = {
                "file_name": file_name,
                "func_name": func_name
            }
            save_user_factor_map(tmp_data)
            st.session_state.user_factor_map = tmp_data
            st.success(f"自定义因子 '{factor_name}' 已成功保存到文件 {file_name}。")

        def my_factors(user_id: str):
            st.markdown("### 我的自定义因子列表")
            data = st.session_state.user_factor_map.get(user_id, {})
            if data:
                for f_name, detail in data.items():
                    f_file = detail.get("file_name", "")
                    f_func = detail.get("func_name", "")
                    st.write(f"- **因子名称**: {f_name}, 文件: {f_file}, 函数: {f_func}")
            else:
                st.write("（暂无自定义因子）")

        # ==================== 4. 保存用户自定义因子（.py 文件） ====================
        def save_user_function(user_id: str, code_str: str):
            func_name_match = re.search(r'def\s+([a-zA-Z_]\w*)\s*\(', code_str)
            if not func_name_match:
                raise ValueError("无法从因子代码中解析出函数名，请检查生成的代码格式。")
            func_name = func_name_match.group(1)
            user_dir = os.path.join(BASE_DIR, user_id)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
            file_name = f"{func_name}_{time.strftime('%Y%m%d_%H%M%S')}.py"
            file_path = os.path.join(user_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code_str)
            return file_path, func_name

        # ==================== 5. 系统预置因子：合理分类 ====================
        system_factor_dict = {
            "均线指标": {
                'Close_MA5_Diff': '收盘价与5日均线差值',
                'MA5_MA20_Diff': '5日均线与20日均线差值',
                'MA_5': '5日均线',
                'MA_20': '20日均线',
                'MA_50': '50日均线',
                'MA_200': '200日均线',
                'EMA_5': '5日指数移动均线',
                'EMA_20': '20日指数移动均线',
                'Slope_MA5': '5日均线斜率'
            },
            "技术指标": {
                'RSI_Signal': 'RSI信号',
                'MACD_Diff': 'MACD差值',
                'Bollinger_Position': '布林带位置',
                'K_D_Diff': 'K线与D线差值',
                'CCI_20': '20周期CCI',
                'Williams_%R_14': '14周期威廉指标',
                'OBV': '能量潮',
                'VWAP': '成交量加权平均价',
                'ZScore_20': '20周期Z分数',
                'Plus_DI': '正方向指标',
                'Minus_DI': '负方向指标',
                'ADX_14': '14周期ADX',
                'Bollinger_Width': '布林带宽度',
                'MFI_14': '14周期MFI',
                'CMF_20': '20周期CMF',
                'TRIX_15': '15周期TRIX',
                'Ultimate_Osc': '终极振荡指标',
                'Chaikin_Osc': 'Chaikin振荡指标',
                'PPO': 'PPO指标',
                'DPO_20': '20周期DPO',
                'KST': 'KST指标',
                'KST_signal': 'KST信号',
                'KAMA_10': '10周期KAMA'
            },
            "价格与成交量": {
                'ConsecutiveUp': '连续上涨天数',
                'ConsecutiveDown': '连续下跌天数',
                'Cross_MA5_Count': '穿越5日均线次数',
                'Volume_Spike_Count': '成交量激增次数',
                'Close': '收盘价',
                'Pch': '价格变动',
                'Volume_Change': '成交量变化',
                'Price_Mean_Diff': '价格均值差异',
                'High_Mean_Diff': '最高价均值差异',
                'Low_Mean_Diff': '最低价均值差异'
            },
            "其他": {
                'one': '因子一'
            }
        }

        # ==================== 6. 分别勾选系统因子 & 自定义因子 ====================
        def factor_selection_system():
            if 'selected_system_factors' not in st.session_state:
                st.session_state.selected_system_factors = []
            new_selection = []
            # 按分类显示，每个类别以标题区分
            for category, factors in system_factor_dict.items():
                st.markdown(f"#### {category}")
                for factor_name, label in factors.items():
                    is_checked = factor_name in st.session_state.selected_system_factors
                    checked = st.checkbox(label, value=is_checked, key=f"sys_{factor_name}")
                    if checked:
                        new_selection.append(factor_name)
            st.session_state.selected_system_factors = new_selection
            return new_selection

        def factor_selection_custom(user_id: str):
            if 'selected_custom_factors' not in st.session_state:
                st.session_state.selected_custom_factors = []
            user_data = st.session_state.user_factor_map.get(user_id, {})
            new_selection = []
            for factor_name, detail in user_data.items():
                is_checked = factor_name in st.session_state.selected_custom_factors
                checked = st.checkbox(factor_name, value=is_checked, key=f"cust_{factor_name}")
                if checked:
                    new_selection.append(factor_name)
            st.session_state.selected_custom_factors = new_selection
            return new_selection

        # ==================== 7. 创建自定义因子 (AI 生成 or 用户粘贴) ====================
        def create_custom_factor():
            if 'generated_code' not in st.session_state:
                st.session_state.generated_code = None
            prompt_str = st.text_area("因子需求描述 / 或粘贴完整因子代码", "")
            if st.button("生成因子代码", key="generate_factor_code"):
                if prompt_str.strip():
                    st.info("AI 正在生成因子代码 (缓存示例) ...")
                    code_str = get_generated_code_cached(prompt_str)
                    if code_str:
                        st.session_state.generated_code = code_str
                        st.success("AI 生成的因子代码：")
                        st.code(code_str, language="python")
                    else:
                        st.error("AI 生成失败，请重试。")
                else:
                    st.warning("请输入描述后再生成。")
            if st.session_state.generated_code:
                user_factor_name = st.text_input("自定义因子名称 (可中文)", "")
                if st.button("保存自定义因子", key="save_custom_factor"):
                    if not user_factor_name.strip():
                        st.warning("请输入自定义因子名称。")
                    else:
                        try:
                            file_path, func_name = save_user_function("user_123", st.session_state.generated_code)
                            register_user_factor(
                                user_id="user_123",
                                factor_name=user_factor_name,
                                file_name=os.path.basename(file_path),
                                func_name=func_name
                            )
                            st.info("更新后的因子列表：")
                            my_factors("user_123")
                        except Exception as e:
                            st.error(f"保存失败: {e}")

        # ==================== 8. 页面初始化: 读 JSON -> 存 session_state ====================
        if 'user_factor_map' not in st.session_state:
            st.session_state.user_factor_map = load_user_factor_map()
        user_id = "user_123"
        if user_id not in st.session_state.user_factor_map:
            st.session_state.user_factor_map[user_id] = {}

        # ==================== 9. 页面布局：使用垂直结构的 Expander ====================
        with st.expander("选择系统因子", expanded=False):
            selected_system = factor_selection_system()
        with st.expander("选择自定义因子", expanded=False):
            selected_custom = factor_selection_custom(user_id)
        with st.expander("创建自定义因子", expanded=False):
            create_custom_factor()

        st.markdown("---")
        # ==================== 10. 展示当前自定义因子及选择结果 ====================
        my_factors(user_id)
        st.subheader("已选系统因子:")
        if selected_system:
            st.write(", ".join(selected_system))
        else:
            st.write("（暂无选择）")
        st.subheader("已选自定义因子:")
        custom_data = st.session_state.user_factor_map[user_id]
        if selected_custom:
            for fac in selected_custom:
                func_name = custom_data[fac].get("func_name", "")
                st.write(f"- {fac} (func: {func_name})")
        else:
            st.write("（暂无选择）")



if __name__ == "__main__":
    main_product()
