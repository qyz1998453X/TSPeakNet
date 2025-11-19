#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析与可视化模块
读取原始数据并进行多维度分析
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os

class DataAnalyzer:
    """数据分析器"""
    
    def __init__(self, data_path="时序数据/原始数据.xlsx"):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            if os.path.exists(self.data_path):
                self.df = pd.read_excel(self.data_path)
                print(f"成功加载数据: {len(self.df)} 条记录")
            else:
                print(f"数据文件不存在: {self.data_path}")
                # 创建示例数据
                self.df = self.create_sample_data()
        except Exception as e:
            print(f"加载数据失败: {e}")
            self.df = self.create_sample_data()
    
    def create_sample_data(self):
        """创建示例数据"""
        import numpy as np
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        data = {
            '日期': dates,
            '地区': np.random.choice(['朝阳区', '海淀区', '昌平区', '顺义区'], len(dates)),
            '病虫害发生数': np.random.poisson(5, len(dates)),
            '温度': np.random.normal(20, 8, len(dates)),
            '湿度': np.random.normal(60, 15, len(dates)),
            '降雨量': np.random.exponential(5, len(dates))
        }
        return pd.DataFrame(data)
    
    def get_yearly_statistics(self):
        """逐年统计"""
        if self.df is None or len(self.df) == 0:
            return {}
        
        try:
            if '日期' in self.df.columns:
                self.df['年份'] = pd.to_datetime(self.df['日期']).dt.year
                yearly_stats = self.df.groupby('年份').agg({
                    '病虫害发生数': ['sum', 'mean', 'max']
                }).to_dict()
                return yearly_stats
            return {}
        except Exception as e:
            print(f"年度统计错误: {e}")
            return {}
    
    def get_monthly_statistics(self):
        """逐月统计"""
        if self.df is None or len(self.df) == 0:
            return {}
        
        try:
            if '日期' in self.df.columns:
                self.df['年月'] = pd.to_datetime(self.df['日期']).dt.to_period('M')
                monthly_stats = self.df.groupby('年月').agg({
                    '病虫害发生数': ['sum', 'mean']
                }).to_dict()
                return monthly_stats
            return {}
        except Exception as e:
            print(f"月度统计错误: {e}")
            return {}
    
    def get_regional_statistics(self):
        """按地区统计"""
        if self.df is None or len(self.df) == 0:
            return {}
        
        try:
            if '地区' in self.df.columns:
                regional_stats = self.df.groupby('地区').agg({
                    '病虫害发生数': ['sum', 'mean', 'max']
                }).to_dict()
                return regional_stats
            return {}
        except Exception as e:
            print(f"地区统计错误: {e}")
            return {}
    
    def create_yearly_chart(self):
        """创建年度趋势图"""
        if self.df is None or '日期' not in self.df.columns:
            return self.create_empty_chart("暂无数据")
        
        try:
            self.df['年份'] = pd.to_datetime(self.df['日期']).dt.year
            yearly = self.df.groupby('年份')['病虫害发生数'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=yearly['年份'],
                y=yearly['病虫害发生数'],
                marker_color='rgb(55, 83, 109)',
                name='病虫害发生数'
            ))
            
            fig.update_layout(
                title='病虫害发生数 - 年度趋势',
                xaxis_title='年份',
                yaxis_title='发生数',
                template='plotly_white',
                height=400
            )
            
            return fig.to_json()
        except Exception as e:
            return self.create_empty_chart(f"图表生成错误: {e}")
    
    def create_monthly_chart(self):
        """创建月度趋势图"""
        if self.df is None or '日期' not in self.df.columns:
            return self.create_empty_chart("暂无数据")
        
        try:
            self.df['年月'] = pd.to_datetime(self.df['日期']).dt.to_period('M').astype(str)
            monthly = self.df.groupby('年月')['病虫害发生数'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['年月'],
                y=monthly['病虫害发生数'],
                mode='lines+markers',
                marker_color='rgb(26, 118, 255)',
                line=dict(width=3),
                name='病虫害发生数'
            ))
            
            fig.update_layout(
                title='病虫害发生数 - 月度趋势',
                xaxis_title='月份',
                yaxis_title='发生数',
                template='plotly_white',
                height=400
            )
            
            return fig.to_json()
        except Exception as e:
            return self.create_empty_chart(f"图表生成错误: {e}")
    
    def create_regional_chart(self):
        """创建地区对比图"""
        if self.df is None or '地区' not in self.df.columns:
            return self.create_empty_chart("暂无数据")
        
        try:
            regional = self.df.groupby('地区')['病虫害发生数'].sum().reset_index()
            regional = regional.sort_values('病虫害发生数', ascending=False)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=regional['地区'],
                y=regional['病虫害发生数'],
                marker_color=px.colors.qualitative.Set3,
                name='病虫害发生数'
            ))
            
            fig.update_layout(
                title='病虫害发生数 - 地区分布',
                xaxis_title='地区',
                yaxis_title='发生数',
                template='plotly_white',
                height=400
            )
            
            return fig.to_json()
        except Exception as e:
            return self.create_empty_chart(f"图表生成错误: {e}")
    
    def create_weather_correlation_chart(self):
        """创建气象相关性图"""
        if self.df is None:
            return self.create_empty_chart("暂无数据")
        
        try:
            # 创建散点图矩阵
            weather_cols = ['温度', '湿度', '降雨量', '病虫害发生数']
            existing_cols = [col for col in weather_cols if col in self.df.columns]
            
            if len(existing_cols) >= 2:
                fig = px.scatter_matrix(
                    self.df[existing_cols],
                    dimensions=existing_cols,
                    title='气象因子与病虫害关系分析'
                )
                fig.update_layout(height=600, template='plotly_white')
                return fig.to_json()
            else:
                return self.create_empty_chart("气象数据不足")
        except Exception as e:
            return self.create_empty_chart(f"图表生成错误: {e}")
    
    def create_empty_chart(self, message="暂无数据"):
        """创建空图表"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        return fig.to_json()


class ModelResultAnalyzer:
    """模型预测结果分析器"""
    
    def __init__(self, model_dir="时序数据"):
        self.model_dir = model_dir
        self.models = self.get_model_files()
    
    def get_model_files(self):
        """获取所有模型文件"""
        models = []
        if os.path.exists(self.model_dir):
            for file in os.listdir(self.model_dir):
                if file.endswith('.xlsx') and '预测' in file and file != '原始数据.xlsx':
                    model_name = file.replace('-预测数据.xlsx', '').replace('-预测模型.xlsx', '')
                    models.append({
                        'name': model_name,
                        'file': file,
                        'path': os.path.join(self.model_dir, file)
                    })
        return models
    
    def load_model_data(self, model_name):
        """加载模型数据"""
        for model in self.models:
            if model['name'] == model_name:
                try:
                    df = pd.read_excel(model['path'])
                    return df
                except Exception as e:
                    print(f"加载模型数据失败: {e}")
                    return None
        return None
    
    def compare_models(self):
        """对比所有模型"""
        comparison = []
        for model in self.models:
            try:
                df = pd.read_excel(model['path'])
                comparison.append({
                    'model': model['name'],
                    'records': len(df),
                    'columns': list(df.columns)
                })
            except:
                pass
        return comparison
    
    def create_model_comparison_chart(self):
        """创建模型对比图"""
        try:
            comparison = self.compare_models()
            if not comparison:
                return self.create_empty_chart("暂无模型数据")
            
            models = [c['model'] for c in comparison]
            records = [c['records'] for c in comparison]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=models,
                y=records,
                marker_color='rgb(158, 202, 225)',
                name='数据记录数'
            ))
            
            fig.update_layout(
                title='各模型数据量对比',
                xaxis_title='模型名称',
                yaxis_title='记录数',
                template='plotly_white',
                height=400
            )
            
            return fig.to_json()
        except Exception as e:
            return self.create_empty_chart(f"图表生成错误: {e}")
    
    def create_empty_chart(self, message="暂无数据"):
        """创建空图表"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        return fig.to_json()

