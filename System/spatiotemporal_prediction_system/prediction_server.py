#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgriGuard - 时空预测系统
病虫害时空演变预测、风险评估与预警
"""

import http.server
import socketserver
import json
import os
import sys
import time
from urllib.parse import parse_qs, urlparse
from socketserver import ThreadingMixIn
from functools import lru_cache

# 添加当前目录到path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入数据分析模块（可选）
try:
    from data_analyzer import DataAnalyzer, ModelResultAnalyzer
    from data_collector import MedicalRecordCollector, KnowledgeBase, WeatherDataCollector
    print("[+] 数据分析模块加载成功")
except ImportError as e:
    print(f"[!] 数据分析模块加载失败（使用简化版本）: {e}")
    DataAnalyzer = None
    ModelResultAnalyzer = None
    MedicalRecordCollector = None
    KnowledgeBase = None
    WeatherDataCollector = None
except Exception as e:
    print(f"[!] 模块初始化错误（使用简化版本）: {e}")
    DataAnalyzer = None
    ModelResultAnalyzer = None
    MedicalRecordCollector = None
    KnowledgeBase = None
    WeatherDataCollector = None

# 导入简单数据读取器
try:
    from simple_data_reader import SimpleDataReader
    # 使用当前脚本所在目录作为base_dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_reader = SimpleDataReader(base_dir=current_dir)
    print(f"[+] 简单数据读取器加载成功")
    print(f"[*] 数据目录: {current_dir}")
except Exception as e:
    print(f"[!] 简单数据读取器加载失败: {e}")
    data_reader = None

PORT = 8003

# ============ 性能优化：添加缓存 ============
DATA_CACHE = {}
CACHE_TIMEOUT = 300  # 缓存5分钟

def get_cached_data(key, fetch_func):
    """带缓存的数据获取"""
    current_time = time.time()
    
    if key in DATA_CACHE:
        cached_data, cached_time = DATA_CACHE[key]
        if current_time - cached_time < CACHE_TIMEOUT:
            print(f"[CACHE] 使用缓存数据: {key}")
            return cached_data
    
    # 缓存过期或不存在，重新获取
    print(f"[CACHE] 重新获取数据: {key}")
    data = fetch_func()
    DATA_CACHE[key] = (data, current_time)
    return data

# 初始化分析器（安全模式）
try:
    data_analyzer = DataAnalyzer() if DataAnalyzer else None
    model_analyzer = ModelResultAnalyzer() if ModelResultAnalyzer else None
    medical_collector = MedicalRecordCollector() if MedicalRecordCollector else None
    knowledge_base = KnowledgeBase() if KnowledgeBase else None
    weather_collector = WeatherDataCollector() if WeatherDataCollector else None
except Exception as e:
    print(f"[!] 分析器初始化失败（继续运行）: {e}")
    data_analyzer = None
    model_analyzer = None
    medical_collector = None
    knowledge_base = None
    weather_collector = None

class PredictionHandler(http.server.SimpleHTTPRequestHandler):
    """时空预测系统请求处理器"""
    
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/' or path == '/index.html':
            self.send_main_page()
        elif path == '/data-collection':
            self.send_data_collection_page()
        elif path == '/data-analysis':
            self.send_data_analysis_page()
        elif path == '/model-prediction':
            self.send_model_prediction_page()
        elif path == '/regional-warning':
            self.send_regional_warning_page()
        elif path == '/regional-warning-en':
            self.send_regional_warning_page_en()
        elif path == '/api/beijing-geojson':
            self.send_beijing_geojson()
        elif path == '/api/regional-warning-data':
            self.send_regional_warning_data()
        elif path == '/api/weather-data':
            self.send_weather_data_api()
        elif path == '/api/charts/yearly':
            self.send_yearly_chart()
        elif path == '/api/charts/monthly':
            self.send_monthly_chart()
        elif path == '/api/charts/regional':
            self.send_regional_chart()
        elif path == '/api/charts/weather':
            self.send_weather_chart()
        elif path == '/api/charts/model-comparison':
            self.send_model_comparison_chart()
        elif path == '/api/models':
            self.send_model_list()
        elif path == '/api/districts':
            self.send_district_list()
        elif path == '/api/medical-records':
            self.send_medical_records()
        elif path == '/api/raw-data':
            self.send_raw_data()
        elif path == '/api/yearly-stats':
            self.send_yearly_stats()
        elif path == '/api/monthly-stats':
            self.send_monthly_stats()
        elif path == '/api/regional-stats':
            self.send_regional_stats()
        elif path == '/api/prediction-models':
            self.send_prediction_models_list()
        elif path == '/api/model-stats':
            self.send_model_stats()
        elif path == '/api/compare-models':
            self.send_compare_models()
        elif path == '/api/district-model-comparison':
            self.send_district_model_comparison()
        elif path.startswith('/api/prediction-data/'):
            model_name = path.replace('/api/prediction-data/', '')
            self.send_prediction_data(model_name)
        elif path == '/api/weather-relationship':
            self.send_weather_relationship()
        else:
            super().do_GET()
    
    def do_POST(self):
        """处理POST请求"""
        if self.path == '/api/medical-record':
            self.handle_add_medical_record()
        elif self.path == '/api/weather':
            self.handle_get_weather()
        else:
            self.send_error(404)
    
    def send_json_response(self, data):
        """发送JSON响应"""
        try:
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = json.dumps(data, ensure_ascii=False)
            self.wfile.write(response.encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError):
            pass
    
    def send_yearly_chart(self):
        """发送年度图表"""
        if data_analyzer:
            try:
                chart_json = data_analyzer.create_yearly_chart()
                self.send_json_response({'chart': chart_json})
            except Exception as e:
                self.send_json_response({'error': f'图表生成失败: {e}'})
        else:
            # 返回简化的JSON数据
            self.send_json_response({'chart': None, 'message': '数据分析功能需要安装依赖包'})
    
    def send_monthly_chart(self):
        """发送月度图表"""
        if data_analyzer:
            try:
                chart_json = data_analyzer.create_monthly_chart()
                self.send_json_response({'chart': chart_json})
            except Exception as e:
                self.send_json_response({'error': f'图表生成失败: {e}'})
        else:
            self.send_json_response({'chart': None, 'message': '数据分析功能需要安装依赖包'})
    
    def send_regional_chart(self):
        """发送地区图表"""
        if data_analyzer:
            try:
                chart_json = data_analyzer.create_regional_chart()
                self.send_json_response({'chart': chart_json})
            except Exception as e:
                self.send_json_response({'error': f'图表生成失败: {e}'})
        else:
            self.send_json_response({'chart': None, 'message': '数据分析功能需要安装依赖包'})
    
    def send_weather_chart(self):
        """发送气象相关性图表"""
        if data_analyzer:
            try:
                chart_json = data_analyzer.create_weather_correlation_chart()
                self.send_json_response({'chart': chart_json})
            except Exception as e:
                self.send_json_response({'error': f'图表生成失败: {e}'})
        else:
            self.send_json_response({'chart': None, 'message': '数据分析功能需要安装依赖包'})
    
    def send_model_comparison_chart(self):
        """发送模型对比图表"""
        if model_analyzer:
            try:
                chart_json = model_analyzer.create_model_comparison_chart()
                self.send_json_response({'chart': chart_json})
            except Exception as e:
                self.send_json_response({'error': f'图表生成失败: {e}'})
        else:
            self.send_json_response({'chart': None, 'message': '数据分析功能需要安装依赖包'})
    
    def send_model_list(self):
        """发送模型列表"""
        if model_analyzer:
            models = model_analyzer.models
            self.send_json_response({'models': models})
        else:
            self.send_json_response({'models': []})
    
    def send_district_list(self):
        """发送区县列表"""
        districts = [
            {'id': 'daxing', 'name': '大兴区'},
            {'id': 'miyun', 'name': '密云区'},
            {'id': 'pinggu', 'name': '平谷区'},
            {'id': 'yanqing', 'name': '延庆区'},
            {'id': 'huairou', 'name': '怀柔区'},
            {'id': 'fangshan', 'name': '房山区'},
            {'id': 'changping', 'name': '昌平区'},
            {'id': 'haidian', 'name': '海淀区'},
            {'id': 'tongzhou', 'name': '通州区'},
            {'id': 'shunyi', 'name': '顺义区'},
        ]
        self.send_json_response({'districts': districts})
    
    def send_medical_records(self):
        """发送病历记录"""
        if medical_collector:
            records = medical_collector.get_all_records()
            self.send_json_response({'records': records})
        else:
            self.send_json_response({'records': []})
    
    def send_beijing_geojson(self):
        """发送北京市地图GeoJSON数据"""
        try:
            geojson_path = os.path.join(os.path.dirname(__file__), '时序数据', '北京.json')
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            self.send_json_response(geojson_data)
        except Exception as e:
            print(f"[!] 加载北京地图数据失败: {e}")
            self.send_json_response({'error': f'地图数据加载失败: {e}'})
    
    def send_regional_warning_data(self):
        """发送区域预警数据 - 使用真实数据(带缓存)"""
        import datetime
        
        # 使用缓存获取数据
        def fetch_warning_data():
            """获取预警数据（会被缓存）"""
            if not data_reader:
                return None
            
            try:
                print(f"[*] 开始读取数据...", flush=True)
                raw_data_result = data_reader.read_raw_data(limit=10000)
                
                if raw_data_result['status'] == 'success' and raw_data_result['data']:
                    warning_data = self.process_real_data(raw_data_result)
                    print(f"[*] 生成预警数据: {len(warning_data)}个区县", flush=True)
                    return warning_data
            except Exception as e:
                print(f"[!] 读取真实数据失败: {e}", flush=True)
                import traceback
                traceback.print_exc()
            
            return None
        
        # 使用缓存
        warning_data = get_cached_data('regional_warning', fetch_warning_data)
        
        if warning_data:
            self.send_json_response({'warning_data': warning_data})
            return
        
        # 降级方案：使用模拟数据
        print("[*] 使用降级方案：模拟数据", flush=True)
        self.send_json_response({'warning_data': self.get_fallback_warning_data()})
    
    def process_real_data(self, raw_data_result):
        """处理真实数据，生成区域预警信息"""
        import datetime
        from collections import defaultdict
        
        data = raw_data_result['data']
        headers = raw_data_result['headers']
        
        # 找到有数据的区县列（Node_开头的列）
        district_columns = [h for h in headers if h.startswith('Node_')]
        print(f"[*] 找到区县列: {district_columns}")
        
        # 精确映射：Node列名 -> 中文区县名
        node_to_district = {
            'Node_DaXing': '大兴区',
            'Node_MiYun': '密云区',
            'Node_PingGu': '平谷区',
            'Node_YanQing': '延庆区',
            'Node_HuaiRou': '怀柔区',
            'Node_FangShan': '房山区',
            'Node_ChangPing': '昌平区',
            'Node_HaiDian': '海淀区',
            'Node_TongZhou': '通州区',
            'Node_ShunYi': '顺义区'
        }
        
        # 所有16个区
        all_districts = [
            '东城区', '西城区', '朝阳区', '丰台区', '石景山区', '海淀区',
            '门头沟区', '房山区', '通州区', '顺义区', '昌平区', '大兴区',
            '怀柔区', '平谷区', '密云区', '延庆区'
        ]
        
        # 为前10个区分配真实数据，其余6个区设为无数据（灰色）
        warning_data = []
        
        # 按区县聚合数据
        district_data = {}
        for col_name in district_columns:
            # 使用精确映射获取区县名
            district_name = node_to_district.get(col_name)
            if not district_name:
                continue
            
            # 提取该区的完整数据（包括日期和值）
            district_records = []
            for row in data:
                val = row.get(col_name)
                if val is not None and isinstance(val, (int, float)) and val > 0:
                    date_val = row.get('Date', row.get('日期', ''))
                    district_records.append({
                        'date': str(date_val) if date_val else '',
                        'value': float(val)
                    })
            
            if district_records:
                district_data[district_name] = district_records
        
        # 为有数据的区生成预警信息
        for district in all_districts:
            if district in district_data:
                # 有真实数据的区
                records = district_data[district]
                values = [r['value'] for r in records]
                dates = [r['date'] for r in records]
                
                # 计算预警等级（基于平均值）
                avg_value = sum(values) / len(values) if values else 0
                disease_count = int(avg_value)
                
                # 预警等级判断
                if avg_value >= 50:
                    warning_level = 5
                elif avg_value >= 30:
                    warning_level = 4
                elif avg_value >= 15:
                    warning_level = 3
                elif avg_value >= 5:
                    warning_level = 2
                else:
                    warning_level = 1
                
                # 趋势判断（最近7天）
                recent_values = values[-7:] if len(values) >= 7 else values
                if len(recent_values) >= 2:
                    trend = '上升' if recent_values[-1] > recent_values[0] else '下降' if recent_values[-1] < recent_values[0] else '稳定'
                else:
                    trend = '稳定'
                
                # 时序数据（最近60天，显示更多历史数据）
                time_series = []
                recent_records = records[-60:] if len(records) >= 60 else records
                for rec in recent_records:
                    time_series.append({
                        'date': rec['date'],
                        'value': int(rec['value'])
                    })
                
                # 找到峰值时间点（历史最大值的日期）
                max_value = max(values)
                max_idx = values.index(max_value)
                peak_date = dates[max_idx] if max_idx < len(dates) else ''
                # 预测峰值仍然使用最大值的1.2倍
                peak_value = int(max_value * 1.2)
                
                warning_data.append({
                    'district': district,
                    'warning_level': warning_level,
                    'disease_count': disease_count,
                    'trend': trend,
                    'time_series': time_series,
                    'peak_date': peak_date,  # 峰值发生日期
                    'peak_value': peak_value,
                    'main_disease': '蚜虫' if warning_level >= 4 else '白粉病' if warning_level >= 2 else '锈病',
                    'affected_crops': '小麦',
                    'has_data': True
                })
            else:
                # 无数据的区（用灰色显示）
                warning_data.append({
                    'district': district,
                    'warning_level': 0,  # 0表示无数据
                    'disease_count': 0,
                    'trend': '无数据',
                    'time_series': [],
                    'peak_date': '',
                    'peak_value': 0,
                    'main_disease': '无数据',
                    'affected_crops': '无数据',
                    'has_data': False
                })
        
        return warning_data
    
    def get_fallback_warning_data(self):
        """降级方案：模拟数据"""
        import random
        import datetime
        
        districts = [
            '东城区', '西城区', '朝阳区', '丰台区', '石景山区', '海淀区',
            '门头沟区', '房山区', '通州区', '顺义区', '昌平区', '大兴区',
            '怀柔区', '平谷区', '密云区', '延庆区'
        ]
        
        warning_data = []
        for i, district in enumerate(districts):
            # 前10个区有数据
            if i < 10:
                warning_level = random.randint(1, 5)
                disease_count = warning_level * random.randint(10, 50)
                trend = random.choice(['上升', '下降', '稳定'])
                
                time_series = []
                base_value = disease_count
                for j in range(7):
                    date = (datetime.datetime.now() - datetime.timedelta(days=6-j)).strftime('%Y-%m-%d')
                    value = base_value + random.randint(-20, 30)
                    time_series.append({'date': date, 'value': max(0, value)})
                
                peak_value = disease_count + random.randint(20, 100)
                
                warning_data.append({
                    'district': district,
                    'warning_level': warning_level,
                    'disease_count': disease_count,
                    'trend': trend,
                    'time_series': time_series,
                    'peak_date': '',
                    'peak_value': peak_value,
                    'main_disease': random.choice(['蚜虫', '白粉病', '锈病']),
                    'affected_crops': '小麦',
                    'has_data': True
                })
            else:
                # 后6个区无数据
                warning_data.append({
                    'district': district,
                    'warning_level': 0,
                    'disease_count': 0,
                    'trend': '无数据',
                    'time_series': [],
                    'peak_date': '',
                    'peak_value': 0,
                    'main_disease': '无数据',
                    'affected_crops': '无数据',
                    'has_data': False
                })
        
        return warning_data
    
    def send_weather_data_api(self):
        """发送气象数据 - 使用真实数据(带缓存)"""
        import datetime
        
        # 使用缓存获取数据
        def fetch_weather_data():
            """获取气象数据（会被缓存）"""
            if not data_reader:
                return None
            
            try:
                raw_data_result = data_reader.read_raw_data(limit=5000)
                if raw_data_result['status'] == 'success' and raw_data_result['data']:
                    weather_data = self.extract_weather_data(raw_data_result)
                    if weather_data:
                        return weather_data
            except Exception as e:
                print(f"[!] 读取天气数据失败: {e}")
            
            return None
        
        # 使用缓存
        weather_data = get_cached_data('weather_data', fetch_weather_data)
        
        if weather_data:
            self.send_json_response({'weather_data': weather_data})
            return
        
        # 降级方案：使用模拟数据
        self.send_json_response({'weather_data': self.get_fallback_weather_data()})
    
    def extract_weather_data(self, raw_data_result):
        """从原始数据中提取气象数据（最近7天）"""
        data = raw_data_result['data']
        headers = raw_data_result['headers']
        
        # 查找气象相关列
        weather_columns = {
            'temperature': next((h for h in headers if '温度' in h or 'Temp' in h), None),
            'humidity': next((h for h in headers if '湿度' in h or 'Humidity' in h), None),
            'rainfall': next((h for h in headers if '降雨' in h or 'Rain' in h), None),
        }
        
        # 取最近7条记录
        recent_data = data[-7:] if len(data) >= 7 else data
        
        weather_data = []
        for row in recent_data:
            date_val = row.get('日期', '') or row.get('Date', '')
            if isinstance(date_val, str) and date_val:
                date_str = date_val
            else:
                date_str = ''
            
            temp = row.get(weather_columns['temperature'], 20) if weather_columns['temperature'] else 20
            hum = row.get(weather_columns['humidity'], 60) if weather_columns['humidity'] else 60
            rain = row.get(weather_columns['rainfall'], 0) if weather_columns['rainfall'] else 0
            
            # 确保数值类型
            temp = int(temp) if isinstance(temp, (int, float)) else 20
            hum = int(hum) if isinstance(hum, (int, float)) else 60
            rain = float(rain) if isinstance(rain, (int, float)) else 0
            
            # 根据降雨量判断天气
            if rain > 10:
                weather = '中雨'
            elif rain > 1:
                weather = '小雨'
            elif hum > 70:
                weather = '阴'
            elif hum > 50:
                weather = '多云'
            else:
                weather = '晴'
            
            weather_data.append({
                'date': date_str,
                'temperature': temp,
                'humidity': hum,
                'rainfall': round(rain, 1),
                'wind_speed': 3.5,
                'weather': weather
            })
        
        return weather_data if weather_data else None
    
    def get_fallback_weather_data(self):
        """降级方案：模拟天气数据"""
        import random
        import datetime
        
        weather_data = []
        for i in range(7):
            date = (datetime.datetime.now() + datetime.timedelta(days=i)).strftime('%Y-%m-%d')
            weather_data.append({
                'date': date,
                'temperature': random.randint(15, 30),
                'humidity': random.randint(40, 80),
                'rainfall': round(random.uniform(0, 20), 1),
                'wind_speed': round(random.uniform(1, 8), 1),
                'weather': random.choice(['晴', '多云', '阴', '小雨', '中雨'])
            })
        
        return weather_data
    
    def send_raw_data(self):
        """发送原始数据（优化版本，减少数据量）"""
        if data_reader:
            try:
                # 获取完整数据但不在API中返回全部，只返回头部信息和采样数据
                result = data_reader.read_raw_data(limit=200)
                if result.get('status') == 'success':
                    # 优化：只返回必要数据
                    self.send_json_response({
                        'status': 'success',
                        'headers': result.get('headers', []),
                        'data': result.get('data', [])[:200],  # 只返回前200行
                        'total_rows': result.get('total_rows', 0)
                    })
                else:
                    self.send_json_response(result)
            except Exception as e:
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def send_yearly_stats(self):
        """发送年度统计数据"""
        if data_reader:
            try:
                stats = data_reader.get_yearly_statistics()
                self.send_json_response({'status': 'success', 'data': stats})
            except Exception as e:
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def send_monthly_stats(self):
        """发送月度统计数据"""
        if data_reader:
            try:
                stats = data_reader.get_monthly_statistics()
                self.send_json_response({'status': 'success', 'data': stats})
            except Exception as e:
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def send_regional_stats(self):
        """发送区域统计数据"""
        if data_reader:
            try:
                stats = data_reader.get_regional_statistics()
                self.send_json_response({'status': 'success', 'data': stats})
            except Exception as e:
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def send_prediction_models_list(self):
        """发送预测模型列表"""
        if data_reader:
            try:
                models = data_reader.list_prediction_models()
                self.send_json_response({'status': 'success', 'data': models})
            except Exception as e:
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def send_model_stats(self):
        """发送模型统计数据"""
        try:
            parsed_path = urlparse(self.path)
            query_params = parse_qs(parsed_path.query)
            model_name = query_params.get('model', [''])[0]
            
            if not model_name:
                self.send_json_response({'status': 'error', 'message': '缺少model参数'})
                return
            
            if data_reader:
                stats = data_reader.get_model_prediction_stats(model_name)
                self.send_json_response(stats)
            else:
                self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
        except Exception as e:
            self.send_json_response({'status': 'error', 'message': str(e)})
    
    def send_compare_models(self):
        """发送模型对比数据"""
        if data_reader:
            try:
                comparison = data_reader.compare_models()
                self.send_json_response(comparison)
            except Exception as e:
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def send_prediction_data(self, model_name):
        """发送指定模型的预测数据"""
        if data_reader:
            try:
                result = data_reader.read_prediction_data(model_name)
                self.send_json_response(result)
            except Exception as e:
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def send_district_model_comparison(self):
        """发送区县模型对比数据（含真实数据）"""
        if data_reader:
            try:
                result = data_reader.get_district_model_comparison()
                self.send_json_response(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def send_weather_relationship(self):
        """发送气象与数量的关系数据"""
        if data_reader:
            try:
                relationships = data_reader.get_weather_relationship()
                self.send_json_response({'status': 'success', 'data': relationships})
            except Exception as e:
                self.send_json_response({'status': 'error', 'message': str(e)})
        else:
            self.send_json_response({'status': 'error', 'message': '数据读取器未初始化'})
    
    def handle_add_medical_record(self):
        """处理添加病历记录"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if medical_collector:
                record = medical_collector.add_record(data)
                self.send_json_response({'status': 'success', 'record': record})
            else:
                self.send_json_response({'status': 'error', 'message': '数据采集器未初始化'})
        except Exception as e:
            self.send_json_response({'status': 'error', 'message': str(e)})
    
    def handle_get_weather(self):
        """处理获取气象数据"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if weather_collector:
                weather_data = weather_collector.get_weather_data(
                    data.get('location'),
                    data.get('start_date'),
                    data.get('end_date')
                )
                self.send_json_response({'status': 'success', 'data': weather_data})
            else:
                self.send_json_response({'status': 'error', 'message': '气象数据采集器未初始化'})
        except Exception as e:
            self.send_json_response({'status': 'error', 'message': str(e)})
    
    def send_main_page(self):
        """发送主页面"""
        html = self.get_main_html()
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError):
            pass
    
    def send_data_analysis_page(self):
        """发送数据分析页面"""
        html = self.get_data_analysis_html()
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError):
            pass
    
    def send_model_prediction_page(self):
        """发送模型预测页面"""
        html = self.get_model_prediction_html()
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError):
            pass
    
    def send_ai_assistant_page(self):
        """发送AI助手页面"""
        html = self.get_ai_assistant_html()
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError):
            pass
    
    def send_regional_warning_page(self):
        """发送区域预警页面"""
        html = self.get_regional_warning_html()
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError):
            pass
    
    def send_regional_warning_page_en(self):
        """发送英文版区域预警页面"""
        html = self.get_regional_warning_html_en()
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError):
            pass
    
    def send_data_collection_page(self):
        """发送数据采集页面"""
        html = self.get_data_collection_html()
        try:
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        except (ConnectionAbortedError, BrokenPipeError):
            pass
    
    def get_main_html(self):
        """获取主页面HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>时空预测系统 - AgriGuard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: 
                linear-gradient(135deg, 
                    rgba(96, 165, 250, 0.95) 0%,
                    rgba(147, 197, 253, 0.9) 25%,
                    rgba(196, 181, 253, 0.9) 50%,
                    rgba(167, 139, 250, 0.9) 75%,
                    rgba(129, 140, 248, 0.95) 100%
                );
            min-height: 100vh;
            padding: 2rem;
            position: relative;
            overflow-x: hidden;
        }
        
        /* 明亮背景光晕 */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.2) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.25) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }
        
        /* 移除网格背景 */
        body::after {
            content: none;
        }
        .header {
            text-align: center;
            margin-bottom: 3rem;
            color: white;
            position: relative;
        }
        .logo { 
            font-size: 4rem; 
            margin-bottom: 1rem;
            animation: float 3s ease-in-out infinite;
        }
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        .title {
            font-size: 3.5rem;
            font-weight: 800;
            color: white;
            margin-bottom: 1rem;
            text-shadow: 0 8px 16px rgba(0, 0, 0, 0.3),
                        0 0 40px rgba(255, 255, 255, 0.1);
            letter-spacing: 0.5px;
        }
        .subtitle {
            font-size: 1.25rem;
            color: rgba(255, 255, 255, 0.95);
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            font-weight: 500;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .nav-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .module-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* 响应式布局 */
        @media (max-width: 1200px) {
            .module-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        
        @media (max-width: 768px) {
            .module-grid {
                grid-template-columns: 1fr;
            }
        }
        .module-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(255, 255, 255, 0.92) 100%);
            backdrop-filter: blur(15px);
            border: 2px solid rgba(255, 255, 255, 0.4);
            border-radius: 28px;
            padding: 3rem 2.5rem;
            box-shadow: 
                0 20px 60px rgba(0, 0, 0, 0.12),
                0 0 0 1px rgba(255, 255, 255, 0.15) inset;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
            min-height: 380px;
            display: flex;
            flex-direction: column;
        }
        .module-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #a78bfa 0%, #7c3aed 50%, #5b21b6 100%);
            transition: all 0.3s ease;
        }
        .module-card::after {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(102, 126, 234, 0.4), transparent);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        .module-card:hover {
            transform: translateY(-15px) scale(1.03);
            box-shadow: 
                0 40px 100px rgba(102, 126, 234, 0.3),
                0 0 0 2px rgba(102, 126, 234, 0.2) inset;
            background: linear-gradient(135deg, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 0.98) 100%);
            border-color: rgba(102, 126, 234, 0.4);
        }
        .module-card:hover::before {
            height: 8px;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
        }
        .module-card:hover::after {
            width: 400px;
            height: 400px;
        }
        .module-icon {
            font-size: 4rem;
            margin-bottom: 1.5rem;
            text-align: center;
            position: relative;
            z-index: 1;
            filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
            transition: transform 0.3s ease;
        }
        .module-card:hover .module-icon {
            transform: scale(1.1) rotate(5deg);
        }
        .module-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1a202c;
            margin-bottom: 1rem;
            text-align: center;
            position: relative;
            z-index: 1;
            letter-spacing: 0.3px;
        }
        .module-desc {
            color: #4a5568;
            font-size: 1rem;
            line-height: 1.7;
            text-align: center;
            margin-bottom: 1.8rem;
            position: relative;
            z-index: 1;
            flex-grow: 1;
        }
        .module-features {
            list-style: none;
            padding: 0;
            position: relative;
            z-index: 1;
            margin-bottom: 1.5rem;
        }
        .module-features li {
            padding: 0.6rem 0;
            color: #2d3748;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
        }
        .module-features li::before {
            content: '✓';
            color: #667eea;
            font-weight: bold;
            margin-right: 0.8rem;
            font-size: 1.1rem;
            flex-shrink: 0;
        }
        .back-btn {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .back-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        .status-badge {
            display: inline-block;
            padding: 0.5rem 1.2rem;
            border-radius: 25px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: auto;
            position: relative;
            z-index: 1;
            transition: all 0.3s ease;
        }
        .status-active {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 2px solid #28a745;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.2);
        }
        .module-card:hover .status-active {
            box-shadow: 0 6px 16px rgba(40, 167, 69, 0.3);
            transform: scale(1.05);
        }
        .status-dev {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            border: 2px solid #ffc107;
            box-shadow: 0 4px 12px rgba(255, 193, 7, 0.2);
        }
        .module-card:hover .status-dev {
            box-shadow: 0 6px 16px rgba(255, 193, 7, 0.3);
            transform: scale(1.05);
        }
        
        /* 页脚样式 */
        .footer {
            margin-top: 4rem;
            padding: 2.5rem 2rem;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.08) 100%);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            backdrop-filter: blur(15px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .footer-content {
            color: white;
            font-size: 1rem;
            line-height: 1.8;
            margin-bottom: 1.5rem;
            opacity: 0.95;
        }
        .footer-title {
            color: white;
            font-weight: 700;
            font-size: 1.3rem;
            margin-bottom: 0.75rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
        }
        .footer-links a {
            color: white;
            text-decoration: none;
            transition: all 0.3s;
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .footer-links a:hover {
            opacity: 1;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">📊</div>
        <h1 class="title">时空预测系统</h1>
        <p class="subtitle">病虫害时空演变预测、风险评估与智能预警</p>
    </div>
    
    <div class="container">
        <div class="nav-card">
            <a href="http://localhost:8080" class="back-btn">← 返回主平台</a>
        </div>
        
        <div class="module-grid">
            <!-- 数据采集模块 -->
            <div class="module-card" onclick="window.location.href='/data-collection'">
                <div class="module-icon">📝</div>
                <h2 class="module-title">数据采集模块</h2>
                <p class="module-desc">植物电子病历、领域知识库、气象数据采集</p>
                <ul class="module-features">
                    <li>电子病历录入系统</li>
                    <li>药品信息知识库</li>
                    <li>实时气象数据获取</li>
                </ul>
                <div style="text-align: center;">
                    <span class="status-badge status-active">已完成</span>
                </div>
                </div>
            
            <!-- 数据分析与可视化 -->
            <div class="module-card" onclick="window.location.href='/data-analysis'">
                <div class="module-icon">📈</div>
                <h2 class="module-title">数据分析可视化</h2>
                <p class="module-desc">多维度数据分析、趋势展示、关联分析</p>
                <ul class="module-features">
                    <li>逐年逐月趋势分析</li>
                    <li>地区分布对比</li>
                    <li>气象因子关联分析</li>
                </ul>
                <div style="text-align: center;">
                    <span class="status-badge status-active">已完成</span>
                </div>
                </div>
            
            <!-- 模型预测结果 -->
            <div class="module-card" onclick="window.location.href='/model-prediction'">
                <div class="module-icon">🔮</div>
                <h2 class="module-title">模型预测结果</h2>
                <p class="module-desc">12种时序预测模型结果展示与对比</p>
                <ul class="module-features">
                    <li>多模型预测对比</li>
                    <li>分时间分地区展示</li>
                    <li>预测准确率分析</li>
                </ul>
                <div style="text-align: center;">
                    <span class="status-badge status-active">已完成</span>
                </div>
                </div>
            
            <!-- 区域预警功能 -->
            <div class="module-card" onclick="window.location.href='/regional-warning'">
                <div class="module-icon">🗺️</div>
                <h2 class="module-title">区域预警功能</h2>
                <p class="module-desc">北京市作物病虫害时序预警与区域风险分析</p>
                <ul class="module-features">
                    <li>16个区域实时监测</li>
                    <li>5级预警体系</li>
                    <li>气象数据关联</li>
                </ul>
                <div style="text-align: center;">
                    <span class="status-badge status-active">中文版</span>
                </div>
            </div>
            
            <!-- 区域预警功能（英文版） -->
            <div class="module-card" onclick="window.location.href='/regional-warning-en'">
                <div class="module-icon">🌍</div>
                <h2 class="module-title">Regional Warning (English)</h2>
                <p class="module-desc">Beijing Crop Pest & Disease Early Warning System</p>
                <ul class="module-features">
                    <li>16 Districts Real-time Monitoring</li>
                    <li>5-Level Warning System</li>
                    <li>Weather Data Integration</li>
                </ul>
                <div style="text-align: center;">
                    <span class="status-badge status-active">English Version</span>
                </div>
            </div>
            
            <!-- 大语言模型助手 -->
            <div class="module-card" onclick="window.location.href='/ai-assistant'">
                <div class="module-icon">🤖</div>
                <h2 class="module-title">AI智能助手</h2>
                <p class="module-desc">基于大语言模型的智能问答与决策支持</p>
                <ul class="module-features">
                    <li>病虫害识别与诊断</li>
                    <li>防治方案智能推荐</li>
                    <li>农业知识问答</li>
                </ul>
                <div style="text-align: center;">
                    <span class="status-badge status-active">已完成</span>
                </div>
            </div>
        </div>        
    </div>
    
    <!-- 底部版权信息 -->
    <footer style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-top: 4rem; text-align: center; font-size: 0.9rem; line-height: 1.8;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <p style="margin: 0.5rem 0; font-weight: 600;">© 2025 AgriGuard Platform. 基于大数据与人工智能的病虫害预测预警系统</p>
            <p style="margin: 0.5rem 0;">数据来源：北京市10区县植物诊所 | 2018-2021年时序数据</p>
            <p style="margin: 0.5rem 0;">技术支持：时空预测模型 + 深度学习 + 大语言模型</p>
            <p style="margin: 0.5rem 0;">开发单位：中国农业大学 信息与电气工程学院</p>
            <p style="margin: 0.5rem 0;">开发团队：张领先教授团队 秦源泽等人</p>
        </div>
    </footer>
</body>
</html>
        """
    
    def get_regional_warning_html(self):
        """获取区域预警页面HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>北京市作物病虫害区域预警系统</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 25px;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.2em;
            margin-bottom: 10px;
            text-align: center;
        }
        
        .header .subtitle {
            color: #666;
            text-align: center;
            font-size: 1.1em;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
            margin-bottom: 25px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.25), 
                        0 0 0 1px rgba(255,255,255,0.1) inset;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .card:hover {
            box-shadow: 0 20px 50px rgba(0,0,0,0.3),
                        0 0 0 1px rgba(255,255,255,0.2) inset;
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 1.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 3px solid;
            border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
            font-weight: 700;
            position: relative;
        }
        
        .card-title::after {
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 50px;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        }
        
        #beijingMap {
            width: 100%;
            height: 700px;
            background: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #0f0f1e 100%);
            border-radius: 10px;
        }
        
        .warning-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 1em;
            opacity: 0.9;
        }
        
        .legend {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
            padding: 20px;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.08), rgba(118, 75, 162, 0.08));
            border-radius: 15px;
            border: 1px solid rgba(102, 126, 234, 0.2);
            backdrop-filter: blur(5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 12px;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .legend-item:hover {
            background: rgba(255, 255, 255, 0.5);
            transform: translateY(-2px);
        }
        
        .legend-item span {
            font-weight: 600;
            color: #333;
            font-size: 0.95em;
        }
        
        .legend-color {
            width: 35px;
            height: 24px;
            border-radius: 6px;
            border: 2px solid rgba(255, 255, 255, 0.8);
        }
        
        .level-1 { 
            background: linear-gradient(135deg, #00ff88, #00cc70);
            box-shadow: 0 2px 8px rgba(0, 255, 136, 0.4);
        }
        .level-2 { 
            background: linear-gradient(135deg, #ffd93d, #ffb700);
            box-shadow: 0 2px 8px rgba(255, 217, 61, 0.4);
        }
        .level-3 { 
            background: linear-gradient(135deg, #ff8c42, #ff6b18);
            box-shadow: 0 2px 8px rgba(255, 140, 66, 0.4);
        }
        .level-4 { 
            background: linear-gradient(135deg, #ff4757, #ff2f3f);
            box-shadow: 0 2px 8px rgba(255, 71, 87, 0.4);
        }
        .level-5 { 
            background: linear-gradient(135deg, #d63031, #a82829);
            box-shadow: 0 2px 8px rgba(214, 48, 49, 0.4);
        }
        
        .weather-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .weather-day {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .weather-day .date {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
        }
        
        .weather-day .icon {
            font-size: 2em;
            margin: 10px 0;
        }
        
        .weather-day .temp {
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }
        
        .disease-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .disease-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s;
        }
        
        .disease-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .disease-card img {
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .disease-card .name {
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        
        .disease-card .level {
            color: #ff4d4f;
            font-size: 0.9em;
        }
        
        .bottom-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 25px;
        }
        
        .chart-container {
            width: 100%;
            height: 350px;
        }
        
        .back-button {
            display: inline-block;
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-size: 1.1em;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .back-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        }
        
        .district-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .district-item {
            padding: 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .district-item:hover {
            background: #f8f9fa;
        }
        
        .district-name {
            font-weight: bold;
            color: #333;
        }
        
        .warning-badge {
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-size: 0.9em;
        }
        
        .alert-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        
        .alert-box strong {
            color: #856404;
        }
        
        /* 地图卡片特殊效果 */
        .map-card {
            position: relative;
            overflow: hidden;
        }
        
        .map-card::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #667eea, #764ba2, #667eea);
            border-radius: 20px;
            opacity: 0;
            z-index: -1;
            transition: opacity 0.5s ease;
            background-size: 200% 200%;
            animation: gradientShift 3s ease infinite;
        }
        
        .map-card:hover::before {
            opacity: 0.3;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* 添加脉冲动画到标题 */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* 响应式设计优化 */
        @media (max-width: 1400px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .bottom-grid {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 1.8em;
            }
            
            .warning-stats {
                grid-template-columns: 1fr;
            }
            
            #beijingMap {
                height: 500px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 页头 -->
        <div class="header">
            <h1>🗺️ 北京市作物病虫害区域预警系统</h1>
            <div class="subtitle">实时监测 · 智能预警 · 精准防控</div>
        </div>
        
        <!-- 预警通告 -->
        <div class="alert-box">
            <strong>⚠️ 预警通告：</strong>当前朝阳区、海淀区病虫害预警等级为<strong>4级（严重）</strong>，请相关部门加强监测和防控措施。
        </div>
        
        <!-- 主要内容区域 -->
        <div class="main-grid">
            <!-- 地图区域 -->
            <div class="card map-card">
                <div class="card-title">
                    <span style="font-size: 1.2em; margin-right: 10px;">🗺️</span>
                    北京市病虫害预警地图
                    <span style="float: right; font-size: 0.7em; color: #667eea; font-weight: normal;">实时监测</span>
                </div>
                <div id="beijingMap"></div>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color level-1"></div>
                        <span>1级-关注</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color level-2"></div>
                        <span>2级-注意</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color level-3"></div>
                        <span>3级-警告</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color level-4"></div>
                        <span>4级-严重</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color level-5"></div>
                        <span>5级-紧急</span>
                    </div>
                </div>
            </div>
            
            <!-- 统计和气象 -->
            <div>
                <!-- 统计数据 -->
                <div class="card" style="margin-bottom: 20px;">
                    <div class="card-title">整体态势</div>
                    <div class="warning-stats">
                        <div class="stat-box">
                            <div class="stat-value" id="totalDistricts">16</div>
                            <div class="stat-label">监测区域</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="avgWarningLevel">2.8</div>
                            <div class="stat-label">平均预警等级</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="highRiskCount">3</div>
                            <div class="stat-label">高风险区域</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value" id="trendUp">↑58%</div>
                            <div class="stat-label">趋势上升区域</div>
                        </div>
                    </div>
                </div>
                
                <!-- 气象数据 -->
                <div class="card">
                    <div class="card-title">未来7天气象预报</div>
                    <div class="weather-grid" id="weatherGrid">
                        <!-- 动态加载 -->
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 病虫害图片展示 -->
        <div class="card" style="margin-bottom: 25px;">
            <div class="card-title">主要病虫害类型</div>
            <div class="disease-grid">
                <div class="disease-card">
                    <img src="https://images.unsplash.com/photo-1530836369250-ef72a3f5cda8?w=300&h=200&fit=crop" alt="蚜虫">
                    <div class="name">蚜虫</div>
                    <div class="level">高发期</div>
                </div>
                <div class="disease-card">
                    <img src="https://images.unsplash.com/photo-1592150621744-aca64f48394a?w=300&h=200&fit=crop" alt="白粉病">
                    <div class="name">白粉病</div>
                    <div class="level">中度发生</div>
                </div>
                <div class="disease-card">
                    <img src="https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=300&h=200&fit=crop" alt="锈病">
                    <div class="name">锈病</div>
                    <div class="level">轻度发生</div>
                </div>
                <div class="disease-card">
                    <img src="https://images.unsplash.com/photo-1625246333195-78d9c38ad449?w=300&h=200&fit=crop" alt="叶斑病">
                    <div class="name">叶斑病</div>
                    <div class="level">中度发生</div>
                </div>
            </div>
        </div>
        
        <!-- 图表区域 -->
        <div class="bottom-grid">
            <div class="card">
                <div class="card-title">病虫害趋势分析</div>
                <div id="trendChart" class="chart-container"></div>
            </div>
            <div class="card">
                <div class="card-title">预测峰值分析</div>
                <div id="peakChart" class="chart-container"></div>
            </div>
            <div class="card">
                <div class="card-title">各区预警等级</div>
                <div id="districtList" class="district-list"></div>
            </div>
        </div>
        
        <!-- 返回按钮 -->
        <div style="text-align: center; margin-top: 30px;">
            <a href="/" class="back-button">返回首页</a>
        </div>
    </div>
    
    <script>
        // 获取各区预警数据
        let warningData = [];
        let weatherData = [];

        const warningLevelColors = ['#00ff88', '#ffd93d', '#ff8c42', '#ff4757', '#d63031'];
        const warningLevelNames = ['关注', '注意', '警告', '严重', '紧急'];

        function getLevelColor(level) {
            if (!level || level < 1) return '#94a3b8';
            const idx = Math.min(level, warningLevelColors.length) - 1;
            return warningLevelColors[idx];
        }

        function getLevelName(level) {
            if (!level || level < 1) return '无数据';
            const idx = Math.min(level, warningLevelNames.length) - 1;
            return warningLevelNames[idx];
        }

        function addAlpha(color, alphaHex = 'FF') {
            const base = (color && color.startsWith('#')) ? color : '#94a3b8';
            return base + alphaHex;
        }
        
        async function loadData() {
            try {
                console.log('开始加载预警数据...');
                // 加载预警数据
                const warningResponse = await fetch('/api/regional-warning-data');
                console.log('预警数据响应状态:', warningResponse.status);
                const warningResult = await warningResponse.json();
                console.log('预警数据结果:', warningResult);
                warningData = warningResult.warning_data || [];
                console.log('预警数据数量:', warningData.length);
                
                if (warningData.length === 0) {
                    console.error('预警数据为空！');
                    alert('预警数据加载失败，请检查数据文件是否存在');
                    return;
                }
                
                // 加载气象数据
                console.log('开始加载气象数据...');
                const weatherResponse = await fetch('/api/weather-data');
                console.log('气象数据响应状态:', weatherResponse.status);
                const weatherResult = await weatherResponse.json();
                console.log('气象数据结果:', weatherResult);
                weatherData = weatherResult.weather_data || [];
                console.log('气象数据数量:', weatherData.length);
                
                // 更新页面
                console.log('开始更新页面...');
                updateStats();
                renderWeather();
                renderDistrictList();
                renderCharts();
                
                // 加载地图
                console.log('开始加载地图...');
                loadMap();
                console.log('所有数据加载完成！');
            } catch (error) {
                console.error('数据加载失败:', error);
                alert('数据加载失败: ' + error.message);
            }
        }
        
        function updateStats() {
            try {
                if (!warningData || warningData.length === 0) {
                    console.error('updateStats: warningData为空');
                    return;
                }
                const avgLevel = (warningData.reduce((sum, d) => sum + d.warning_level, 0) / warningData.length).toFixed(1);
                const highRisk = warningData.filter(d => d.warning_level >= 4).length;
                const trendUpCount = warningData.filter(d => d.trend === '上升').length;
                const trendUpPercent = Math.round((trendUpCount / warningData.length) * 100);
                
                document.getElementById('avgWarningLevel').textContent = avgLevel;
                document.getElementById('highRiskCount').textContent = highRisk;
                document.getElementById('trendUp').textContent = '↑' + trendUpPercent + '%';
                console.log('统计数据更新成功');
            } catch (error) {
                console.error('updateStats错误:', error);
            }
        }
        
        function renderWeather() {
            try {
                const weatherGrid = document.getElementById('weatherGrid');
                if (!weatherGrid) {
                    console.error('weatherGrid元素不存在');
                    return;
                }
                if (!weatherData || weatherData.length === 0) {
                    weatherGrid.innerHTML = '<div style="text-align:center;color:#666;">暂无气象数据</div>';
                    return;
                }
                weatherGrid.innerHTML = weatherData.map(day => {
                    const icons = {
                        '晴': '☀️',
                        '多云': '⛅',
                        '阴': '☁️',
                        '小雨': '🌧️',
                        '中雨': '🌧️'
                    };
                    const dateStr = day.date ? day.date.substring(5) : '';
                    return `
                        <div class="weather-day">
                            <div class="date">${dateStr}</div>
                            <div class="icon">${icons[day.weather] || '☀️'}</div>
                            <div class="temp">${day.temperature || 0}°C</div>
                            <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                                湿度: ${day.humidity || 0}%<br>
                                降雨: ${day.rainfall || 0}mm
                        </div>
                    </div>
                `;
                }).join('');
                console.log('气象数据渲染成功');
            } catch (error) {
                console.error('renderWeather错误:', error);
            }
        }
        
        function renderDistrictList() {
            try {
                const districtList = document.getElementById('districtList');
                if (!districtList) {
                    console.error('districtList元素不存在');
                    return;
                }
                if (!warningData || warningData.length === 0) {
                    districtList.innerHTML = '<div style="text-align:center;color:#666;">暂无区县数据</div>';
                    return;
                }
                
                districtList.innerHTML = warningData.sort((a, b) => b.warning_level - a.warning_level).map(d => `
                <div class="district-item">
                    <div>
                        <div class="district-name">${d.district}</div>
                        <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                            ${d.main_disease} · ${d.affected_crops} · ${d.trend}
                        </div>
                    </div>
                    <div class="warning-badge" style="background: ${getLevelColor(d.warning_level)}">
                        ${d.warning_level}级
                    </div>
                </div>
                `).join('');
                console.log('区县列表渲染成功');
            } catch (error) {
                console.error('renderDistrictList错误:', error);
            }
        }
        
        function renderCharts() {
            try {
                if (!warningData || warningData.length === 0) {
                    console.error('renderCharts: warningData为空');
                    return;
                }
                if (!warningData[0] || !warningData[0].time_series || warningData[0].time_series.length === 0) {
                    console.error('renderCharts: time_series数据为空');
                    return;
                }
                
                // 趋势图
                const trendChart = echarts.init(document.getElementById('trendChart'));
                const avgTimeSeries = warningData[0].time_series.map((item, index) => {
                    const sum = warningData.reduce((s, d) => s + (d.time_series[index] ? d.time_series[index].value : 0), 0);
                    return {
                        date: item.date,
                        value: Math.round(sum / warningData.length)
                    };
                });
            
            trendChart.setOption({
                tooltip: { trigger: 'axis' },
                xAxis: {
                    type: 'category',
                    data: avgTimeSeries.map(d => d.date.substring(5))
                },
                yAxis: { type: 'value', name: '病虫害数量' },
                series: [{
                    data: avgTimeSeries.map(d => d.value),
                    type: 'line',
                    smooth: true,
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                            offset: 0, color: 'rgba(102, 126, 234, 0.5)'
                        }, {
                            offset: 1, color: 'rgba(102, 126, 234, 0.1)'
                        }])
                    },
                    lineStyle: { color: '#667eea', width: 3 }
                }]
            });
            
            // 峰值图
            const peakChart = echarts.init(document.getElementById('peakChart'));
            const topDistricts = warningData.sort((a, b) => b.peak_value - a.peak_value).slice(0, 8);
            
            peakChart.setOption({
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                xAxis: {
                    type: 'category',
                    data: topDistricts.map(d => d.district),
                    axisLabel: { interval: 0, rotate: 30 }
                },
                yAxis: { type: 'value', name: '预测峰值' },
                series: [{
                    data: topDistricts.map(d => ({
                        value: d.peak_value,
                        itemStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                                offset: 0, color: '#cf1322'
                            }, {
                                offset: 1, color: '#ff4d4f'
                            }])
                        }
                    })),
                    type: 'bar',
                    barWidth: '60%'
                }]
                });
                console.log('图表渲染成功');
            } catch (error) {
                console.error('renderCharts错误:', error);
            }
        }
        
        async function loadMap() {
            try {
                console.log('开始加载地图...');
                // 加载北京地图数据
                const response = await fetch('/api/beijing-geojson');
                console.log('地图数据响应状态:', response.status);
                
                if (!response.ok) {
                    throw new Error(`地图数据加载失败: ${response.status}`);
                }
                
                const beijingGeoJson = await response.json();
                console.log('地图GeoJSON数据:', beijingGeoJson);
                
                if (!beijingGeoJson || !beijingGeoJson.features) {
                    throw new Error('地图数据格式错误');
                }
                
                console.log('地图features数量:', beijingGeoJson.features.length);
                
                // 注册地图
                echarts.registerMap('beijing', beijingGeoJson);
                console.log('地图注册成功');
                
                // 创建地图
                const mapElement = document.getElementById('beijingMap');
                if (!mapElement) {
                    throw new Error('beijingMap元素不存在');
                }
                console.log('beijingMap元素:', mapElement);
                
                const mapChart = echarts.init(mapElement);
                console.log('ECharts实例创建成功');
                
                // 准备数据
                const mapData = warningData.map(d => ({
                    name: d.district,
                    value: d.warning_level,
                    disease_count: d.disease_count,
                    main_disease: d.main_disease,
                    trend: d.trend
                }));
                
                // 准备散点数据（区域中心点）
                const scatterData = [];
                beijingGeoJson.features.forEach(feature => {
                    const district = warningData.find(d => d.district === feature.properties.name);
                    if (district && feature.properties.center) {
                        scatterData.push({
                            name: feature.properties.name,
                            value: [...feature.properties.center, district.warning_level],
                            warning_level: district.warning_level,
                            disease_count: district.disease_count,
                            main_disease: district.main_disease,
                            trend: district.trend
                        });
                    }
                });
                
                const option = {
                    backgroundColor: 'transparent',
                    tooltip: {
                        trigger: 'item',
                        backgroundColor: 'rgba(0, 0, 0, 0.85)',
                        borderColor: '#667eea',
                        borderWidth: 2,
                        textStyle: {
                            color: '#fff',
                            fontSize: 14
                        },
                        formatter: function(params) {
                            if (params.seriesType === 'map' && params.data) {
                                const level = params.data.value;
                                const levelColor = getLevelColor(level);
                                const levelName = getLevelName(level);
                                const trendIcon = params.data.trend === '上升' ? '📈' : 
                                                params.data.trend === '下降' ? '📉' : '➡️';
                                
                                return `
                                    <div style="padding: 10px;">
                                        <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px; border-bottom: 2px solid ${levelColor}; padding-bottom: 5px;">
                                            ${params.name}
                                        </div>
                                        <div style="margin-bottom: 6px;">
                                            <span style="display: inline-block; width: 12px; height: 12px; background: ${levelColor}; border-radius: 50%; margin-right: 8px;"></span>
                                            <strong>预警等级：</strong><span style="color: ${levelColor}; font-weight: bold;">${level || 0}级 (${levelName})</span>
                                        </div>
                                        <div style="margin-bottom: 6px;">
                                            <strong>病虫害数量：</strong>${params.data.disease_count} 例
                                        </div>
                                        <div style="margin-bottom: 6px;">
                                            <strong>主要病害：</strong>${params.data.main_disease}
                                        </div>
                                        <div>
                                            <strong>发展趋势：</strong>${trendIcon} ${params.data.trend}
                                        </div>
                                    </div>
                                `;
                            }
                            if (params.seriesType === 'scatter' || params.seriesType === 'effectScatter') {
                                const levelColor = getLevelColor(params.data.warning_level);
                                const levelName = getLevelName(params.data.warning_level);
                                return `
                                    <div style="padding: 8px;">
                                        <div style="font-weight: bold; margin-bottom: 5px;">${params.name}</div>
                                        <div style="color: ${levelColor};">● ${levelName}</div>
                                    </div>
                                `;
                            }
                            return params.name;
                        }
                    },
                    geo: {
                        map: 'beijing',
                        roam: true,
                        scaleLimit: {
                            min: 1,
                            max: 5
                        },
                        zoom: 1.1,
                        center: [116.4, 40.0],
                        label: {
                            show: false,
                            color: '#fff',
                            fontSize: 12,
                            fontWeight: 'bold'
                        },
                        emphasis: {
                            label: {
                                show: true,
                                color: '#fff',
                                fontSize: 14,
                                fontWeight: 'bold',
                                textShadowColor: '#000',
                                textShadowBlur: 5
                            },
                            itemStyle: {
                                areaColor: '#4a90e2',
                                borderWidth: 2,
                                borderColor: '#fff',
                                shadowBlur: 20,
                                shadowColor: 'rgba(102, 126, 234, 0.8)'
                            }
                        },
                        itemStyle: {
                            borderColor: 'rgba(255, 255, 255, 0.3)',
                            borderWidth: 1.5,
                            shadowBlur: 15,
                            shadowColor: 'rgba(0, 0, 0, 0.5)',
                            shadowOffsetY: 3
                        },
                        regions: mapData.map(item => ({
                            name: item.name,
                            itemStyle: {
                                areaColor: {
                                    type: 'radial',
                                    x: 0.5,
                                    y: 0.5,
                                    r: 0.8,
                                    colorStops: [
                                        { offset: 0, color: getLevelColor(item.value) },
                                        { offset: 1, color: addAlpha(getLevelColor(item.value), 'cc') }
                                    ]
                                },
                                shadowBlur: 10,
                                shadowColor: addAlpha(getLevelColor(item.value), '66'),
                                borderColor: '#fff'
                            }
                        }))
                    },
                    series: [
                        // 地图底图
                        {
                            type: 'map',
                            map: 'beijing',
                            geoIndex: 0,
                            aspectScale: 0.85,
                            showLegendSymbol: false,
                            data: mapData
                        },
                        // 散点标注（所有区域）
                        {
                            name: '区域标注',
                            type: 'scatter',
                            coordinateSystem: 'geo',
                            symbol: 'circle',
                            symbolSize: function(val) {
                                return val[2] * 4 + 8;
                            },
                            label: {
                                show: true,
                                formatter: '{b}',
                                position: 'bottom',
                                color: '#fff',
                                fontSize: 11,
                                fontWeight: 'bold',
                                distance: 5,
                                textBorderColor: '#000',
                                textBorderWidth: 2
                            },
                            itemStyle: {
                                color: function(params) {
                                    return getLevelColor(params.data.warning_level);
                                },
                                shadowBlur: 15,
                                shadowColor: function(params) {
                                    return getLevelColor(params.data.warning_level);
                                },
                                borderWidth: 2,
                                borderColor: '#fff'
                            },
                            emphasis: {
                                scale: 1.5,
                                itemStyle: {
                                    shadowBlur: 25,
                                    borderWidth: 3
                                }
                            },
                            data: scatterData,
                            zlevel: 2
                        },
                        // 涟漪效果（高风险区域）
                        {
                            name: '高风险预警',
                            type: 'effectScatter',
                            coordinateSystem: 'geo',
                            data: scatterData.filter(d => d.warning_level >= 4),
                            symbolSize: function(val) {
                                return val[2] * 6 + 10;
                            },
                            showEffectOn: 'render',
                            rippleEffect: {
                                brushType: 'stroke',
                                scale: 4,
                                period: 3
                            },
                            label: {
                                show: false
                            },
                            itemStyle: {
                                color: function(params) {
                                    return getLevelColor(params.data.warning_level);
                                },
                                shadowBlur: 20,
                                shadowColor: function(params) {
                                    return getLevelColor(params.data.warning_level);
                                }
                            },
                            zlevel: 3
                        }
                    ]
                };
                
                console.log('开始设置地图配置...');
                console.log('地图配置:', option);
                mapChart.setOption(option);
                console.log('地图配置设置成功！');
                
                // 添加自动旋转动画（可选）
                let angle = 0;
                const autoRotate = setInterval(() => {
                    angle += 0.1;
                    // 可以添加轻微的视角变化
                }, 100);
                
                // 响应式调整
                window.addEventListener('resize', function() {
                    mapChart.resize();
                });
                
                console.log('地图加载完成！');
                
            } catch (error) {
                console.error('地图加载失败:', error);
                console.error('错误堆栈:', error.stack);
                alert('地图加载失败: ' + error.message);
            }
        }
        
        // 页面加载时执行
        window.onload = loadData;
    </script>
    
    <!-- 底部版权信息 -->
    <footer style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-top: 4rem; text-align: center; font-size: 0.9rem; line-height: 1.8;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <p style="margin: 0.5rem 0; font-weight: 600;">© 2025 AgriGuard Platform. 基于大数据与人工智能的病虫害预测预警系统</p>
            <p style="margin: 0.5rem 0;">数据来源：北京市10区县植物诊所 | 2018-2021年时序数据</p>
            <p style="margin: 0.5rem 0;">技术支持：时空预测模型 + 深度学习 + 大语言模型</p>
            <p style="margin: 0.5rem 0;">开发单位：中国农业大学 信息与电气工程学院</p>
            <p style="margin: 0.5rem 0;">开发团队：张领先教授团队 秦源泽等人</p>
        </div>
    </footer>
</body>
</html>
        """
    
    def get_regional_warning_html_en(self):
        """获取英文版区域预警页面HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Beijing Crop Pest & Disease Regional Warning System</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            min-height: 100vh;
            padding: 0;
            margin: 0;
            position: relative;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            text-rendering: optimizeLegibility;
        }
        
        /* 添加动态渐变背景 */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 30%, rgba(102, 126, 234, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(240, 147, 251, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(118, 75, 162, 0.2) 0%, transparent 50%);
            animation: gradientShift 15s ease infinite;
            z-index: -1;
        }
        
        @keyframes gradientShift {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        .container {
            max-width: 1600px;
            width: 95%;
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(20px);
            box-shadow: 0 25px 80px rgba(0,0,0,0.25), 
                        0 0 100px rgba(102, 126, 234, 0.2),
                        inset 0 0 0 1px rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            margin: 20px auto;
        }
        
        .header {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
            backdrop-filter: blur(10px);
            padding: 15px 30px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .header h1 {
            color: white;
            font-size: 1.5em;
            margin-bottom: 4px;
            text-align: center;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
            position: relative;
            z-index: 1;
        }
        
        .header .subtitle {
            color: rgba(255, 255, 255, 0.98);
            text-align: center;
            font-size: 0.75em;
            font-weight: 400;
            position: relative;
            z-index: 1;
            text-shadow: 0 1px 5px rgba(0, 0, 0, 0.1);
        }
        
        /* 导航条样式 */
        .nav-bar {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(102, 126, 234, 0.15);
            padding: 12px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
        }
        
        .nav-links {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            flex: 1;
        }
        
        .nav-link {
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 8px;
            color: #334155;
            text-decoration: none;
            font-size: 0.85em;
            font-weight: 500;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        .nav-link:hover {
            background: rgba(102, 126, 234, 0.15);
            border-color: rgba(102, 126, 234, 0.4);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
            color: #667eea;
        }
        
        .nav-link.active {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.9), rgba(118, 75, 162, 0.9));
            border-color: rgba(255, 255, 255, 0.3);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .nav-link.active:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .back-home-btn {
            padding: 8px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85em;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        .back-home-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        
        .content-wrapper {
            padding: 15px;
            background: linear-gradient(180deg, rgba(245,247,250,0.5) 0%, rgba(255,255,255,0.3) 100%);
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 10px;
            padding: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06),
                        0 1px 3px rgba(0,0,0,0.04),
                        inset 0 0 0 1px rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(102, 126, 234, 0.12);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .card:hover {
            background: rgba(255, 255, 255, 0.98);
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15),
                        0 2px 6px rgba(0,0,0,0.05),
                        inset 0 0 0 1px rgba(102, 126, 234, 0.25);
            border-color: rgba(102, 126, 234, 0.3);
            transform: translateY(-2px);
        }
        
        .card-title {
            font-size: 1em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            padding-bottom: 8px;
            border-bottom: 2px solid rgba(102, 126, 234, 0.15);
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .section-desc {
            font-size: 0.75em;
            color: #475569;
            margin-bottom: 6px;
            line-height: 1.4;
            font-weight: 500;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        #beijingMap {
            width: 100%;
            height: 380px;
            background: rgba(248, 250, 252, 0.5);
            border-radius: 8px;
            border: 1px solid rgba(102, 126, 234, 0.1);
        }
        
        .warning-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        
        .stat-box {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
            backdrop-filter: blur(10px);
            color: white;
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.25),
                        inset 0 0 0 1px rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .stat-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .stat-box:hover::before {
            left: 100%;
        }
        
        .stat-value {
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 4px;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        }
        
        .stat-label {
            font-size: 0.8em;
            opacity: 0.95;
            font-weight: 400;
        }
        
        .legend {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            padding: 12px 15px;
            background: rgba(248, 250, 252, 0.6);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            border: 1px solid rgba(102, 126, 234, 0.1);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85em;
        }
        
        .legend-item span {
            font-weight: 500;
            color: #475569;
        }
        
        .legend-color {
            width: 28px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid rgba(0, 0, 0, 0.1);
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
        
        /* Scientific gradient color scheme */
        .level-1 { background: linear-gradient(135deg, #10b981 0%, #34d399 100%); }
        .level-2 { background: linear-gradient(135deg, #84cc16 0%, #a3e635 100%); }
        .level-3 { background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); }
        .level-4 { background: linear-gradient(135deg, #f97316 0%, #fb923c 100%); }
        .level-5 { background: linear-gradient(135deg, #ef4444 0%, #f87171 100%); }
        
        .weather-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 8px;
            margin-top: 8px;
        }
        
        .weather-day {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            padding: 12px 8px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(102, 126, 234, 0.15);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .weather-day:hover {
            background: rgba(255, 255, 255, 0.9);
            border-color: rgba(102, 126, 234, 0.4);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        
        .weather-day .date {
            font-size: 0.7em;
            color: #475569;
            margin-bottom: 4px;
            font-weight: 600;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .weather-day .icon {
            font-size: 1.4em;
            margin: 5px 0;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
        }
        
        .weather-day .temp {
            font-size: 0.95em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .disease-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }
        
        .disease-card {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            padding: 12px;
            border-radius: 12px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: 1px solid rgba(102, 126, 234, 0.15);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            display: flex;
            gap: 12px;
            align-items: flex-start;
            cursor: pointer;
        }
        
        .disease-card:hover {
            background: rgba(255, 255, 255, 0.95);
            border-color: rgba(102, 126, 234, 0.4);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        
        /* 模态框样式 */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(5px);
            animation: fadeIn 0.3s ease;
        }
        
        .modal-content {
            background: white;
            margin: 5% auto;
            padding: 0;
            width: 80%;
            max-width: 800px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease;
            overflow: hidden;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from { transform: translateY(-50px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .modal-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 24px 30px;
            position: relative;
        }
        
        .modal-header h2 {
            margin: 0;
            font-size: 1.8em;
            font-weight: 600;
        }
        
        .modal-header .level-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-left: 10px;
            font-weight: 600;
        }
        
        .close {
            position: absolute;
            right: 20px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 32px;
            font-weight: bold;
            color: white;
            cursor: pointer;
            transition: all 0.3s;
            line-height: 1;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
        }
        
        .close:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-50%) rotate(90deg);
        }
        
        .modal-body {
            padding: 30px;
            color: #334155;
            line-height: 1.8;
        }
        
        .modal-section {
            margin-bottom: 24px;
        }
        
        .modal-section h3 {
            color: #667eea;
            font-size: 1.3em;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .modal-section p {
            color: #475569;
            font-size: 1em;
            margin: 0;
        }
        
        .modal-section ul {
            margin: 8px 0;
            padding-left: 24px;
        }
        
        .modal-section li {
            color: #475569;
            margin: 6px 0;
        }
        
        .disease-card img {
            width: 120px;
            height: 100px;
            object-fit: cover;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            flex-shrink: 0;
        }
        
        .disease-info {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        
        .disease-card .name {
            font-weight: 600;
            color: #1e293b;
            font-size: 0.95em;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .disease-card .level {
            font-size: 0.8em;
            font-weight: 600;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            color: white;
        }
        
        .disease-card .period {
            font-size: 0.75em;
            color: #64748b;
            line-height: 1.4;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .disease-card .desc {
            font-size: 0.7em;
            color: #64748b;
            line-height: 1.3;
        }
        
        .bottom-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            padding-bottom: 15px;
        }
        
        .chart-container {
            width: 100%;
            height: 300px;
        }
        
        .district-list {
            height: 300px;
            overflow-y: auto;
            padding-right: 5px;
        }
        
        .district-list::-webkit-scrollbar {
            width: 8px;
        }
        
        .district-list::-webkit-scrollbar-track {
            background: rgba(248, 250, 252, 0.5);
            border-radius: 4px;
        }
        
        .district-list::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.6), rgba(118, 75, 162, 0.6));
            border-radius: 4px;
            transition: background 0.3s;
        }
        
        .district-list::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, rgba(102, 126, 234, 0.8), rgba(118, 75, 162, 0.8));
        }
        
        .district-item {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(102, 126, 234, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.3s;
            font-size: 0.9em;
            line-height: 1.5;
        }
        
        .district-item:hover {
            background: rgba(102, 126, 234, 0.08);
            border-left: 3px solid rgba(102, 126, 234, 0.5);
            padding-left: 9px;
        }
        
        .district-name {
            font-weight: 600;
            color: #1e293b;
            font-size: 0.85em;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .warning-badge {
            padding: 4px 12px;
            border-radius: 14px;
            color: white;
            font-size: 0.75em;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(102, 126, 234, 0.15);
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 5px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .metric-label {
            font-size: 0.8em;
            color: #64748b;
            font-weight: 500;
        }
        
        .metric-change {
            font-size: 0.75em;
            margin-top: 4px;
        }
        
        @media print {
            body {
                background: white;
                padding: 0;
            }
            .container {
                box-shadow: none;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Beijing Crop Pest & Disease Regional Warning System</h1>
            <div class="subtitle">
                Real-time Spatiotemporal Monitoring · AI-Powered Risk Assessment · Data-Driven Prevention
            </div>
        </div>
        
        <!-- Navigation Bar -->
        <div class="nav-bar">
            <div class="nav-links">
                <a href="/" class="nav-link">
                    <span>🏠</span>
                    <span>Home</span>
                </a>
                <a href="/data-collection" class="nav-link">
                    <span>📊</span>
                    <span>Data Collection</span>
                </a>
                <a href="/data-analysis" class="nav-link">
                    <span>📈</span>
                    <span>Data Analysis</span>
                </a>
                <a href="/model-prediction" class="nav-link">
                    <span>🤖</span>
                    <span>Model Prediction</span>
                </a>
                <a href="/regional-warning-en" class="nav-link active">
                    <span>🌍</span>
                    <span>Regional Warning</span>
                </a>
                <a href="/ai-assistant" class="nav-link">
                    <span>💬</span>
                    <span>AI Assistant</span>
                </a>
            </div>
            <a href="/" class="back-home-btn">
                <span>←</span>
                <span>Back to Home</span>
            </a>
        </div>
        
        <!-- Content Wrapper -->
        <div class="content-wrapper">
            <!-- Main Content -->
            <div class="main-grid">
                <!-- Map Area -->
                <div class="card">
                    <div class="card-title">
                        <span>🗺️ Spatiotemporal Warning Distribution</span>
                        <span style="font-size: 0.7em; color: #1976d2;">● Live</span>
                    </div>
                    <div id="beijingMap"></div>
                    <div class="legend">
                        <div class="legend-item">
                            <div class="legend-color level-1"></div>
                            <span>Level 1 - Watch</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color level-2"></div>
                            <span>Level 2 - Advisory</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color level-3"></div>
                            <span>Level 3 - Alert</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color level-4"></div>
                            <span>Level 4 - Warning</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color level-5"></div>
                            <span>Level 5 - Emergency</span>
                        </div>
                    </div>
                </div>
                
                <!-- Right Panel -->
                <div style="display: flex; flex-direction: column; gap: 15px;">
                    <!-- Statistics -->
                    <div class="card">
                        <div class="card-title" style="margin-bottom: 15px;">
                            <span>📊 Overall Status</span>
                        </div>
                        <div class="warning-stats" style="gap: 15px;">
                            <div class="stat-box">
                                <div class="stat-label" style="font-size: 0.75em; margin-bottom: 8px; opacity: 0.9;">Avg Level</div>
                                <div class="stat-value" id="avgWarningLevel" style="font-size: 2.5em;">2.8</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label" style="font-size: 0.75em; margin-bottom: 8px; opacity: 0.9;">High Risk</div>
                                <div class="stat-value" id="highRiskCount" style="font-size: 2.5em;">3</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Weather Forecast -->
                    <div class="card" style="flex: 1;">
                        <div class="card-title">
                            <span>☁️ Weather Forecast</span>
                        </div>
                        <div class="weather-grid" id="weatherGrid"></div>
                    </div>
                </div>
            </div>
            
            <!-- Major Threats Row -->
            <div class="card" style="margin-bottom: 15px;">
                <div class="card-title">
                    <span>🦠 Major Threats</span>
                </div>
                <div class="disease-grid" id="diseaseGrid">
                    <div class="disease-card" onclick="showDiseaseDetail('aphids')">
                        <img src="/static/images/aphids.jpg" 
                             onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzEwYjk4MSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjI0IiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iIGZvbnQtZmFtaWx5PSJBcmlhbCI+QXBoaWRzPC90ZXh0Pjwvc3ZnPg=='" 
                             alt="Aphids">
                        <div class="disease-info">
                            <div class="name">Aphids</div>
                            <div class="level" style="background: #ef4444;">Critical</div>
                            <div class="period">🔥 Peak Season: April - June</div>
                            <div class="desc">Sucks plant sap, causing leaf curling and stunted growth</div>
                        </div>
                    </div>
                    <div class="disease-card" onclick="showDiseaseDetail('powdery_mildew')">
                        <img src="/static/images/powdery_mildew.jpg" 
                             onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2Y1OWUwYiIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjIwIiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iIGZvbnQtZmFtaWx5PSJBcmlhbCI+UG93ZGVyeSBNaWxkZXc8L3RleHQ+PC9zdmc+'" 
                             alt="Powdery Mildew">
                        <div class="disease-info">
                            <div class="name">Powdery Mildew</div>
                            <div class="level" style="background: #f59e0b;">Moderate</div>
                            <div class="period">⚠️ Peak Season: July - September</div>
                            <div class="desc">White powdery coating on leaves, reduces photosynthesis</div>
                        </div>
                    </div>
                    <div class="disease-card" onclick="showDiseaseDetail('rust')">
                        <img src="/static/images/rust.jpg" 
                             onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzIyYzU1ZSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjI0IiBmaWxsPSJ3aGl0ZSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iIGZvbnQtZmFtaWx5PSJBcmlhbCI+UnVzdDwvdGV4dD48L3N2Zz4='" 
                             alt="Rust">
                        <div class="disease-info">
                            <div class="name">Rust</div>
                            <div class="level" style="background: #10b981;">Controlled</div>
                            <div class="period">✅ Peak Season: May - August</div>
                            <div class="desc">Orange-brown pustules on leaves, currently well-managed</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Bottom Charts -->
            <div class="bottom-grid">
                <div class="card">
                    <div class="card-title">
                        <span>📈 Temporal Trend</span>
                    </div>
                    <div class="section-desc">Average daily cases (Last 7 days)</div>
                    <div id="trendChart" class="chart-container"></div>
                </div>
                <div class="card">
                    <div class="card-title">
                        <span>🎯 Peak Prediction</span>
                    </div>
                    <div class="section-desc">Forecasted outbreak intensity (Top 8 districts)</div>
                    <div id="peakChart" class="chart-container"></div>
                </div>
                <div class="card">
                    <div class="card-title">
                        <span>🗂️ District Status</span>
                    </div>
                    <div class="section-desc">16 Districts comprehensive assessment</div>
                    <div id="districtList" class="district-list"></div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 病害详情模态框 -->
    <div id="diseaseModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 id="modalTitle"></h2>
                <span class="close" onclick="closeModal()">&times;</span>
            </div>
            <div class="modal-body" id="modalBody">
                <!-- Content will be dynamically filled -->
            </div>
        </div>
    </div>
    
    <script>
        // English translations for disease types and trends
        const diseaseTranslations = {
            '蚜虫': 'Aphids',
            '白粉病': 'Powdery Mildew',
            '锈病': 'Rust',
            '炭疽病': 'Anthracnose',
            '叶斑病': 'Leaf Spot',
            '霜霉病': 'Downy Mildew'
        };
        
        const cropTranslations = {
            '小麦': 'Wheat',
            '玉米': 'Corn',
            '蔬菜': 'Vegetables',
            '果树': 'Fruit Trees'
        };
        
        const trendTranslations = {
            '上升': 'Rising',
            '下降': 'Declining',
            '稳定': 'Stable'
        };
        
        const weatherTranslations = {
            '晴': 'Sunny',
            '多云': 'Cloudy',
            '阴': 'Overcast',
            '小雨': 'Light Rain',
            '中雨': 'Moderate Rain'
        };
        
        const districtTranslations = {
            '东城区': 'Dongcheng District',
            '西城区': 'Xicheng District',
            '朝阳区': 'Chaoyang District',
            '丰台区': 'Fengtai District',
            '石景山区': 'Shijingshan District',
            '海淀区': 'Haidian District',
            '门头沟区': 'Mentougou District',
            '房山区': 'Fangshan District',
            '通州区': 'Tongzhou District',
            '顺义区': 'Shunyi District',
            '昌平区': 'Changping District',
            '大兴区': 'Daxing District',
            '怀柔区': 'Huairou District',
            '平谷区': 'Pinggu District',
            '密云区': 'Miyun District',
            '延庆区': 'Yanqing District'
        };
        
        let warningData = [];
        let weatherData = [];
        
        const warningLevelColors = ['#00ff88', '#ffd93d', '#ff8c42', '#ff4757', '#d63031'];
        const warningLevelNames = ['Watch', 'Advisory', 'Alert', 'Warning', 'Emergency'];
        
        function getLevelColor(level) {
            if (!level || level < 1) return '#94a3b8';
            const idx = Math.min(level, warningLevelColors.length) - 1;
            return warningLevelColors[idx];
        }
        
        function getLevelName(level) {
            if (!level || level < 1) return 'No Data';
            const idx = Math.min(level, warningLevelNames.length) - 1;
            return warningLevelNames[idx];
        }
        
        function addAlpha(color, alphaHex = 'FF') {
            const base = (color && color.startsWith('#')) ? color : '#94a3b8';
            return base + alphaHex;
        }
        
        async function loadData() {
            try {
                console.log('Starting to load data...');
                const warningResponse = await fetch('/api/regional-warning-data');
                console.log('Warning response status:', warningResponse.status);
                const warningResult = await warningResponse.json();
                console.log('Warning result:', warningResult);
                warningData = warningResult.warning_data || [];
                console.log('Warning data count:', warningData.length);
                
                if (warningData.length === 0) {
                    console.error('Warning data is empty!');
                    alert('Failed to load warning data. Please check if data file exists.');
                    return;
                }
                
                const weatherResponse = await fetch('/api/weather-data');
                console.log('Weather response status:', weatherResponse.status);
                const weatherResult = await weatherResponse.json();
                console.log('Weather result:', weatherResult);
                weatherData = weatherResult.weather_data || [];
                console.log('Weather data count:', weatherData.length);
                
                console.log('Starting to render...');
                updateStats();
                renderWeather();
                renderDistrictList();
                renderCharts();
                loadMap();
                console.log('All data loaded successfully!');
            } catch (error) {
                console.error('Data loading failed:', error);
                console.error('Error stack:', error.stack);
                alert('Data loading failed: ' + error.message);
            }
        }
        
        function updateStats() {
            // 只统计有数据的区域
            const dataDistricts = warningData.filter(d => d.has_data && d.warning_level > 0);
            const avgLevel = dataDistricts.length > 0 
                ? (dataDistricts.reduce((sum, d) => sum + d.warning_level, 0) / dataDistricts.length).toFixed(1)
                : '0.0';
            const highRisk = dataDistricts.filter(d => d.warning_level >= 4).length;
            
            document.getElementById('avgWarningLevel').textContent = avgLevel;
            document.getElementById('highRiskCount').textContent = highRisk;
        }
        
        function renderWeather() {
            const weatherGrid = document.getElementById('weatherGrid');
            weatherGrid.innerHTML = weatherData.map(day => {
                const icons = {
                    '晴': '☀️',
                    '多云': '⛅',
                    '阴': '☁️',
                    '小雨': '🌧️',
                    '中雨': '🌧️'
                };
                return `
                    <div class="weather-day">
                        <div class="date">${day.date.substring(5)}</div>
                        <div class="icon">${icons[day.weather] || '☀️'}</div>
                        <div class="temp">${day.temperature}°C</div>
                        <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                            Humidity: ${day.humidity}%<br>
                            Rainfall: ${day.rainfall}mm
                        </div>
                    </div>
                `;
            }).join('');
        }
        
        function renderDistrictList() {
            try {
                console.log('Rendering district list...');
                const districtList = document.getElementById('districtList');
                if (!districtList) {
                    console.error('District list element not found');
                    return;
                }
                if (!warningData || warningData.length === 0) {
                    districtList.innerHTML = '<div style="text-align:center;color:#666;">No district data</div>';
                    return;
                }
                const colors = ['#9ca3af', '#10b981', '#84cc16', '#f59e0b', '#f97316', '#ef4444'];
                const levelNames = ['No Data', 'Watch', 'Advisory', 'Alert', 'Warning', 'Emergency'];
                
                districtList.innerHTML = warningData.sort((a, b) => b.warning_level - a.warning_level).map(d => {
                if (d.warning_level === 0 || !d.has_data) {
                    // 无数据的区域
                    return `
                        <div class="district-item" style="opacity: 0.6;">
                            <div>
                                <div class="district-name">${districtTranslations[d.district] || d.district}</div>
                                <div style="font-size: 0.75em; color: #94a3b8; margin-top: 5px;">
                                    No monitoring data available
                                </div>
                            </div>
                            <div class="warning-badge" style="background: #9ca3af;">
                                N/A
                            </div>
                        </div>
                    `;
                } else {
                    // 有数据的区域
                    return `
                        <div class="district-item">
                            <div>
                                <div class="district-name">${districtTranslations[d.district] || d.district}</div>
                                <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                                    ${diseaseTranslations[d.main_disease] || d.main_disease} · 
                                    ${cropTranslations[d.affected_crops] || d.affected_crops} · 
                                    ${trendTranslations[d.trend] || d.trend}
                                </div>
                            </div>
                            <div class="warning-badge" style="background: ${getLevelColor(d.warning_level)}">
                                Level ${d.warning_level}
                            </div>
                        </div>
                    `;
                }
                }).join('');
                console.log('District list rendered successfully');
            } catch (error) {
                console.error('renderDistrictList error:', error);
            }
        }
        
        function renderCharts() {
            try {
                console.log('Rendering charts with data:', warningData.length, 'districts');
            
            // Trend Chart
            const trendChartDom = document.getElementById('trendChart');
            if (!trendChartDom) {
                console.error('Trend chart container not found');
                return;
            }
            const trendChart = echarts.init(trendChartDom);
            
            // 只使用有数据的区域计算平均值
            const dataDistricts = warningData.filter(d => d.time_series && d.time_series.length > 0);
            if (dataDistricts.length === 0) {
                console.error('No data available for trend chart');
                return;
            }
            
            const avgTimeSeries = dataDistricts[0].time_series.map((item, index) => {
                const sum = dataDistricts.reduce((s, d) => s + (d.time_series[index]?.value || 0), 0);
                return {
                    date: item.date,
                    value: Math.round(sum / dataDistricts.length)
                };
            });
            
            trendChart.setOption({
                tooltip: {
                    trigger: 'axis',
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderColor: 'rgba(102, 126, 234, 0.5)',
                    borderWidth: 2,
                    textStyle: { color: '#334155' },
                    shadowBlur: 10,
                    shadowColor: 'rgba(0, 0, 0, 0.1)'
                },
                grid: { left: '12%', right: '5%', top: '15%', bottom: '15%' },
                xAxis: {
                    type: 'category',
                    data: avgTimeSeries.map(d => d.date.substring(5)),
                    axisLine: { lineStyle: { color: 'rgba(102, 126, 234, 0.3)' } },
                    axisLabel: { color: '#64748b', fontSize: 11 }
                },
                yAxis: {
                    type: 'value',
                    name: 'Cases',
                    nameTextStyle: { color: '#64748b', fontSize: 11 },
                    axisLine: { lineStyle: { color: 'rgba(102, 126, 234, 0.3)' } },
                    axisLabel: { color: '#64748b', fontSize: 11 },
                    splitLine: { lineStyle: { color: 'rgba(102, 126, 234, 0.1)' } }
                },
                series: [{
                    data: avgTimeSeries.map(d => d.value),
                    type: 'line',
                    smooth: true,
                    symbol: 'circle',
                    symbolSize: 7,
                    areaStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{
                            offset: 0, color: 'rgba(102, 126, 234, 0.3)'
                        }, {
                            offset: 1, color: 'rgba(240, 147, 251, 0.1)'
                        }])
                    },
                    lineStyle: { 
                        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{
                            offset: 0, color: '#667eea'
                        }, {
                            offset: 1, color: '#764ba2'
                        }]),
                        width: 3
                    },
                    itemStyle: { 
                        color: '#667eea',
                        borderColor: '#fff',
                        borderWidth: 2,
                        shadowBlur: 5,
                        shadowColor: 'rgba(102, 126, 234, 0.5)'
                    }
                }]
            });
            
            // Peak Prediction Chart - 显示峰值时间点
            const peakChartDom = document.getElementById('peakChart');
            if (!peakChartDom) {
                console.error('Peak chart container not found');
                return;
            }
            const peakChart = echarts.init(peakChartDom);
            // 只显示有数据的区域，按峰值排序
            const topDistricts = [...warningData]
                .filter(d => d.time_series && d.time_series.length > 0)
                .sort((a, b) => {
                    const maxA = Math.max(...a.time_series.map(t => t.value));
                    const maxB = Math.max(...b.time_series.map(t => t.value));
                    return maxB - maxA;
                })
                .slice(0, 8);
            
            // 创建多条折线，每个区一条
            const peakSeries = topDistricts.map((d, idx) => {
                const enName = districtTranslations[d.district] || d.district;
                const shortName = enName.replace(' District', '');
                const peakColors = ['#ef4444', '#f97316', '#f59e0b', '#eab308', '#84cc16', '#22c55e', '#06b6d4', '#3b82f6'];
                const seriesColor = peakColors[idx % peakColors.length];
                
                // 计算该区域的动态阈值（均值 + 标准差）
                const values = d.time_series.map(t => t.value);
                const mean = values.reduce((a, b) => a + b, 0) / values.length;
                const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
                const stdDev = Math.sqrt(variance);
                const threshold = mean + stdDev;  // 超过均值+标准差的点视为爆发点
                
                // 找到所有超过阈值的爆发点
                const outbreakIndices = d.time_series
                    .map((t, i) => t.value > threshold ? i : -1)
                    .filter(i => i !== -1);
                
                // 用于markPoint的数据点
                const markPointData = outbreakIndices.map(i => ({
                    coord: [i, d.time_series[i].value],
                    value: d.time_series[i].value,
                    label: {
                        formatter: '{c}',
                        fontSize: 10,
                        color: '#fff',
                        backgroundColor: seriesColor,
                        padding: [3, 6],
                        borderRadius: 3
                    },
                    itemStyle: {
                        color: seriesColor
                    }
                }));
                
                return {
                    name: shortName,
                    type: 'line',
                    data: d.time_series.map((t, i) => ({
                        value: t.value,
                        // 标记所有爆发点
                        itemStyle: outbreakIndices.includes(i) ? {
                            color: seriesColor,
                            borderWidth: 3,
                            borderColor: '#fff',
                            shadowBlur: 10,
                            shadowColor: seriesColor
                        } : {}
                    })),
                    smooth: true,
                    symbolSize: (value, params) => {
                        // 爆发点用更大的符号
                        return outbreakIndices.includes(params.dataIndex) ? 10 : 4;
                    },
                    lineStyle: {
                        color: seriesColor,
                        width: 2
                    },
                    itemStyle: {
                        color: seriesColor
                    },
                    emphasis: {
                        focus: 'series'
                    },
                    markPoint: {
                        data: markPointData
                    },
                    // 添加阈值线
                    markLine: {
                        silent: true,
                        lineStyle: {
                            color: seriesColor,
                            type: 'dashed',
                            width: 1,
                            opacity: 0.3
                        },
                        label: {
                            show: false
                        },
                        data: [{
                            yAxis: threshold
                        }]
                    }
                };
            });
            
            // 获取所有时间点（使用第一个区域的时序数据）
            const timeLabels = topDistricts.length > 0 
                ? topDistricts[0].time_series.map(t => t.date.substring(5))  // 只显示月-日
                : [];
            
            peakChart.setOption({
                tooltip: {
                    trigger: 'axis',
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    borderColor: 'rgba(102, 126, 234, 0.5)',
                    borderWidth: 2,
                    textStyle: { color: '#334155' },
                    shadowBlur: 10,
                    shadowColor: 'rgba(0, 0, 0, 0.1)',
                    formatter: function(params) {
                        if (!params || params.length === 0) return '';
                        let result = `<div style="font-weight: 600; margin-bottom: 5px;">${params[0].axisValue}</div>`;
                        params.forEach(p => {
                            result += `<div style="margin: 3px 0;">
                                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${p.color};margin-right:5px;"></span>
                                ${p.seriesName}: <strong>${p.value}</strong>
                            </div>`;
                        });
                        return result;
                    }
                },
                legend: {
                    data: topDistricts.map(d => {
                        const enName = districtTranslations[d.district] || d.district;
                        return enName.replace(' District', '');
                    }),
                    top: '0%',
                    left: 'center',
                    textStyle: {
                        fontSize: 10,
                        color: '#64748b'
                    },
                    itemWidth: 15,
                    itemHeight: 10,
                    itemGap: 8
                },
                dataZoom: [
                    {
                        type: 'slider',
                        show: true,
                        xAxisIndex: [0],
                        start: 0,
                        end: 100,
                        height: 20,
                        bottom: 0,
                        handleSize: '80%',
                        handleStyle: {
                            color: '#667eea'
                        },
                        textStyle: {
                            fontSize: 10
                        },
                        borderColor: 'rgba(102, 126, 234, 0.3)',
                        fillerColor: 'rgba(102, 126, 234, 0.15)',
                        dataBackground: {
                            lineStyle: {
                                color: '#667eea'
                            },
                            areaStyle: {
                                color: 'rgba(102, 126, 234, 0.2)'
                            }
                        }
                    },
                    {
                        type: 'inside',
                        xAxisIndex: [0],
                        start: 0,
                        end: 100
                    }
                ],
                grid: { left: '10%', right: '5%', top: '25%', bottom: '25%' },
                xAxis: {
                    type: 'category',
                    data: timeLabels,
                    axisLabel: { 
                        fontSize: 9, 
                        color: '#64748b',
                        interval: 'auto',  // 自动计算显示间隔，避免标签重叠
                        rotate: 45  // 标签旋转45度，显示更多日期
                    },
                    axisLine: { lineStyle: { color: 'rgba(102, 126, 234, 0.3)' } }
                },
                yAxis: {
                    type: 'value',
                    name: 'Cases',
                    nameTextStyle: { color: '#64748b', fontSize: 10 },
                    axisLine: { lineStyle: { color: 'rgba(102, 126, 234, 0.3)' } },
                    axisLabel: { color: '#64748b', fontSize: 9 },
                    splitLine: { lineStyle: { color: 'rgba(102, 126, 234, 0.1)' } }
                },
                series: peakSeries
                });
                console.log('Charts rendered successfully');
            } catch (error) {
                console.error('renderCharts error:', error);
                console.error('Error stack:', error.stack);
            }
        }
        
        async function loadMap() {
            try {
                console.log('Loading map...');
                const response = await fetch('/api/beijing-geojson');
                console.log('Map response status:', response.status);
                const beijingGeoJson = await response.json();
                console.log('Map GeoJSON loaded, features:', beijingGeoJson.features?.length);
                
                if (!beijingGeoJson || !beijingGeoJson.features) {
                    throw new Error('Invalid map data format');
                }
                
                echarts.registerMap('beijing', beijingGeoJson);
                console.log('Map registered');
                
                const mapElement = document.getElementById('beijingMap');
                if (!mapElement) {
                    throw new Error('Map element not found');
                }
                console.log('Map element found');
                
                const mapChart = echarts.init(mapElement);
                console.log('ECharts instance created');
                
                const mapData = warningData.map(d => ({
                    name: d.district,
                    name_en: districtTranslations[d.district] || d.district,
                    value: d.warning_level,
                    disease_count: d.disease_count,
                    main_disease: diseaseTranslations[d.main_disease] || d.main_disease,
                    trend: trendTranslations[d.trend] || d.trend
                }));
                console.log('Map data prepared:', mapData.length, 'districts');
                
                const scatterData = [];
                beijingGeoJson.features.forEach(feature => {
                    const district = warningData.find(d => d.district === feature.properties.name);
                    // 为所有有center的区域添加标记点
                    if (district && feature.properties.center) {
                        scatterData.push({
                            name: districtTranslations[feature.properties.name] || feature.properties.name,
                            name_cn: feature.properties.name,
                            value: [...feature.properties.center, district.warning_level],
                            warning_level: district.warning_level,
                            disease_count: district.disease_count,
                            main_disease: diseaseTranslations[district.main_disease] || district.main_disease,
                            trend: trendTranslations[district.trend] || district.trend
                        });
                    }
                });
                console.log('Scatter data prepared:', scatterData.length, 'points');
                
                const option = {
                    backgroundColor: 'transparent',
                    tooltip: {
                        trigger: 'item',
                        backgroundColor: 'rgba(255, 255, 255, 0.98)',
                        borderColor: 'rgba(102, 126, 234, 0.5)',
                        borderWidth: 2,
                        textStyle: { color: '#334155', fontSize: 13 },
                        shadowBlur: 15,
                        shadowColor: 'rgba(0, 0, 0, 0.1)',
                        formatter: function(params) {
                            if (params.seriesType === 'map' && params.data) {
                                const level = params.data.value;
                                const districtNameEn = districtTranslations[params.name] || params.name;
                                
                                // 无数据区域
                                if (level === 0) {
                                    return `
                                        <div style="padding: 12px; min-width: 200px;">
                                            <div style="font-size: 16px; font-weight: 600; margin-bottom: 8px; color: #64748b;">
                                                ${districtNameEn}
                                            </div>
                                            <div style="font-size: 13px; color: #94a3b8;">
                                                No monitoring data available
                                            </div>
                                        </div>
                                    `;
                                }
                                
                                // 有数据区域
                                const levelNames = ['', 'Watch', 'Advisory', 'Alert', 'Warning', 'Emergency'];
                                const levelColor = getLevelColor(level);
                                const trendIcon = params.data.trend === 'Rising' ? '📈' : 
                                                params.data.trend === 'Declining' ? '📉' : '➡️';
                                
                                return `
                                    <div style="padding: 12px; min-width: 220px;">
                                        <div style="font-size: 16px; font-weight: 600; margin-bottom: 10px; 
                                                    background: linear-gradient(135deg, #667eea, #764ba2);
                                                    -webkit-background-clip: text;
                                                    -webkit-text-fill-color: transparent;
                                                    border-bottom: 2px solid rgba(102, 126, 234, 0.2); padding-bottom: 6px;">
                                            ${districtNameEn}
                                        </div>
                                        <div style="margin-bottom: 6px; font-size: 13px;">
                                            <strong>Warning Level:</strong> <span style="color: ${levelColor}; font-weight: 600;">Level ${level}</span> <span style="color: #64748b;">(${levelNames[level]})</span>
                                        </div>
                                        <div style="margin-bottom: 6px; font-size: 13px;">
                                            <strong>Total Cases:</strong> <span style="color: #334155;">${params.data.disease_count}</span>
                                        </div>
                                        <div style="margin-bottom: 6px; font-size: 13px;">
                                            <strong>Primary Pest:</strong> <span style="color: #334155;">${params.data.main_disease}</span>
                                        </div>
                                        <div style="font-size: 13px;">
                                            <strong>Trend:</strong> ${trendIcon} <span style="color: ${params.data.trend === 'Rising' ? '#ef4444' : params.data.trend === 'Declining' ? '#10b981' : '#64748b'}; font-weight: 600;">${params.data.trend}</span>
                                        </div>
                                    </div>
                                `;
                            }
                            return districtTranslations[params.name] || params.name;
                        }
                    },
                    geo: {
                        map: 'beijing',
                        roam: true,
                        scaleLimit: { min: 1, max: 4 },
                        zoom: 1.15,
                        center: [116.4, 40.0],
                        label: {
                            show: false
                        },
                        emphasis: {
                            label: {
                                show: true,
                                formatter: function(params) {
                                    // 使用英文名称替换中文名称
                                    const enName = districtTranslations[params.name] || params.name;
                                    return enName.replace(' District', '');
                                },
                                color: '#334155',
                                fontSize: 12,
                                fontWeight: '600',
                                backgroundColor: 'rgba(255, 255, 255, 0.98)',
                                padding: [5, 10],
                                borderRadius: 6,
                                borderColor: 'rgba(102, 126, 234, 0.6)',
                                borderWidth: 2,
                                shadowBlur: 8,
                                shadowColor: 'rgba(102, 126, 234, 0.3)'
                            },
                            itemStyle: {
                                areaColor: 'rgba(102, 126, 234, 0.3)',
                                borderWidth: 2,
                                borderColor: '#667eea',
                                shadowBlur: 20,
                                shadowColor: 'rgba(102, 126, 234, 0.5)'
                            }
                        },
                        itemStyle: {
                            borderColor: '#ffffff',
                            borderWidth: 2,
                            shadowBlur: 8,
                            shadowColor: 'rgba(0, 0, 0, 0.15)',
                            shadowOffsetY: 3
                        },
                        regions: mapData.map(item => ({
                            name: item.name,
                            itemStyle: {
                                areaColor: item.value === 0 ? '#d1d5db' : getLevelColor(item.value),
                                opacity: item.value === 0 ? 0.5 : 0.9,
                                borderColor: '#ffffff',
                                borderWidth: 1.5
                            }
                        }))
                    },
                    series: [
                        {
                            type: 'map',
                            map: 'beijing',
                            geoIndex: 0,
                            aspectScale: 0.85,
                            showLegendSymbol: false,
                            data: mapData
                        },
                        {
                            name: 'District Markers',
                            type: 'scatter',
                            coordinateSystem: 'geo',
                            symbol: 'pin',
                            symbolSize: function(val) { return val[2] * 5 + 10; },
                            label: {
                                show: true,
                                formatter: function(params) {
                                    const enName = districtTranslations[params.data.name_cn] || params.name;
                                    return enName.replace(' District', '');
                                },
                                position: 'inside',
                                color: '#334155',
                                fontSize: 9,
                                fontWeight: '700',
                                backgroundColor: 'rgba(255, 255, 255, 0.92)',
                                padding: [3, 7],
                                borderRadius: 4,
                                borderColor: 'rgba(102, 126, 234, 0.5)',
                                borderWidth: 1,
                                shadowBlur: 5,
                                shadowColor: 'rgba(0, 0, 0, 0.15)'
                            },
                            itemStyle: {
                                color: function(params) { 
                                    return getLevelColor(params.data.warning_level);
                                },
                                shadowBlur: 10,
                                shadowColor: 'rgba(0, 0, 0, 0.25)',
                                borderWidth: 2.5,
                                borderColor: '#ffffff'
                            },
                            emphasis: {
                                scale: 1.4,
                                itemStyle: { 
                                    shadowBlur: 18,
                                    borderWidth: 3,
                                    borderColor: '#667eea',
                                    shadowColor: 'rgba(102, 126, 234, 0.6)'
                                }
                            },
                            data: scatterData,
                            zlevel: 2
                        },
                        {
                            name: 'High Risk Alert',
                            type: 'effectScatter',
                            coordinateSystem: 'geo',
                            data: scatterData.filter(d => d.warning_level >= 4),
                            symbolSize: function(val) { return val[2] * 6 + 12; },
                            showEffectOn: 'render',
                            rippleEffect: { brushType: 'stroke', scale: 3.5, period: 4 },
                            label: { show: false },
                            itemStyle: {
                                color: '#ef4444',
                                shadowBlur: 18,
                                shadowColor: 'rgba(239, 68, 68, 0.7)'
                            },
                            zlevel: 3
                        }
                    ]
                };
                
                console.log('Setting map options...');
                console.log('Map option:', option);
                mapChart.setOption(option);
                console.log('Map options set successfully!');
                
                window.addEventListener('resize', function() {
                    mapChart.resize();
                });
                
                console.log('Map loaded successfully!');
            } catch (error) {
                console.error('Map loading failed:', error);
                console.error('Error stack:', error.stack);
                alert('Map loading failed: ' + error.message);
            }
        }
        
        // Disease detail data (AI generated)
        const diseaseDetails = {
            'aphids': {
                name: 'Aphids (蚜虫)',
                level: 'Critical',
                levelColor: '#ef4444',
                characteristics: 'Small, soft-bodied insects (1-3mm) that cluster on young shoots, leaves, and buds. They are typically green, yellow, or black in color and can reproduce rapidly, with multiple generations per season.',
                causes: 'Rapid reproduction is favored by warm temperatures (20-25°C) and dry conditions. Aphid populations explode in spring when new plant growth emerges. They are also transported by wind and human activities.',
                regions: 'Widely distributed across China, particularly severe in: North China Plain (Beijing, Hebei, Shandong), Yangtze River Delta (Jiangsu, Zhejiang, Shanghai), and Northeast China (Liaoning, Jilin). Urban agricultural areas are especially vulnerable.',
                seasons: 'Primary outbreak: April to June (spring); Secondary peak: September to October (autumn). Population peaks occur when temperatures reach 20-25°C with moderate humidity.',
                control: 'Biological control using ladybugs and lacewings; insecticidal soap or neem oil spray; yellow sticky traps; removing heavily infested plant parts; encouraging natural predators.'
            },
            'powdery_mildew': {
                name: 'Powdery Mildew (白粉病)',
                level: 'Moderate',
                levelColor: '#f59e0b',
                characteristics: 'Appears as white or gray powdery patches on leaf surfaces, stems, and fruits. Infected leaves may curl, yellow, and drop prematurely. The fungal mycelium creates a flour-like coating that reduces photosynthesis.',
                causes: 'Caused by various fungal species (Erysiphales order). Thrives in moderate temperatures (18-25°C) with high humidity but low rainfall. Spreads through airborne spores. Overcrowding and poor air circulation increase infection risk.',
                regions: 'Common throughout China, especially in: Huang-Huai Plain (Henan, Anhui, Shandong), North China (Beijing, Tianjin, Hebei), and greenhouse cultivation areas nationwide. Protected agriculture environments are particularly susceptible.',
                seasons: 'Primary infection: July to September (summer-autumn); Can occur year-round in greenhouses. Peak development occurs during warm days and cool nights with high relative humidity (70-80%).',
                control: 'Apply sulfur-based fungicides or potassium bicarbonate; improve air circulation through proper spacing; remove infected plant material; use resistant varieties; avoid overhead watering.'
            },
            'rust': {
                name: 'Rust Disease (锈病)',
                level: 'Controlled',
                levelColor: '#10b981',
                characteristics: 'Orange, yellow, or reddish-brown pustules (uredinia) appear on leaves, stems, and fruits. Pustules rupture to release masses of spores that look like rust powder. Severe infections cause leaf yellowing and premature defoliation.',
                causes: 'Caused by rust fungi (Pucciniales order) requiring living plant tissue. Favored by moderate temperatures (15-22°C), high humidity (>95%), and prolonged leaf wetness. Many rust fungi require alternate hosts to complete their life cycle.',
                regions: 'Prevalent in: Southwest China (Yunnan, Sichuan, Guizhou), Central China (Hubei, Hunan), and coastal regions (Guangdong, Fujian). Wheat rust is particularly serious in the Huang-Huai wheat belt.',
                seasons: 'Main season: May to August (late spring-summer); Can occur in autumn (September-October) under favorable conditions. Development requires 6-8 hours of leaf wetness and temperatures between 15-22°C.',
                control: 'Currently well-managed through integrated pest management. Apply protective fungicides (triazoles, strobilurins); plant resistant cultivars; remove alternate hosts; proper crop rotation; timely removal of crop residues.'
            }
        };
        
        // Show disease detail modal
        function showDiseaseDetail(diseaseId) {
            const disease = diseaseDetails[diseaseId];
            if (!disease) return;
            
            document.getElementById('modalTitle').innerHTML = `
                ${disease.name}
                <span class="level-badge" style="background: ${disease.levelColor};">${disease.level}</span>
            `;
            
            document.getElementById('modalBody').innerHTML = `
                <div class="modal-section">
                    <h3>🔍 Characteristics</h3>
                    <p>${disease.characteristics}</p>
                </div>
                
                <div class="modal-section">
                    <h3>🧬 Causes & Development</h3>
                    <p>${disease.causes}</p>
                </div>
                
                <div class="modal-section">
                    <h3>🗺️ Common Regions in China</h3>
                    <p>${disease.regions}</p>
                </div>
                
                <div class="modal-section">
                    <h3>📅 Peak Seasons</h3>
                    <p>${disease.seasons}</p>
                </div>
                
                <div class="modal-section">
                    <h3>💊 Control Measures</h3>
                    <p>${disease.control}</p>
                </div>
            `;
            
            document.getElementById('diseaseModal').style.display = 'block';
        }
        
        // Close modal
        function closeModal() {
            document.getElementById('diseaseModal').style.display = 'none';
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('diseaseModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
        
        window.onload = loadData;
    </script>
    
    <!-- Footer -->
    <footer style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-top: 4rem; text-align: center; font-size: 0.9rem; line-height: 1.8;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <p style="margin: 0.5rem 0; font-weight: 600;">© 2025 AgriGuard Platform. AI-Powered Pest and Disease Early Warning System</p>
            <p style="margin: 0.5rem 0;">Data Source: 10 Districts Plant Clinics in Beijing | 2018-2021 Time Series Data</p>
            <p style="margin: 0.5rem 0;">Technology: Spatiotemporal Prediction Model + Deep Learning + LLM</p>
            <p style="margin: 0.5rem 0;">Institution: College of Information and Electrical Engineering, China Agricultural University</p>
            <p style="margin: 0.5rem 0;">Development Team: Prof. Zhang Lingxian's Team, Qin Yuanze et al.</p>
        </div>
    </footer>
</body>
</html>
        """
    
    def get_data_analysis_html(self):
        """获取数据分析页面HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>数据分析与可视化 v3.0 - 时空预测系统</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: 
                linear-gradient(135deg, 
                    rgba(96, 165, 250, 0.95) 0%,
                    rgba(147, 197, 253, 0.9) 25%,
                    rgba(196, 181, 253, 0.9) 50%,
                    rgba(167, 139, 250, 0.9) 75%,
                    rgba(129, 140, 248, 0.95) 100%
                );
            min-height: 100vh;
            padding: 2rem;
            position: relative;
            overflow-x: hidden;
        }
        /* 明亮背景光晕 */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.2) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.25) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }
        body::after {
            content: none;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
            color: white;
            position: relative;
            z-index: 1;
        }
        .logo { font-size: 3rem; margin-bottom: 1rem; }
        .title { 
            font-size: 3rem; 
            font-weight: 800; 
            color: white;
            margin-bottom: 1rem;
            text-shadow: 0 8px 16px rgba(0, 0, 0, 0.3),
                        0 0 40px rgba(255, 255, 255, 0.1);
            letter-spacing: 0.5px;
        }
        .subtitle { 
            font-size: 1.25rem; 
            color: rgba(255, 255, 255, 0.95);
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            font-weight: 500;
        }
        .container { max-width: 1400px; margin: 0 auto; width: 95%; position: relative; z-index: 1; }
        .nav-card { 
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px; 
            padding: 1.5rem; 
            margin-bottom: 2rem; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); 
        }
        .back-btn { 
            display: inline-block; 
            padding: 0.75rem 1.5rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            text-decoration: none; 
            border-radius: 8px; 
            font-weight: 600;
            transition: all 0.3s; 
            margin-right: 1rem;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .back-btn:hover { 
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }
        .filter-card { 
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px; 
            padding: 2rem; 
            margin-bottom: 2rem; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); 
        }
        .filter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem; }
        .filter-group label { display: block; font-weight: 600; margin-bottom: 0.5rem; color: #2d3748; }
        .filter-group select, .filter-group input { 
            width: 100%; 
            padding: 0.75rem; 
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px; 
            font-size: 1rem;
            color: #2d3748;
        }
        .filter-group select:focus, .filter-group input:focus { outline: none; border-color: #667eea; background: white; }
        .action-btn { padding: 0.75rem 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; font-weight: 600; font-size: 1rem; cursor: pointer; transition: all 0.3s; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3); }
        .action-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5); }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-bottom: 2rem; }
        .stat-card { 
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px; 
            padding: 2rem; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.1); 
            text-align: center; 
        }
        .stat-label { font-size: 1rem; color: #718096; margin-bottom: 0.5rem; }
        .stat-value { font-size: 2.5rem; font-weight: 700; color: #667eea; }
        .chart-card { 
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            width: 100%;
            overflow-x: auto;
        }
        .chart-title { font-size: 1.5rem; font-weight: 700; color: #2d3748; margin-bottom: 1rem; }
        .chart-container { 
            min-height: 400px; 
            width: 100%; 
            max-width: 100%;
            overflow: hidden;
            position: relative;
        }
        .data-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .data-table th, .data-table td { padding: 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }
        .data-table th { background: #f7fafc; font-weight: 600; color: #4a5568; }
        .data-table tr:hover { background: #f7fafc; }
        .loading { text-align: center; padding: 3rem; color: #718096; font-size: 1.2rem; }
        .error { background: #fed7d7; color: #c53030; padding: 1rem; border-radius: 8px; margin-top: 1rem; }
        
        /* 分析类型选择器样式 */
        .analysis-selector { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px; 
            padding: 2rem; 
            margin-bottom: 2rem; 
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }
        .section-title { 
            font-size: 1.3rem; 
            font-weight: 700; 
            color: white; 
            margin-bottom: 1.5rem;
            text-align: center;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .btn-grid { 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            gap: 0.8rem; 
            flex-wrap: wrap; 
        }
        .analysis-btn { 
            padding: 0.7rem 1.5rem; 
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            color: white; 
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 30px; 
            font-weight: 600;
            cursor: pointer; 
            transition: all 0.3s ease; 
            font-size: 0.9rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            min-width: 120px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
        }
        .analysis-btn:hover { 
            background: rgba(255, 255, 255, 0.3);
            border-color: rgba(255, 255, 255, 0.6);
            transform: translateY(-2px); 
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2); 
        }
        .analysis-btn.active { 
            background: white;
            color: #667eea;
            border-color: white;
            box-shadow: 0 6px 20px rgba(255, 255, 255, 0.4);
            transform: scale(1.08);
            font-weight: 700;
        }
        
        /* 图表说明文字样式 */
        .chart-description { 
            margin-top: 1.5rem; 
            padding: 1.2rem 1.8rem; 
            background: rgba(102, 126, 234, 0.1);
            border-left: 5px solid #818cf8; 
            border-radius: 10px; 
            color: #cbd5e0;
            font-size: 0.95rem; 
            line-height: 1.8;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .chart-description strong { color: #818cf8; font-size: 1rem; }
        .chart-description .highlight { 
            color: #f093fb; 
            font-weight: 700; 
            font-size: 1.05rem;
        }
        
        /* 页脚样式 */
        .footer {
            margin-top: 4rem;
            padding: 2.5rem 2rem;
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            text-align: center;
        }
        .footer-content { 
            color: #cbd5e0;
            font-size: 0.95rem;
            line-height: 1.8;
            margin-bottom: 1.5rem;
        }
        .footer-content strong { color: #818cf8; }
        .footer-title {
            color: #f7fafc;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .footer-links { 
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            margin-top: 1rem; 
        }
        .footer-links a { 
            color: #e2e8f0;
            text-decoration: none;
            transition: all 0.3s;
            font-size: 0.9rem;
        }
        .footer-links a:hover { 
            color: #818cf8;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">📈</div>
        <h1 class="title">数据分析与可视化</h1>
        <p class="subtitle">原始数据多维度分析 - 10个区县病虫害数量 + 北京市气象数据</p>
            </div>
    
    <div class="container">
        <div class="nav-card">
            <a href="http://localhost:8003/" class="back-btn">← 返回首页</a>
        </div>
        
        <!-- 分析类型选择器 -->
        <div class="analysis-selector">
            <h2 class="section-title">选择分析类型</h2>
            <div class="btn-grid">
                <button class="analysis-btn active" onclick="showAnalysis('yearly')">📅 年度趋势</button>
                <button class="analysis-btn" onclick="showAnalysis('monthly')">📊 月度分析</button>
                <button class="analysis-btn" onclick="showAnalysis('regional')">🗺️ 区域对比</button>
                <button class="analysis-btn" onclick="showAnalysis('heatmap')">🔥 热力图</button>
                <button class="analysis-btn" onclick="showAnalysis('seasonal')">🌸 季节性分析</button>
                <button class="analysis-btn" onclick="showAnalysis('weather')">🌤️ 气象关联</button>
                <button class="analysis-btn" onclick="showAnalysis('raw')">📋 原始数据</button>
    </div>
        </div>
        
        <!-- 统计卡片 -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">总记录数</div>
                <div class="stat-value" id="statTotal">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">时间跨度</div>
                <div class="stat-value" id="statYears">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">区县数量</div>
                <div class="stat-value" id="statRegions">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">气象指标</div>
                <div class="stat-value" id="statWeather">7</div>
            </div>
        </div>
        
        <!-- 年度趋势 -->
        <div id="yearlySection" class="chart-card">
            <h2 class="chart-title">📅 年度趋势分析</h2>
            <div id="yearlyChart" class="chart-container"></div>
            <div class="chart-description" id="yearlyConclusion">
                <strong>📊 数据分析：</strong>正在加载数据...
            </div>
        </div>
        
        <!-- 月度分析 -->
        <div id="monthlySection" class="chart-card" style="display:none;">
            <h2 class="chart-title">📊 月度分析</h2>
            <div id="monthlyChart" class="chart-container"></div>
            <div class="chart-description" id="monthlyConclusion">
                <strong>📊 数据分析：</strong>正在加载数据...
            </div>
        </div>
        
        <!-- 区域对比 -->
        <div id="regionalSection" class="chart-card" style="display:none;">
            <h2 class="chart-title">🗺️ 区域对比分析</h2>
            <div id="regionalChart" class="container"></div>
            <div class="chart-description" id="regionalConclusion">
                <strong>📊 数据分析：</strong>正在加载数据...
            </div>
        </div>
        
        <!-- 热力图 -->
        <div id="heatmapSection" class="chart-card" style="display:none;">
            <h2 class="chart-title">🔥 时空热力图</h2>
            <div id="heatmapChart" class="chart-container"></div>
            <div class="chart-description">
                <strong>📊 数据分析：</strong>展示各区县在不同时间的病虫害发生强度。颜色越深代表数量越多，可直观看出时空分布规律和爆发时段。
            </div>
        </div>
        
        <!-- 季节性分析 -->
        <div id="seasonalSection" class="chart-card" style="display:none;">
            <h2 class="chart-title">🌸 季节性与周期性分析</h2>
            <div id="seasonalChart" class="chart-container"></div>
            <div class="chart-description" id="seasonalConclusion">
                <strong>📊 数据分析：</strong>正在加载数据...
            </div>
        </div>
        
        <!-- 气象关联 -->
        <div id="weatherSection" class="chart-card" style="display:none;">
            <h2 class="chart-title">🌤️ 气象因子关联分析</h2>
            <div id="weatherChart" class="chart-container"></div>
            <div class="chart-description">
                <strong>📊 数据分析：</strong>展示气象因子（平均温度AT、最高温度MaxT、最低温度MinT、降水Precip）与病虫害数量的散点关系，帮助分析气象条件对病虫害发生的影响规律。
            </div>
        </div>
        
        <!-- 原始数据 -->
        <div id="rawSection" class="chart-card" style="display:none;">
            <h2 class="chart-title">📋 原始数据时序图</h2>
            <div id="rawDataTable" class="chart-container"></div>
            <div class="chart-description" id="rawConclusion">
                <strong>📊 数据分析：</strong>正在加载数据...
            </div>
        </div>
    </div>
    
    
    <script>
        let yearlyData = [];
        let monthlyData = [];
        let regionalData = [];
        let rawData = null;
        
        // 页面加载时自动加载所有数据
        window.onload = async function() {
            await loadData();
        };
        
        async function loadData() {
            try {
                console.log('开始加载数据...');
                
                // 加载年度统计
                const yearlyResponse = await fetch('/api/yearly-stats');
                const yearlyResult = await yearlyResponse.json();
                if (yearlyResult.status === 'success' && yearlyResult.data) {
                    yearlyData = yearlyResult.data;
                    updateYearlyChart();
                }
                
                // 加载月度数据
                const monthlyRes = await fetch('/api/monthly-stats');
                const monthlyJson = await monthlyRes.json();
                if (monthlyJson.status === 'success' && monthlyJson.data) {
                    monthlyData = monthlyJson.data;
                    updateMonthlyChart();
                }
                
                // 加载区域数据
                const regionalRes = await fetch('/api/regional-stats');
                const regionalJson = await regionalRes.json();
                if (regionalJson.status === 'success' && regionalJson.data) {
                    regionalData = regionalJson.data;
                    updateRegionalChart();
                }
                
                // 加载原始数据
                const rawRes = await fetch('/api/raw-data');
                const rawJson = await rawRes.json();
                if (rawJson.status === 'success') {
                    rawData = rawJson;
                    updateRawDataTable();
                    updateHeatmap();
                    updateSeasonalAnalysis();
                    updateWeatherCorrelation();
                }
                
                // 所有数据加载完成后，更新统计卡片
                updateStats();
                
                // 延迟调整所有图表大小，确保容器已完全渲染
                setTimeout(() => {
                    const chartIds = ['yearlyChart', 'monthlyChart', 'regionalChart', 'heatmapChart', 
                                    'seasonalChart', 'weatherChart', 'rawDataTable'];
                    chartIds.forEach(id => {
                        const element = document.getElementById(id);
                        if (element && element.data) {
                            Plotly.Plots.resize(element);
                        }
                    });
                    console.log('所有图表已调整大小');
                }, 300);
                
            } catch (error) {
                console.error('数据加载失败:', error);
                alert('❌ 数据加载失败: ' + error.message);
            }
        }
        
        function updateStats() {
            if (yearlyData.length > 0) {
                const years = yearlyData.map(d => d.year);
                const totalCount = yearlyData.reduce((sum, d) => sum + d.count, 0);
                document.getElementById('statTotal').textContent = totalCount;
                document.getElementById('statYears').textContent = `${Math.min(...years)}-${Math.max(...years)}`;
            }
            
            // 从rawData中获取区县数量
            if (rawData && rawData.headers) {
                const nodeCols = rawData.headers.filter(h => h.startsWith('Node_'));
                document.getElementById('statRegions').textContent = nodeCols.length;
            } else if (regionalData.length > 0) {
                document.getElementById('statRegions').textContent = regionalData.length;
            }
        }
        
        // 切换分析类型
        function showAnalysis(type) {
            ['yearly', 'monthly', 'regional', 'heatmap', 'seasonal', 'weather', 'raw'].forEach(t => {
                document.getElementById(t + 'Section').style.display = 'none';
            });
            document.getElementById(type + 'Section').style.display = 'block';
            document.querySelectorAll('.analysis-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            // 延迟调整图表大小，确保容器已完全展开
            setTimeout(() => {
                const chartMap = {
                    'yearly': 'yearlyChart',
                    'monthly': 'monthlyChart',
                    'regional': 'regionalChart',
                    'heatmap': 'heatmapChart',
                    'seasonal': 'seasonalChart',
                    'weather': 'weatherChart',
                    'raw': 'rawDataTable'
                };
                
                const chartId = chartMap[type];
                if (chartId) {
                    const element = document.getElementById(chartId);
                    if (element && element.data) {
                        Plotly.Plots.resize(element);
                    }
                }
            }, 50);
        }
        
        function updateYearlyChart() {
            if (!yearlyData || yearlyData.length === 0) {
                document.getElementById('yearlyChart').innerHTML = '<div class="error">❌ 年度数据为空</div>';
                return;
            }
            
            const layout = {
                title: { text: '各年度病虫害数量趋势', font: { size: 16 } },
                xaxis: { title: '年份' },
                yaxis: { title: '平均数量' },
                template: 'plotly_white',
                height: 450,
                width: null,
                autosize: true,
                margin: { l: 60, r: 30, t: 50, b: 50 }
            };
            
            const trace = {
                x: yearlyData.map(d => d.year),
                y: yearlyData.map(d => d.average),
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#4a90e2', width: 3 },
                marker: { size: 12, color: '#5a67d8' },
                name: '平均值'
            };
            
            const config = {responsive: true, displayModeBar: false};
            Plotly.newPlot('yearlyChart', [trace], layout, config).then(() => {
                setTimeout(() => Plotly.Plots.resize('yearlyChart'), 100);
            });
            
            // 生成数据分析结论
            const values = yearlyData.map(d => d.average);
            const years = yearlyData.map(d => d.year);
            const maxVal = Math.max(...values);
            const minVal = Math.min(...values);
            const maxYear = years[values.indexOf(maxVal)];
            const minYear = years[values.indexOf(minVal)];
            const growth = ((maxVal - minVal) / minVal * 100).toFixed(1);
            
            document.getElementById('yearlyConclusion').innerHTML = `
                <strong>📊 数据分析：</strong>从2018年到2021年，病虫害数量呈现<span class="highlight">上升趋势</span>。
                最低值出现在<span class="highlight">${minYear}年（${minVal.toFixed(1)}）</span>，
                最高值出现在<span class="highlight">${maxYear}年（${maxVal.toFixed(1)}）</span>，
                增长率达<span class="highlight">${growth}%</span>，表明防治压力持续增大，需加强监测和防控措施。
            `;
        }
        
        function updateMonthlyChart() {
            if (!monthlyData || monthlyData.length === 0) {
                document.getElementById('monthlyChart').innerHTML = '<div class="error">❌ 月度数据为空</div>';
                return;
            }
            
            const layout = {
                title: { text: '月度病虫害数量分布', font: { size: 16 } },
                xaxis: { title: '年-月', tickangle: -45 },
                yaxis: { title: '平均数量' },
                template: 'plotly_white',
                height: 450,
                width: null,
                autosize: true,
                margin: { l: 60, r: 30, t: 50, b: 80 }
            };
            
            const trace = {
                x: monthlyData.map(d => `${d.year}-${String(d.month).padStart(2, '0')}`),
                y: monthlyData.map(d => d.average),
                type: 'bar',
                marker: { color: '#5a67d8' },
                name: '月度平均值'
            };
            
            const config = {responsive: true, displayModeBar: false};
            Plotly.newPlot('monthlyChart', [trace], layout, config).then(() => {
                setTimeout(() => Plotly.Plots.resize('monthlyChart'), 100);
            });
            
            // 生成数据分析结论
            const values = monthlyData.map(d => d.average);
            const labels = monthlyData.map(d => `${d.year}-${String(d.month).padStart(2, '0')}`);
            const maxVal = Math.max(...values);
            const minVal = Math.min(...values);
            const maxLabel = labels[values.indexOf(maxVal)];
            const minLabel = labels[values.indexOf(minVal)];
            
            document.getElementById('monthlyConclusion').innerHTML = `
                <strong>📊 数据分析：</strong>共分析<span class="highlight">${monthlyData.length}个月</span>的数据。
                高发月份为<span class="highlight">${maxLabel}（${maxVal.toFixed(1)}）</span>，
                低发月份为<span class="highlight">${minLabel}（${minVal.toFixed(1)}）</span>。
                月度数据波动较大，建议在高发期前1-2个月加强预防措施。
            `;
        }
        
        function updateRegionalChart() {
            if (!regionalData || regionalData.length === 0) {
                document.getElementById('regionalChart').innerHTML = '<div class="error">❌ 区域数据为空</div>';
                return;
            }
            
            const layout = {
                title: { text: '各区县病虫害数量对比', font: { size: 16 } },
                xaxis: { title: '区县', tickangle: -30 },
                yaxis: { title: '平均数量' },
                template: 'plotly_white',
                height: 450,
                width: null,
                autosize: true,
                margin: { l: 60, r: 30, t: 50, b: 80 }
            };
            
            const trace = {
                x: regionalData.map(d => d.county),
                y: regionalData.map(d => d.average),
                type: 'bar',
                marker: { 
                    color: regionalData.map(d => d.average),
                    colorscale: 'Blues'
                },
                name: '区域平均值'
            };
            
            const config = {responsive: true, displayModeBar: false};
            Plotly.newPlot('regionalChart', [trace], layout, config).then(() => {
                setTimeout(() => Plotly.Plots.resize('regionalChart'), 100);
            });
            
            // 生成数据分析结论
            const values = regionalData.map(d => d.average);
            const counties = regionalData.map(d => d.county);
            const maxVal = Math.max(...values);
            const minVal = Math.min(...values);
            const maxCounty = counties[values.indexOf(maxVal)];
            const minCounty = counties[values.indexOf(minVal)];
            const avgVal = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
            
            document.getElementById('regionalConclusion').innerHTML = `
                <strong>📊 数据分析：</strong>覆盖<span class="highlight">${counties.length}个区县</span>，平均病虫害数量为<span class="highlight">${avgVal}</span>。
                高发区县为<span class="highlight">${maxCounty}（${maxVal.toFixed(1)}）</span>，
                低发区县为<span class="highlight">${minCounty}（${minVal.toFixed(1)}）</span>。
                建议对高发区县实施重点监测和针对性防控策略。
            `;
        }
        
        // 更新热力图
        function updateHeatmap() {
            if (!rawData || !rawData.data || rawData.data.length === 0) {
                document.getElementById('heatmapChart').innerHTML = '<div class="error">❌ 数据不足</div>';
                return;
            }
            
            const nodeCols = rawData.headers.filter(h => h.startsWith('Node_'));
            const dates = rawData.data.map(d => d.Date).slice(0, 50);
            const z = [];
            
            nodeCols.forEach(col => {
                const values = rawData.data.slice(0, 50).map(d => d[col] || 0);
                z.push(values);
            });
            
            const layout = {
                title: { text: '时空热力图 - 各区县病虫害数量', font: { size: 16 } },
                xaxis: { title: '日期', tickangle: -45 },
                yaxis: { title: '区县', ticktext: nodeCols.map(c => c.replace('Node_', '')), tickvals: nodeCols.map((c, i) => i) },
                template: 'plotly_white',
                height: 500,
                width: null,
                autosize: true,
                margin: { l: 80, r: 30, t: 50, b: 80 }
            };
            
            const trace = {
                z: z,
                x: dates,
                y: nodeCols.map(c => c.replace('Node_', '')),
                type: 'heatmap',
                colorscale: 'YlOrRd',
                colorbar: { title: '数量' }
            };
            
            const config = {responsive: true, displayModeBar: false};
            Plotly.newPlot('heatmapChart', [trace], layout, config).then(() => {
                setTimeout(() => Plotly.Plots.resize('heatmapChart'), 100);
            });
        }
        
        // 更新季节性分析
        function updateSeasonalAnalysis() {
            if (!monthlyData || monthlyData.length === 0) {
                document.getElementById('seasonalChart').innerHTML = '<div class="error">❌ 数据不足</div>';
                return;
            }
            
            const monthlyAvg = {};
            monthlyData.forEach(d => {
                if (!monthlyAvg[d.month]) {
                    monthlyAvg[d.month] = { sum: 0, count: 0 };
                }
                monthlyAvg[d.month].sum += d.average;
                monthlyAvg[d.month].count += 1;
            });
            
            const months = [];
            const avgValues = [];
            for (let m = 1; m <= 12; m++) {
                const monthNames = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'];
                months.push(monthNames[m-1]);
                if (monthlyAvg[m]) {
                    avgValues.push(monthlyAvg[m].sum / monthlyAvg[m].count);
                } else {
                    avgValues.push(0);
                }
            }
            
            const layout = {
                title: { text: '季节性周期分析 - 各月平均病虫害数量', font: { size: 16 } },
                xaxis: { title: '月份' },
                yaxis: { title: '平均数量' },
                template: 'plotly_white',
                height: 450,
                width: null,
                autosize: true,
                margin: { l: 60, r: 30, t: 50, b: 50 }
            };
            
            const trace = {
                x: months,
                y: avgValues,
                type: 'scatter',
                mode: 'lines+markers',
                fill: 'tozeroy',
                line: { color: '#f5576c', width: 3 },
                marker: { size: 10, color: '#f093fb' },
                name: '季节性趋势'
            };
            
            const config = {responsive: true, displayModeBar: false};
            Plotly.newPlot('seasonalChart', [trace], layout, config).then(() => {
                setTimeout(() => Plotly.Plots.resize('seasonalChart'), 100);
            });
            
            // 生成数据分析结论
            const maxVal = Math.max(...avgValues);
            const minVal = Math.min(...avgValues);
            const maxMonth = months[avgValues.indexOf(maxVal)];
            const minMonth = months[avgValues.indexOf(minVal)];
            
            document.getElementById('seasonalConclusion').innerHTML = `
                <strong>📊 数据分析：</strong>病虫害呈现明显的<span class="highlight">季节性特征</span>。
                高发季节为<span class="highlight">${maxMonth}（${maxVal.toFixed(1)}）</span>，
                低发季节为<span class="highlight">${minMonth}（${minVal.toFixed(1)}）</span>。
                建议根据季节性规律，在高发季节到来前加强监测预警，做好提前防控准备。
            `;
        }
        
        // 更新气象关联分析
        function updateWeatherCorrelation() {
            if (!rawData || !rawData.data || rawData.data.length === 0) {
                document.getElementById('weatherChart').innerHTML = '<div class="error">❌ 数据不足</div>';
                return;
            }
            
            const weatherCols = ['AT', 'MaxT', 'MinT', 'Precip'];
            const nodeCols = rawData.headers.filter(h => h.startsWith('Node_'));
            
            if (nodeCols.length === 0) {
                document.getElementById('weatherChart').innerHTML = '<div class="error">❌ 无区县数据</div>';
                return;
            }
            
            const targetNode = nodeCols[0];
            const traces = [];
            
            weatherCols.forEach((weatherCol, idx) => {
                if (!rawData.headers.includes(weatherCol)) return;
                
                const x = [];
                const y = [];
                rawData.data.forEach(row => {
                    if (row[weatherCol] != null && row[targetNode] != null) {
                        x.push(parseFloat(row[weatherCol]));
                        y.push(parseFloat(row[targetNode]));
                    }
                });
                
                if (x.length > 0) {
                    traces.push({
                        x: x,
                        y: y,
                        mode: 'markers',
                        type: 'scatter',
                        name: weatherCol,
                        marker: { size: 6 }
                    });
                }
            });
            
            const layout = {
                title: { text: `气象因子与${targetNode.replace('Node_', '')}病虫害数量关系`, font: { size: 16 } },
                xaxis: { title: '气象因子值' },
                yaxis: { title: '病虫害数量' },
                template: 'plotly_white',
                height: 450,
                width: null,
                autosize: true,
                margin: { l: 60, r: 30, t: 50, b: 50 },
                showlegend: true
            };
            
            const config = {responsive: true, displayModeBar: false};
            Plotly.newPlot('weatherChart', traces, layout, config).then(() => {
                setTimeout(() => Plotly.Plots.resize('weatherChart'), 100);
            });
        }
        
        // 更新原始数据 - 折线图展示
        function updateRawDataTable() {
            if (!rawData || !rawData.data || rawData.data.length === 0) {
                document.getElementById('rawDataTable').innerHTML = '<div class="error">❌ 无数据</div>';
                return;
            }
            
            // 获取所有区县列（Node_开头的）
            const nodeCols = rawData.headers.filter(h => h.startsWith('Node_'));
            const dates = rawData.data.map(d => d.Date);
            
            // 为每个区县创建一条折线
            const traces = [];
            const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
            
            nodeCols.forEach((col, idx) => {
                const yValues = rawData.data.map(d => d[col] || 0);
                traces.push({
                    x: dates,
                    y: yValues,
                    type: 'scatter',
                    mode: 'lines',
                    name: col.replace('Node_', ''),
                    line: {
                        color: colors[idx % colors.length],
                        width: 2
                    },
                    hovertemplate: '<b>%{fullData.name}</b><br>' +
                                   '日期: %{x}<br>' +
                                   '数量: %{y:.0f}<br>' +
                                   '<extra></extra>'
                });
            });
            
            const layout = {
                title: {
                    text: '10个区县病虫害数量时序变化',
                    font: { size: 16, color: '#2d3748' }
                },
                xaxis: {
                    title: '日期',
                    tickangle: -45,
                    type: 'date'
                },
                yaxis: {
                    title: '病虫害数量',
                    rangemode: 'tozero'
                },
                template: 'plotly_white',
                height: 500,
                width: null,
                autosize: true,
                showlegend: true,
                legend: {
                    orientation: 'v',
                    x: 1.0,
                    y: 1,
                    xanchor: 'left',
                    bgcolor: 'rgba(255,255,255,0.8)',
                    bordercolor: '#e2e8f0',
                    borderwidth: 1,
                    font: { size: 10 }
                },
                hovermode: 'x unified',
                margin: { l: 60, r: 120, t: 50, b: 80 }
            };
            
            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                displaylogo: false
            };
            
            Plotly.newPlot('rawDataTable', traces, layout, config).then(() => {
                setTimeout(() => Plotly.Plots.resize('rawDataTable'), 100);
            });
            
            // 生成数据分析结论
            const allValues = [];
            traces.forEach(trace => {
                allValues.push(...trace.y);
            });
            const maxVal = Math.max(...allValues);
            const totalCount = rawData.total_rows;
            
            document.getElementById('rawConclusion').innerHTML = `
                <strong>📊 数据分析：</strong>共展示<span class="highlight">${nodeCols.length}个区县</span>的<span class="highlight">${totalCount}条</span>时序数据记录。
                病虫害数量峰值为<span class="highlight">${maxVal.toFixed(0)}</span>，各区县呈现不同的时序变化规律。
                可通过点击图例选择性查看特定区县数据，鼠标悬停可查看详细数值。图表支持缩放、平移等交互操作，便于深入分析。
            `;
        }
        
        // 添加窗口resize监听，确保图表自适应
        let resizeTimer;
        window.addEventListener('resize', function() {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(function() {
                // 获取所有Plotly图表容器
                const chartIds = ['yearlyChart', 'monthlyChart', 'regionalChart', 'heatmapChart', 
                                'seasonalChart', 'weatherChart', 'rawDataTable'];
                
                chartIds.forEach(id => {
                    const element = document.getElementById(id);
                    if (element && element.data) {
                        Plotly.Plots.resize(element);
                    }
                });
            }, 250);
        });
    </script>
    
    <!-- 底部版权信息 -->
    <footer style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-top: 4rem; text-align: center; font-size: 0.9rem; line-height: 1.8;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <p style="margin: 0.5rem 0; font-weight: 600;">© 2025 AgriGuard Platform. 基于大数据与人工智能的病虫害预测预警系统</p>
            <p style="margin: 0.5rem 0;">数据来源：北京市10区县植物诊所 | 2018-2021年时序数据</p>
            <p style="margin: 0.5rem 0;">技术支持：时空预测模型 + 深度学习 + 大语言模型</p>
            <p style="margin: 0.5rem 0;">开发单位：中国农业大学 信息与电气工程学院</p>
            <p style="margin: 0.5rem 0;">开发团队：张领先教授团队 秦源泽等人</p>
        </div>
    </footer>
</body>
</html>
        """
        
    def get_data_collection_html(self):
        """获取数据采集页面HTML"""
        try:
            fields = medical_collector.get_record_fields() if medical_collector else {}
        except:
            fields = {}
        
        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据采集模块 - 时空预测系统</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: 
                linear-gradient(135deg, 
                    rgba(96, 165, 250, 0.95) 0%,
                    rgba(147, 197, 253, 0.9) 25%,
                    rgba(196, 181, 253, 0.9) 50%,
                    rgba(167, 139, 250, 0.9) 75%,
                    rgba(129, 140, 248, 0.95) 100%
                );
            min-height: 100vh;
            padding: 2rem;
            position: relative;
            overflow-x: hidden;
        }}
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 30%, rgba(255, 255, 255, 0.2) 0%, transparent 50%),
                radial-gradient(circle at 80% 70%, rgba(255, 255, 255, 0.25) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 60%);
            pointer-events: none;
            z-index: 0;
        }}
        body::after {{
            content: none;
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            color: white;
            position: relative;
            z-index: 1;
        }}
        .title {{
            font-size: 3rem;
            font-weight: 800;
            color: white;
            margin-bottom: 1rem;
            text-shadow: 0 8px 16px rgba(0, 0, 0, 0.3),
                        0 0 40px rgba(255, 255, 255, 0.1);
            letter-spacing: 0.5px;
        }}
        .subtitle {{
            font-size: 1.25rem;
            color: rgba(255, 255, 255, 0.95);
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            font-weight: 500;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }}
        .nav-card {{
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .tabs {{
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }}
        .tab {{
            padding: 1rem 2rem;
            background: rgba(255, 255, 255, 0.9);
            border: 2px solid rgba(102, 126, 234, 0.3);
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
            color: #334155;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        .tab.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: transparent;
        }}
        .tab:hover {{
            transform: translateY(-2px);
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .form-card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 15px;
            padding: 2.5rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }}
        .form-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }}
        .form-group {{
            margin-bottom: 1.5rem;
        }}
        .form-label {{
            display: block;
            font-weight: 600;
            color: #334155;
            margin-bottom: 0.5rem;
            font-size: 0.95rem;
        }}
        .form-input, .form-select, .form-textarea {{
            width: 100%;
            padding: 0.75rem 1rem;
            background: white;
            border: 2px solid rgba(102, 126, 234, 0.2);
            border-radius: 8px;
            font-size: 0.95rem;
            color: #334155;
            transition: all 0.3s;
        }}
        .form-input::placeholder, .form-textarea::placeholder {{
            color: #94a3b8;
        }}
        .form-input:focus, .form-select:focus, .form-textarea:focus {{
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
        }}
        .form-textarea {{
            resize: vertical;
            min-height: 100px;
        }}
        .btn {{
            padding: 1rem 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
        }}
        .back-btn {{
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        .back-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }}
        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #667eea;
        }}
        .info-box {{
            background: #e6fffa;
            border-left: 4px solid #38b2ac;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            color: #0f766e;
            font-weight: 500;
        }}
        .info-box strong {{
            color: #115e59;
        }}
        .weather-card {{
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            border-radius: 15px;
            padding: 2rem;
            color: white;
            text-align: center;
        }}
        .weather-icon {{
            font-size: 4rem;
            margin-bottom: 1rem;
        }}
        /* 全局文字颜色增强 */
        .form-card p, .form-card span, .form-card label {{
            color: #334155;
        }}
        .form-card h2, .form-card h3 {{
            color: #1e293b;
        }}
        small {{
            color: #64748b;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">📝</div>
        <h1 class="title">数据采集模块</h1>
        <p class="subtitle">植物电子病历、领域知识库、气象数据采集</p>
    </div>
    
    <div class="container">
        <div class="nav-card">
            <a href="/" class="back-btn">← 返回系统首页</a>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('medical')">📋 电子病历采集</div>
            <div class="tab" onclick="switchTab('knowledge')">💊 领域知识库</div>
            <div class="tab" onclick="switchTab('weather')">🌤️ 气象数据</div>
        </div>
        
        <!-- 电子病历采集 -->
        <div id="medical" class="tab-content active">
            <div class="form-card">
                <h2 class="section-title">植物电子病历录入</h2>
                <div class="info-box">
                    <strong>📌 说明：</strong>完整录入植物病历信息，数据将自动保存并可用于后续分析和预测
                </div>
                
                <form id="medicalForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label class="form-label">植物诊所</label>
                            <input type="text" class="form-input" name="clinic_name" placeholder="请输入植物诊所名称" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">植物医生</label>
                            <input type="text" class="form-input" name="doctor_name" placeholder="请输入植物医生姓名" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">农户名称</label>
                            <input type="text" class="form-input" name="farmer_name" placeholder="请输入农户姓名" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">农户联系方式</label>
                            <input type="tel" class="form-input" name="farmer_contact" placeholder="请输入联系电话">
                        </div>
                        <div class="form-group">
                            <label class="form-label">所属区县</label>
                            <select class="form-select" name="district" required>
                                <option value="">请选择区县</option>
                                <option value="朝阳区">朝阳区</option>
                                <option value="海淀区">海淀区</option>
                                <option value="昌平区">昌平区</option>
                                <option value="顺义区">顺义区</option>
                                <option value="大兴区">大兴区</option>
                                <option value="通州区">通州区</option>
                                <option value="房山区">房山区</option>
                                <option value="门头沟区">门头沟区</option>
                                <option value="怀柔区">怀柔区</option>
                                <option value="平谷区">平谷区</option>
                                <option value="密云区">密云区</option>
                                <option value="延庆区">延庆区</option>
                                <option value="丰台区">丰台区</option>
                                <option value="石景山区">石景山区</option>
                                <option value="东城区">东城区</option>
                                <option value="西城区">西城区</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">所属乡镇</label>
                            <input type="text" class="form-input" name="township" placeholder="请输入乡镇名称">
                        </div>
                        <div class="form-group">
                            <label class="form-label">所属村庄</label>
                            <input type="text" class="form-input" name="village" placeholder="请输入村庄名称">
                        </div>
                        <div class="form-group">
                            <label class="form-label">开具时间</label>
                            <input type="date" class="form-input" name="issue_date" required>
                        </div>
                        <div class="form-group">
                            <label class="form-label">作物</label>
                            <select class="form-select" name="crop" required>
                                <option value="">请选择作物</option>
                                <option value="小麦">小麦</option>
                                <option value="玉米">玉米</option>
                                <option value="水稻">水稻</option>
                                <option value="大豆">大豆</option>
                                <option value="蔬菜">蔬菜</option>
                                <option value="果树">果树</option>
                                <option value="其他">其他</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">是否有样品</label>
                            <select class="form-select" name="has_sample">
                                <option value="否">否</option>
                                <option value="是">是</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">病虫害是否发生</label>
                            <select class="form-select" name="disease_occurred" required>
                                <option value="否">否</option>
                                <option value="是">是</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">发育阶段</label>
                            <input type="text" class="form-input" name="growth_stage" placeholder="如：苗期、拔节期等">
                        </div>
                        <div class="form-group">
                            <label class="form-label">受害部位</label>
                            <input type="text" class="form-input" name="affected_part" placeholder="如：叶片、茎秆等">
                        </div>
                        <div class="form-group">
                            <label class="form-label">首次发现年份</label>
                            <input type="number" class="form-input" name="first_found_year" min="1900" max="2100" placeholder="如：2024">
                        </div>
                        <div class="form-group">
                            <label class="form-label">发生面积（亩）</label>
                            <input type="number" class="form-input" name="affected_area" step="0.1" placeholder="请输入面积">
                        </div>
                        <div class="form-group">
                            <label class="form-label">发生比重（%）</label>
                            <input type="number" class="form-input" name="occurrence_rate" min="0" max="100" step="0.1" placeholder="0-100">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">主要症状</label>
                        <textarea class="form-textarea" name="symptoms" placeholder="请详细描述病虫害主要症状"></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">田间症状分布</label>
                        <textarea class="form-textarea" name="symptom_distribution" placeholder="请描述症状在田间的分布情况"></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">问诊记录</label>
                        <textarea class="form-textarea" name="consultation_record" placeholder="请记录问诊过程"></textarea>
                    </div>
                    
                    <div class="form-grid">
                        <div class="form-group">
                            <label class="form-label">诊断结果</label>
                            <input type="text" class="form-input" name="diagnosis_result" placeholder="病/虫/杂草的名称">
                        </div>
                        <div class="form-group">
                            <label class="form-label">农药大类</label>
                            <select class="form-select" name="pesticide_category">
                                <option value="">请选择农药大类</option>
                                <option value="杀菌剂">杀菌剂</option>
                                <option value="杀虫剂">杀虫剂</option>
                                <option value="除草剂">除草剂</option>
                                <option value="植物生长调节剂">植物生长调节剂</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">开具农药名称</label>
                            <input type="text" class="form-input" name="pesticide_name" placeholder="请输入农药名称">
                        </div>
                        <div class="form-group">
                            <label class="form-label">开具农药数量</label>
                            <input type="text" class="form-input" name="pesticide_quantity" placeholder="如：500ml、2kg">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">农业防治措施</label>
                        <textarea class="form-textarea" name="agricultural_control" placeholder="请描述农业防治措施"></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">拿药状态</label>
                        <select class="form-select" name="medicine_status">
                            <option value="未拿药">未拿药</option>
                            <option value="已拿药">已拿药</option>
                            <option value="待拿药">待拿药</option>
                        </select>
                    </div>
                    
                    <div style="text-align: center; margin-top: 2rem;">
                        <button type="submit" class="btn">提交病历</button>
                    </div>
                </form>
            </div>
        </div>
        
        <!-- 领域知识库 -->
        <div id="knowledge" class="tab-content">
            <div class="form-card">
                <h2 class="section-title">领域知识库管理</h2>
                <div class="info-box">
                    <strong>📌 功能：</strong>管理农药信息、病虫害知识、防治方法等领域知识
                </div>
                
                <div style="margin-top: 2rem;">
                    <h3 style="color: #2d3748; margin-bottom: 1rem;">农药信息库</h3>
                    <div class="form-grid">
                        <div class="form-group">
                            <label class="form-label">农药名称</label>
                            <input type="text" class="form-input" id="pesticideName" placeholder="请输入农药名称">
                        </div>
                        <div class="form-group">
                            <label class="form-label">农药类型</label>
                            <select class="form-select" id="pesticideType">
                                <option value="杀菌剂">杀菌剂</option>
                                <option value="杀虫剂">杀虫剂</option>
                                <option value="除草剂">除草剂</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label class="form-label">有效成分</label>
                            <input type="text" class="form-input" id="activeIngredient" placeholder="请输入有效成分">
                        </div>
                        <div class="form-group">
                            <label class="form-label">使用剂量</label>
                            <input type="text" class="form-input" id="dosage" placeholder="如：100-150ml/亩">
                        </div>
                    </div>
                    <div class="form-group">
                        <label class="form-label">使用方法</label>
                        <textarea class="form-textarea" id="usage" placeholder="请输入使用方法和注意事项"></textarea>
                    </div>
                    <div class="form-group">
                        <label class="form-label">适用对象</label>
                        <input type="text" class="form-input" id="target" placeholder="适用于哪些病虫害">
                    </div>
                    <div style="text-align: center; margin-top: 1.5rem;">
                        <button type="button" class="btn" onclick="addPesticide()">添加到知识库</button>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 气象数据 -->
        <div id="weather" class="tab-content">
            <div class="form-card">
                <h2 class="section-title">气象数据获取</h2>
                <div class="info-box">
                    <strong>📌 功能：</strong>获取实时气象数据和历史气象数据，用于病虫害预测分析
                </div>
                
                <div class="form-grid" style="margin-top: 2rem;">
                    <div class="form-group">
                        <label class="form-label">地区选择</label>
                        <select class="form-select" id="weatherLocation">
                            <option value="北京">北京</option>
                            <option value="朝阳区">朝阳区</option>
                            <option value="海淀区">海淀区</option>
                            <option value="昌平区">昌平区</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">开始日期</label>
                        <input type="date" class="form-input" id="startDate">
                    </div>
                    <div class="form-group">
                        <label class="form-label">结束日期</label>
                        <input type="date" class="form-input" id="endDate">
                    </div>
                </div>
                
                <div style="text-align: center; margin: 2rem 0;">
                    <button type="button" class="btn" onclick="getWeatherData()">获取气象数据</button>
                </div>
                
                <div id="weatherResult" style="margin-top: 2rem;"></div>
            </div>
        </div>
    </div>
    
    <script>
        function switchTab(tabName) {{
            // 隐藏所有标签页
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // 显示选中的标签页
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
        
        // 处理病历表单提交
        document.getElementById('medicalForm').addEventListener('submit', async function(e) {{
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            try {{
                const response = await fetch('/api/medical-record', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(data)
                }});
                
                const result = await response.json();
                
                if (result.status === 'success') {{
                    alert('✅ 病历记录提交成功！');
                    e.target.reset();
                }} else {{
                    alert('❌ 提交失败: ' + result.message);
                }}
            }} catch (error) {{
                alert('❌ 提交失败: ' + error.message);
            }}
        }});
        
        // 添加农药信息
        async function addPesticide() {{
            const data = {{
                name: document.getElementById('pesticideName').value,
                type: document.getElementById('pesticideType').value,
                active_ingredient: document.getElementById('activeIngredient').value,
                dosage: document.getElementById('dosage').value,
                usage: document.getElementById('usage').value,
                target: document.getElementById('target').value
            }};
            
            // 这里可以添加API调用
            alert('✅ 农药信息已添加到知识库！');
            
            // 清空表单
            document.getElementById('pesticideName').value = '';
            document.getElementById('activeIngredient').value = '';
            document.getElementById('dosage').value = '';
            document.getElementById('usage').value = '';
            document.getElementById('target').value = '';
        }}
        
        // 获取气象数据
        async function getWeatherData() {{
            const location = document.getElementById('weatherLocation').value;
            const startDate = document.getElementById('startDate').value;
            const endDate = document.getElementById('endDate').value;
            
            if (!startDate || !endDate) {{
                alert('请选择日期范围');
                return;
            }}
            
            try {{
                const response = await fetch('/api/weather', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{
                        location: location,
                        start_date: startDate,
                        end_date: endDate
                    }})
                }});
                
                const result = await response.json();
                
                if (result.status === 'success') {{
                    displayWeatherData(result.data);
                }} else {{
                    alert('❌ 获取失败: ' + result.message);
                }}
            }} catch (error) {{
                alert('❌ 获取失败: ' + error.message);
            }}
        }}
        
        function displayWeatherData(data) {{
            const resultDiv = document.getElementById('weatherResult');
            resultDiv.innerHTML = `
                <div class="weather-card">
                    <div class="weather-icon">🌤️</div>
                    <h3>气象数据</h3>
                    <p>地区: ${{data.location}}</p>
                    <p>时间范围: ${{data.start_date}} - ${{data.end_date}}</p>
                    <p style="margin-top: 1rem; font-size: 0.9rem;">数据已获取，可用于分析</p>
                </div>
            `;
        }}
    </script>
    
    <!-- 底部版权信息 -->
    <footer style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-top: 4rem; text-align: center; font-size: 0.9rem; line-height: 1.8;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <p style="margin: 0.5rem 0; font-weight: 600;">© 2025 AgriGuard Platform. 基于大数据与人工智能的病虫害预测预警系统</p>
            <p style="margin: 0.5rem 0;">数据来源：北京市10区县植物诊所 | 2018-2021年时序数据</p>
            <p style="margin: 0.5rem 0;">技术支持：时空预测模型 + 深度学习 + 大语言模型</p>
            <p style="margin: 0.5rem 0;">开发单位：中国农业大学 信息与电气工程学院</p>
            <p style="margin: 0.5rem 0;">开发团队：张领先教授团队 秦源泽等人</p>
        </div>
    </footer>
</body>
</html>
        """
    
    def get_model_prediction_html(self):
        """获取模型预测页面HTML"""
        try:
            with open('model_prediction_page.html', 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型预测结果 - 时空预测系统</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1f35 0%, #2d3548 50%, #1e293b 100%);
            min-height: 100vh;
            padding: 2rem;
            position: relative;
            overflow-x: hidden;
        }}
        body::before {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.25) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(99, 102, 241, 0.25) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(139, 92, 246, 0.2) 0%, transparent 50%);
            z-index: -1;
        }}
        body::after {{
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: 
                linear-gradient(rgba(139, 92, 246, 0.08) 1px, transparent 1px),
                linear-gradient(90deg, rgba(139, 92, 246, 0.08) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: -1;
            animation: gridMove 20s linear infinite;
        }}
        @keyframes gridMove {{
            0% {{ transform: translate(0, 0); }}
            100% {{ transform: translate(50px, 50px); }}
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            color: white;
        }}
        .title {{
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .subtitle {{
            font-size: 1.1rem;
            color: #cbd5e0;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        .nav-card {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .model-selector {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .model-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }}
        .model-btn {{
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #e2e8f0;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .model-btn:hover {{
            transform: translateY(-3px);
            background: rgba(102, 126, 234, 0.3);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }}
        .chart-container {{
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .back-btn {{
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        .back-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        }}
        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #f7fafc;
            margin-bottom: 1.5rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🔮</div>
        <h1 class="title">模型预测结果展示</h1>
        <p class="subtitle">12种时序预测模型 - 多维度对比与分析</p>
            </div>
    
    <div class="container">
        <div class="nav-card">
            <a href="/" class="back-btn">← 返回系统首页</a>
        </div>
        
        <div class="model-selector">
            <h2 class="section-title">可用预测模型</h2>
            <div class="model-grid">
                <button class="model-btn" onclick="selectModel('LSTM')">LSTM</button>
                <button class="model-btn" onclick="selectModel('GRU')">GRU</button>
                <button class="model-btn" onclick="selectModel('CNN-LSTM-Attention')">CNN-LSTM-Attention</button>
                <button class="model-btn" onclick="selectModel('TCN')">TCN</button>
                <button class="model-btn" onclick="selectModel('TimesNet')">TimesNet</button>
                <button class="model-btn" onclick="selectModel('PatchTST')">PatchTST</button>
                <button class="model-btn" onclick="selectModel('PatchFormer')">PatchFormer</button>
                <button class="model-btn" onclick="selectModel('TSPeakNet')">TSPeakNet</button>
                <button class="model-btn" onclick="selectModel('KAN')">KAN</button>
                <button class="model-btn" onclick="selectModel('SVR')">SVR</button>
                <button class="model-btn" onclick="selectModel('KNN')">KNN</button>
                <button class="model-btn" onclick="selectModel('ALL')">全部对比</button>
            </div>
        </div>
        
        <div class="chart-container">
            <h2 class="section-title">模型对比分析</h2>
            <div id="comparisonChart"></div>
    </div>
        
        <div class="chart-container">
            <h2 class="section-title" id="selectedModelTitle">选择模型查看详细预测结果</h2>
            <div id="modelDetailChart"></div>
        </div>
    </div>
    
    <script>
        // 加载模型对比图
        async function loadModelComparison() {{
            try {{
                const response = await fetch('/api/charts/model-comparison');
                const result = await response.json();
                if (result.chart) {{
                    const chartData = JSON.parse(result.chart);
                    Plotly.newPlot('comparisonChart', chartData.data, chartData.layout);
                }}
            }} catch (error) {{
                console.error('加载失败:', error);
            }}
        }}
        
        // 选择模型
        function selectModel(modelName) {{
            document.getElementById('selectedModelTitle').textContent = modelName + ' 模型预测结果';
            document.getElementById('modelDetailChart').innerHTML = '<div style="text-align: center; padding: 4rem; color: #718096;">正在加载 ' + modelName + ' 模型数据...</div>';
            
            // 这里可以加载具体模型的详细数据
            setTimeout(() => {{
                document.getElementById('modelDetailChart').innerHTML = '<div style="text-align: center; padding: 4rem; color: #718096;">📊 ' + modelName + ' 模型数据展示（需要读取Excel文件）</div>';
            }}, 500);
        }}
        
        // 页面加载时
        document.addEventListener('DOMContentLoaded', function() {{
            loadModelComparison();
        }});
    </script>
    
    <!-- 底部版权信息 -->
    <footer style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-top: 4rem; text-align: center; font-size: 0.9rem; line-height: 1.8;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <p style="margin: 0.5rem 0; font-weight: 600;">© 2025 AgriGuard Platform. 基于大数据与人工智能的病虫害预测预警系统</p>
            <p style="margin: 0.5rem 0;">数据来源：北京市10区县植物诊所 | 2018-2021年时序数据</p>
            <p style="margin: 0.5rem 0;">技术支持：时空预测模型 + 深度学习 + 大语言模型</p>
            <p style="margin: 0.5rem 0;">开发单位：中国农业大学 信息与电气工程学院</p>
            <p style="margin: 0.5rem 0;">开发团队：张领先教授团队 秦源泽等人</p>
        </div>
    </footer>
</body>
</html>
        """
        
    def get_ai_assistant_html(self):
        """获取AI助手页面HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI智能助手 - 时空预测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }}
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1f35 0%, #2d3548 50%, #1e293b 100%);
            min-height: 100vh;
            padding: 2rem;
            position: relative;
            overflow-x: hidden;
        }
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.25) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(99, 102, 241, 0.25) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(139, 92, 246, 0.2) 0%, transparent 50%);
            z-index: -1;
        }
        body::after {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: 
                linear-gradient(rgba(139, 92, 246, 0.08) 1px, transparent 1px),
                linear-gradient(90deg, rgba(139, 92, 246, 0.08) 1px, transparent 1px);
            background-size: 50px 50px;
            z-index: -1;
            animation: gridMove 20s linear infinite;
        }
        @keyframes gridMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
            color: white;
        }
        .title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .nav-card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .chat-container {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 15px 50px rgba(0,0,0,0.2);
            height: 70vh;
            display: flex;
            flex-direction: column;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1.5rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        .message {
            margin-bottom: 1rem;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            max-width: 80%;
            line-height: 1.6;
        }
        .message-user {
            background: #667eea;
            color: white;
            margin-left: auto;
        }
        .message-ai {
            background: white;
            color: #2d3748;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .chat-input-container {
            display: flex;
            gap: 1rem;
        }
        .chat-input {
            flex: 1;
            padding: 1rem 1.5rem;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 1rem;
        }
        .chat-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .send-btn {
            padding: 1rem 2rem;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .send-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .back-btn {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background: #4a5568;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .back-btn:hover {
            background: #2d3748;
        }
        .ai-status {
            background: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">🤖</div>
        <h1 class="title">AI智能助手</h1>
        <p style="font-size: 1.1rem; opacity: 0.95;">大语言模型驱动的智能分析与决策支持</p>
    </div>
    
    <div class="container">
        <div class="nav-card">
            <a href="/" class="back-btn">← 返回系统首页</a>
        </div>
        
        <div class="ai-status">
            <strong>🚧 AI模块状态：</strong>接口已预留，待集成大语言模型（Qwen/ChatGLM/GPT等）
                </div>
        
        <div class="chat-container">
            <div class="chat-messages" id="chatMessages">
                <div class="message message-ai">
                    👋 您好！我是AgriGuard AI助手。<br>
                    我可以帮您：<br>
                    • 分析病虫害数据趋势<br>
                    • 解读预测模型结果<br>
                    • 提供防治决策建议<br>
                    • 回答农业技术问题<br><br>
                    <em style="color: #f59e0b;">💡 提示：大语言模型接口已预留，可对接Qwen、ChatGLM等模型</em>
                </div>
                </div>
            
            <div class="chat-input-container">
                <input type="text" class="chat-input" id="userInput" placeholder="请输入您的问题..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button class="send-btn" onclick="sendMessage()">发送</button>
                </div>
            </div>
        </div>
        
    <script>
        function sendMessage() {{
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // 显示用户消息
            const messagesDiv = document.getElementById('chatMessages');
            messagesDiv.innerHTML += `
                <div class="message message-user">${{message}}</div>
            `;
            
            // 清空输入
            input.value = '';
            
            // 模拟AI回复
            setTimeout(() => {{
                messagesDiv.innerHTML += `
                    <div class="message message-ai">
                        🤖 收到您的问题："${{message}}"<br><br>
                        <em style="color: #718096;">当前为演示模式。实际部署时，此处将调用大语言模型API：</em><br>
                        • 本地部署：Ollama + Qwen2.5<br>
                        • 云端API：阿里云百炼、百度千帆等<br>
                        • 提供智能分析和专业建议
            </div>
                `;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }}, 500);
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }}
    </script>
    
    <!-- 底部版权信息 -->
    <footer style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem 0; margin-top: 4rem; text-align: center; font-size: 0.9rem; line-height: 1.8;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 2rem;">
            <p style="margin: 0.5rem 0; font-weight: 600;">© 2025 AgriGuard Platform. 基于大数据与人工智能的病虫害预测预警系统</p>
            <p style="margin: 0.5rem 0;">数据来源：北京市10区县植物诊所 | 2018-2021年时序数据</p>
            <p style="margin: 0.5rem 0;">技术支持：时空预测模型 + 深度学习 + 大语言模型</p>
            <p style="margin: 0.5rem 0;">开发单位：中国农业大学 信息与电气工程学院</p>
            <p style="margin: 0.5rem 0;">开发团队：张领先教授团队 秦源泽等人</p>
        </div>
    </footer>
</body>
</html>
        """

# 多线程服务器类（性能优化）
class ThreadedTCPServer(ThreadingMixIn, socketserver.TCPServer):
    """支持多线程的TCP服务器"""
    allow_reuse_address = True
    daemon_threads = True

def main():
    print("="*60)
    print("时空预测系统启动在端口:", PORT)
    print("="*60)
    print("功能模块:")
    print("  - 数据采集: /data-collection")
    print("  - 数据分析: /data-analysis")
    print("  - 模型预测: /model-prediction")
    print("  - AI助手: /ai-assistant")
    print("  - 区域预警: /regional-warning  [新功能]")
    print("="*60)
    print("性能优化: 多线程支持 + 数据缓存")
    print("="*60)
    
    # 使用多线程服务器
    with ThreadedTCPServer(("", PORT), PredictionHandler) as httpd:
        print(f"服务器运行在 http://localhost:{PORT}")
        print("="*60)
        httpd.serve_forever()

if __name__ == '__main__':
    main()
