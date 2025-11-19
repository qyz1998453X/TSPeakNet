#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集模块
包括：植物电子病历采集、领域知识库、气象数据获取
"""

import json
import os
from datetime import datetime

class MedicalRecordCollector:
    """植物电子病历数据采集器"""
    
    def __init__(self, data_file="medical_records.json"):
        self.data_file = data_file
        self.records = self.load_records()
    
    def load_records(self):
        """加载已有记录"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_records(self):
        """保存记录"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
    
    def add_record(self, record):
        """添加新记录"""
        record['record_id'] = len(self.records) + 1
        record['created_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.records.append(record)
        self.save_records()
        return record
    
    def get_all_records(self):
        """获取所有记录"""
        return self.records
    
    def get_record_fields(self):
        """获取电子病历字段定义"""
        return {
            'clinic_name': '植物诊所',
            'doctor_name': '植物医生',
            'farmer_name': '农户名称',
            'farmer_contact': '农户联系方式',
            'district': '所属区县',
            'township': '所属乡镇',
            'village': '所属村庄',
            'issue_date': '开具时间',
            'crop': '作物',
            'has_sample': '是否有样品',
            'disease_occurred': '病虫害是否发生',
            'growth_stage': '发育阶段',
            'affected_part': '受害部位',
            'first_found_year': '首次发现年份',
            'affected_area': '发生面积（亩）',
            'occurrence_rate': '发生比重',
            'symptoms': '主要症状',
            'symptom_distribution': '田间症状分布',
            'consultation_record': '问诊记录',
            'diagnosis_result': '诊断结果(病/虫/杂草的名称)',
            'pesticide_category': '农药大类',
            'pesticide_name': '开具农药名称',
            'pesticide_quantity': '开具农药数量',
            'agricultural_control': '农业防治',
            'medicine_status': '拿药状态'
        }


class KnowledgeBase:
    """领域知识库管理"""
    
    def __init__(self, db_file="knowledge_base.json"):
        self.db_file = db_file
        self.data = self.load_knowledge()
    
    def load_knowledge(self):
        """加载知识库"""
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'pesticides': [],
            'diseases': [],
            'crops': [],
            'treatment_methods': []
        }
    
    def save_knowledge(self):
        """保存知识库"""
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def add_pesticide(self, pesticide_info):
        """添加农药信息"""
        self.data['pesticides'].append(pesticide_info)
        self.save_knowledge()
    
    def get_pesticides(self):
        """获取农药列表"""
        return self.data['pesticides']
    
    def search_pesticide(self, keyword):
        """搜索农药"""
        results = []
        for p in self.data['pesticides']:
            if keyword.lower() in p.get('name', '').lower():
                results.append(p)
        return results


class WeatherDataCollector:
    """气象数据采集器"""
    
    def __init__(self):
        self.api_key = None  # 气象API密钥
    
    def set_api_key(self, key):
        """设置API密钥"""
        self.api_key = key
    
    def get_weather_data(self, location, start_date, end_date):
        """获取气象数据"""
        # 这里可以对接气象API
        # 示例数据结构
        return {
            'location': location,
            'start_date': start_date,
            'end_date': end_date,
            'data': [
                {
                    'date': start_date,
                    'temperature': 25.5,
                    'humidity': 65,
                    'rainfall': 10.2,
                    'wind_speed': 3.5
                }
            ]
        }
    
    def get_forecast(self, location, days=7):
        """获取天气预报"""
        return {
            'location': location,
            'forecast_days': days,
            'data': []
        }

