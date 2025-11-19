- # AgriGuard Platform - 病虫害时空预测预警系统

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
    <img src="https://img.shields.io/badge/Deep%20Learning-12%20Models-orange.svg" alt="Models">
    <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
  </p>

  <p align="center">
    <strong>基于大数据与人工智能的作物病虫害预测预警系统</strong>
    <br>
    <em>AI-Powered Crop Pest and Disease Forecasting & Early Warning System</em>
    <br>
    <em>TSPeakNet: Dual-scale time-series modeling for district-level crop-disease forecasting and peak-event warning</em>
    <br>
    <em>Yuanze Qin et al</em>
  </p>
  ## 📸 系统展示
  
  ### 🏠 主页 - 功能导航
  ![image-20251118105536358](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118105536358.png)
  
  ### 📊 数据分析 - 多维可视化
  ![image-20251118105613532](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118105613532.png)
  
  
  
  ![image-20251118105632452](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118105632452.png)
  
  ![image-20251118105714655](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118105714655.png)
  
  ![image-20251118105746599](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118105746599.png)
  
  ![image-20251118105805674](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118105805674.png)
  
  ![image-20251118105822093](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118105822093.png)
  
  *年度、月度、区域多维度统计分析，交互式图表*
  
  ### 🔮 模型预测 - 12种AI模型对比
  ![image-20251118132308853](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118132308853.png)
  *集成TSPeakNet、LSTM、GRU等12种深度学习模型*
  
  ### 🗺️ 区域预警 - 实时风险地图
  ![image-20251118132550848](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118132550848.png)
  *北京市区县级预警地图，颜色分级展示风险*
  
  ### 🌐 英文版界面 - 国际化支持
  ![image-20251118132538433](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118132538433.png)
  **
  
  ### 🦠 病害详情 - 专业知识库
  ![image-20251118132633146](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118132633146.png)
  
  *点击查看详细病害特征、发生规律、防治措施*
  
  ### 📝 数据采集 - 病历录入
  ![image-20251118132732173](C:\Users\CAU\Desktop\Github上传\README.assets\image-20251118132732173.png)
  *病历数据和气象数据录入界面，支持多字段表单*
  
  ---
  
  ## 🌟 核心特性
  
  - 🌾 **多维度数据分析** - 整合多年时序数据，支持年度、月度、区域多维度统计分析
  - 🔮 **智能预测模型** - 集成12种深度学习模型，包括TSPeakNet、LSTM、GRU、Transformer等
  - 🗺️ **区域预警可视化** - 实时区域风险地图，精准到区县级别的预警信息
  - 📊 **交互式图表** - 基于ECharts和Plotly的动态数据可视化
  - 🌐 **多语言支持** - 中英双语界面，支持国际化应用
  - 🎯 **事件级评估** - 峰值检测、时间匹配、提前预警时间计算
  
  ## 🏗️ 技术架构
  
  ```
  ┌─────────────────────────────────────────────────────┐
  │                   前端展示层                          │
  │   ECharts 5.x | Plotly.js | HTML5 | CSS3 | ES6+   │
  └─────────────────────┬───────────────────────────────┘
                        │
  ┌─────────────────────┴───────────────────────────────┐
  │                   业务逻辑层                          │
  │   Python 3.8+ | HTTP Server | Data Processing      │
  └─────────────────────┬───────────────────────────────┘
                        │
  ┌─────────────────────┴───────────────────────────────┐
  │                   AI模型层                           │
  │   TSPeakNet | LSTM | GRU | Transformer | 12 Models │
  └─────────────────────┬───────────────────────────────┘
                        │
  ┌─────────────────────┴───────────────────────────────┐
  │                   数据层                             │
  │   Excel (openpyxl) | Time Series | GeoJSON         │
  └─────────────────────────────────────────────────────┘
  ```
  
  **技术栈**:
  - **后端**: Python 3.8+, openpyxl
  - **前端**: HTML5, CSS3, JavaScript (ES6+)
  - **可视化**: ECharts 5.x, Plotly.js
  - **AI模型**: 时空预测模型 + 深度学习
  
  ## 📊 数据来源
  
  - **数据源**: 北京市10区县植物诊所电子病历系统 (PEMR)
  - **时间范围**: 2018-2021年连续时序数据
  - **空间范围**: 大兴区、密云区、平谷区、延庆区、怀柔区、房山区、昌平区、海淀区、通州区、顺义区
  - **数据量**: 覆盖4年 × 10区县 × 365天的病虫害监测记录
  
  ---
  
  ## ⚡ 快速开始
  
  ### 安装运行（3步搞定）
  
  ```bash
  # 1. 安装依赖
  pip install openpyxl
  
  # 2. 启动服务器
  python prediction_server.py
  
  # 3. 访问系统
  # 浏览器打开 http://localhost:8003
  ```
  
  ### 主要页面
  
  | 页面 | 地址 | 功能 |
  |------|------|------|
  | 🏠 主页 | `/` | 系统导航和概览 |
  | 📝 数据采集 | `/data-collection` | 病历和气象数据录入 |
  | 📊 数据分析 | `/data-analysis` | 多维统计分析和可视化 |
  | 🔮 模型预测 | `/model-prediction` | 12种模型预测对比 |
  | 🗺️ 区域预警 | `/regional-warning` | 实时风险地图（中文版） |
  | 🌐 English Warning | `/regional-warning-en` | Risk map (English) |
  
  ---
  
  ## 🎯 系统亮点
  
  ### 1. � 区县级模型对比 - 全国首创
  一张图展示12个深度学习模型在单个区县的预测效果，真实数据vs预测数据直观对比
  - **技术创新**: 宽格式数据自动转换，支持任意区县选择
  - **可视化**: Plotly.js交互式折线图，支持缩放、悬停查看
  - **实用价值**: 帮助选择最优模型进行区县级精准预测
  
  ### 2. 🗺️ 实时预警地图 - 动态风险监控
  ECharts驱动的北京市区县风险地图，颜色分级一目了然
  - **5级预警**: 从正常(绿)到紧急(深红)的渐变色编码
  - **实时更新**: 基于最新预测模型动态计算风险等级
  - **交互体验**: 悬停显示详情，点击查看历史趋势
  
  ### 3. 🦠 智能病害知识库 - AI生成内容
  点击病害卡片，弹出详细的专业知识
  - **内容丰富**: 特征、原因、地区、季节、防治措施
  - **AI生成**: 基于大语言模型生成的专业植保知识
  - **用户友好**: 美观的模态框设计，分类清晰
  

---

  ## 项目结构

  ```
  spatiotemporal_prediction_system/
  ├── prediction_server.py          # 主服务器文件
  ├── simple_data_reader.py         # 数据读取模块
  ├── data_analyzer.py              # 数据分析模块
  ├── data_collector.py             # 数据采集模块
  ├── model_prediction_page.html    # 模型预测页面
  ├── requirements.txt              # Python依赖
  ├── README.md                     # 本说明文档
  ├── 时序数据/                     # 数据目录
  │   ├── 原始数据.xlsx
  │   ├── LSTM-预测数据.xlsx
  │   ├── GRU-预测数据.xlsx
  │   ├── TSPeakNet-预测模型.xlsx
  │   ├── ... (其他模型预测数据)
  │   └── 北京.json                 # 地图数据
  └── static/                       # 静态资源目录
  ```

---

  ### 数据准备

  **原始数据格式** (`原始数据.xlsx`):
  ```
  Date        | Node_DaXing | Node_MiYun | Node_PingGu | ...
  2018-09-25  | 3.65        | 15.71      | 16.32       | ...
  2018-09-26  | 4.23        | 14.88      | 17.45       | ...
  ```

  **预测数据格式** (`LSTM-预测数据.xlsx`):
  ```
  Date        | Node_DaXing | Node_MiYun | Node_PingGu | ...
  2021-01-01  | 2.34        | 12.45      | 15.67       | ...
  2021-01-02  | 2.56        | 13.21      | 16.23       | ...
  ```

---

  ## 技术细节

  ### 模型预测流程

  ```
  历史时序数据
      ↓
  特征工程
      ↓
  深度学习模型 (12种)
      ↓
  预测结果生成
      ↓
  性能评估 (MAE/RMSE/R²)
      ↓
  可视化对比
  ```

  ### 预警生成逻辑

  ```
  实时数据采集
      ↓
  时空预测模型
      ↓
  风险等级评估
      ↓
  阈值判断
      ↓
  生成预警信息
      ↓
  地图可视化
  ```

---

  ## 版权与引用

  ### 版权信息

  © 2025 AgriGuard Platform. 基于大数据与人工智能的病虫害预测预警系统

  **数据来源**: 北京市10区县植物诊所 | 2018-2021年处方数据 
  **技术支持**: 时空预测模型 + 深度学习 + 大语言模型 
  **开发单位**: 中国农业大学 信息与电气工程学院 
  **开发团队**: 张领先教授团队 秦源泽等人

  ### 学术引用

  如使用本系统或相关技术，请引用以下论文:

  ```bibtex
  @article{qin2025tspeaknet,
    title={TSPeakNet: Dual-scale time-series modeling for district-level crop-disease forecasting and peak-event warning},
    author={Qin, Yuanze and Han, Zonghuan and Zhang, Lingxian and Zhang, Yiding},
    year={2025}
  }
  ```

  ### 开源协议

  本项目采用 [MIT License / Apache 2.0] 开源协议

---

  ## 联系方式

  - **技术支持**: zhanglx@cau.edu.cn    qinyuanze@cau.edu.cn 
  - **项目主页**: [GitHub链接]
