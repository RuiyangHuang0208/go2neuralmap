# Navigation Module 需求规格说明书
## 项目概述
- 项目名称：网格路径规划模块（Navigation Module）
- 核心目标：在规则网格地图上实现可配置算法的路径规划，提供完整的规划、测试和可视化功能。
## 核心功能需求
1. 地图系统
    基于@`utils.neural_map`中定义的cost_map 进行索引
2. 移动规则
    - 移动方向：支持八方向移动（上、下、左、右、左上、右上、左下、右下）
    - 移动代价：从单元格A移动到相邻单元格B的代价 = 单元格B的代价值
    - 斜向移动：对角线移动的代价不缩放，直接使用目标单元格的代价
3. 路径规划
    - 输入：
        - 起点坐标：`start: Tuple[int, int]`
        - 终点坐标：`goal: Tuple[int, int]`
        - 代价地图：`cost_map: torch.Tensor`
    - 输出：标准化规划结果 PlanResult，包含：
        - 状态：枚举类型（成功、各类失败原因）
        - 路径：成功时为包含起点和终点的坐标列表 List[Tuple[int, int]]；失败时为 None
        - 总代价：成功时为路径上所有经过单元格代价之和（浮点数）；失败时为 0.0
    - 算法支持：
        初步实现 A-star算法作为核心演示
        使用切比雪夫距离作为启发函数：`h(n) = max(abs(dx), abs(dy))`
        设计为可扩展架构，未来可兼容其他算法
4. 接口设计
    采用策略模式实现算法兼容性：
        抽象基类：BasePlanner，定义 plan(start, goal, cost_map) -> PlanResult接口
        具体实现：AStarPlanner实现该接口
        规划结果：PlanResult数据类，封装状态、路径、总代价等字段
## 非功能性需求
正确性：在均匀代价地图上，算法应能找到几何最短路径
可扩展性：算法接口设计支持未来添加 Dijkstra、JPS 等其他规划算法
易用性：提供清晰的API文档和示例代码
Note: All comments should be written in English
## 依赖说明
核心依赖：torch（代价地图容器）
可视化依赖：matplotlib（绘图），numpy（数据转换）
测试框架：pytest（推荐）
