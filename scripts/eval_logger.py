"""
评测日志记录工具（统一版）

所有评测脚本使用 agents.hymem_logger 中的统一日志系统。
本模块提供向后兼容的导入入口。
"""

from agents.hymem_logger import (
    HyMemEventLog,
    HyMemLogFormatter,
    HyMemEvalLogger,
    TokenUsage,
    LogLimits,
)

# 向后兼容：原 HyMemLogger 映射到 HyMemEvalLogger
HyMemLogger = HyMemEvalLogger
