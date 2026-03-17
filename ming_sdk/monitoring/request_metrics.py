import logging
import time
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

REQUEST_METRICS_LOG_FILE = "/home/admin/logs/request_metrics.log"


def setup_request_metrics_logger():
    metrics_logger = logging.getLogger("request_metrics")

    if not metrics_logger.handlers:
        log_dir = os.path.dirname(REQUEST_METRICS_LOG_FILE)
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except OSError as e:
                print(f"Warning: Cannot create log directory {log_dir}: {e}")
                return metrics_logger

        try:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                REQUEST_METRICS_LOG_FILE,
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=5,
                encoding="utf-8",
            )

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)

            metrics_logger.addHandler(file_handler)
            metrics_logger.setLevel(logging.INFO)
        except Exception as e:
            print(f"Warning: Cannot setup file logging: {e}")

    return metrics_logger


# 获取独立配置的logger
logger = setup_request_metrics_logger()


@dataclass
class ReqState:
    """请求状态统计类"""

    request_id: str
    service_name: str
    created_time: float = field(default_factory=time.time)
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    input_token_length: int = 0
    output_token_length: int = 0
    finish_reason: str = ""
    status: str = "success"
    stream_mode: bool = False
    extra_data: Dict[str, Any] = field(default_factory=dict)
    is_record_input_token_length: bool = False

    def record_first_token(self):
        """记录首token时间"""
        if self.first_token_time == 0.0:
            self.first_token_time = time.time()

    def record_input_tokens(self, count: int = 0):
        """记录输入token数量"""
        if not self.is_record_input_token_length:
            self.input_token_length = count
            self.is_record_input_token_length = True

    def increment_input_tokens(self, count: int = 0):
        """增加token计数"""
        self.input_token_length += count

    def increment_output_tokens(self, count: int = 0):
        """增加token计数"""
        self.output_token_length += count

    def finish(self, status: str = "success", **kwargs):
        """完成请求统计"""
        self.status = status
        self.finished_time = time.time()
        self.extra_data.update(kwargs)
        self._log_metrics()

    def _log_metrics(self):
        """打印日志"""
        e2e_latency = (self.finished_time - self.created_time) * 1000

        # 流式请求记录TTFT，非流式默认为0
        ttft = 0.0

        if self.stream_mode and self.first_token_time:
            ttft = (self.first_token_time - self.created_time) * 1000

        # 根据流式/非流式计算不同的TPOT
        if self.output_token_length > 0:
            if not self.stream_mode:
                # 非流式：tpot = e2e / output_token_length
                tpot = e2e_latency / self.output_token_length
            else:
                # 流式：tpot = (e2e - ttft) / (output_token_length - 1)
                # 注意：当output_token_length=1时，避免除以0
                if self.output_token_length > 1 and ttft > 0:
                    tpot = (e2e_latency - ttft) / (self.output_token_length - 1)
                else:
                    tpot = 0.0
        else:
            tpot = 0.0

        log_parts = [
            f"service=[{self.service_name}]",
            f"request_id=[{self.request_id}]",
            f"timestamp=[{datetime.utcfromtimestamp(self.created_time).isoformat()}]",
            f"status=[{self.status}]",
            f"stream_mode=[{self.stream_mode}]",
            f"e2e_latency_ms=[{round(e2e_latency, 2)}]",
            f"ttft_ms=[{round(ttft, 2)}]",  # 非流式显示0.0
            f"tpot_ms=[{round(tpot, 2)}]",
            f"input_token_length=[{self.input_token_length}]",
            f"output_token_length=[{self.output_token_length}]",
        ]

        # 添加额外数据
        for key, value in self.extra_data.items():
            if value is not None:
                log_parts.append(f"{key}=[{value}]")

        logger.info("[REQUEST_METRICS] " + ",".join(log_parts))


class RequestMetrics:
    def __init__(self, service_name: str):
        self.service_name = service_name

    def create_state(
        self, request_id: Optional[str] = "0", stream_mode: bool = False
    ) -> ReqState:
        """创建新的请求状态对象"""
        return ReqState(
            request_id=request_id,
            service_name=self.service_name,
            stream_mode=stream_mode,
        )


metrics_text = RequestMetrics("text")
metrics_image = RequestMetrics("image")
metrics_speech = RequestMetrics("speech")
metrics_tts = RequestMetrics("tts")
metrics_speech_text_audio = RequestMetrics("speech_audio")