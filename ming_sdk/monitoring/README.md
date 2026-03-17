# 请求指标监控使用说明

## 概述
为三个推理服务（text、speech、image）添加了请求粒度的指标监控，通过日志输出。

## 支持指标
每个请求都会记录以下指标：
- **ttft**：首个token响应时间（毫秒）
- **tpot**：每个token平均生成时间（毫秒）  
- **e2e_latency**：端到端总延迟（毫秒）
- **input_token_length**：输入token长度
- **output_token_length**：输出token长度
- **status**：请求状态（success/fail）

## 日志格式
```
2024-01-15 14:30:45,123 - ming_sdk.monitoring.request_metrics - INFO - [REQUEST_METRICS] service=[speech],request_id=[f47ac10b-58cc-4372-a567-0e02b2c3d479],timestamp=[2024-01-15T06:30:45.123456],status=[success],e2e_latency_ms=[1234.56],ttft_ms=[234.78],tpot_ms=[45.67],input_token_length=[25],output_token_length=[22050],speaker=[luna]
```

## 使用方法
当前支持 moe、talker、img。如需新增可参考如下代码：

```python
#!/usr/bin/env python3

def demonstrate_new_usage():
    from ming_sdk.monitoring.request_metrics import metrics_speech

    # 1. 创建状态对象
    state = metrics_speech.create_state()

    # 2. 设置初始信息
    state.input_token_length = len("输入文本")

    try:
        # 3. 请求处理中...
        state.record_first_token()  # 记录首token时间
        state.increment_output_tokens(100)  # 累计token数

        # 4. 成功完成
        state.finish("success", speaker="luna")

    except Exception as e:
        # 5. 失败完成
        state.finish("fail", error=str(e), speaker="luna")


if __name__ == "__main__":
    demonstrate_new_usage()
```

### 查看日志
使用关键字过滤日志：
```bash
grep "\[REQUEST_METRICS\]" application.log
```

### 日志级别
所有监控日志使用INFO级别输出。
