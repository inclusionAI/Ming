#!/bin/bash
pip install grouped_gemm==0.1.4
pip install --no-cache-dir transformer_engine[pytorch]==1.11.0
exec "$@"