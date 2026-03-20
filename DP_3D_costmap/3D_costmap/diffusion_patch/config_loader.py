"""
설정 파일 로드 유틸리티
"""

import yaml


def load_config(config_path):
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
