import re
from datetime import datetime

def format_timestamp(ts=None):
    return (ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")

def clean_html_tags(raw_html: str) -> str:
    return re.sub(r'<[^>]+>', '', raw_html)