#!/usr/bin/env python3
"""
语音识别
    1. 使用 vosk 离线识别中文语音
    2. 可自定义目标窗口（或不启用）
    3. 可自定义指令关键词与对应键盘操作
    4. 防抖、防误触（仅最终结果触发 + 冷却时间 + 可选的音量门限）
    5. 支持全词匹配 / 包含匹配两种模式
    6. 支持通过 config.json 配置文件调整参数（首次运行自动生成）
"""

import json
import math
import os
import re
import struct   # 提前导入，用于 RMS 计算
import time
import logging
from collections import deque

import pyaudio
import pygetwindow as gw
from pynput.keyboard import Controller, Key
from vosk import Model, KaldiRecognizer

# 隐藏 vosk 内部调试警告
logging.getLogger("vosk").setLevel(logging.ERROR)
logging.getLogger("VoskAPI").setLevel(logging.ERROR)

# ======================== 配置文件管理 ========================
CONFIG_FILE = "config.json"

# 所有可配置项的默认值
DEFAULT_CONFIG = {
    "model_path": "vosk-model-small-cn-0.22",
    "sample_rate": 16000,
    "chunk_size": 8000,
    "channels": 1,
    "format": "paInt16",
    "target_window_keyword": "",
    "activate_on_start": False,
    "command_map": {
        "和解":     "esc",
        "没耐力":     "esc",
        "打不动":     "esc",
        "换行":     "enter",
        "退格":     "backspace",
        "删除":     "delete",
        "空格":     "space",
        "上":       "up",
        "下":       "down",
        "左":       "left",
        "右":       "right",
        "复制":     "ctrl_c",
        "粘贴":     "ctrl_v",
        "剪切":     "ctrl_x",
        "全选":     "ctrl_a",
        "保存":     "ctrl_s",
        "关闭窗口": "alt_f4"
    },
    "match_mode": "contains",
    "cooldown_time": 0.3,
    "enable_partial_results": False,
    "min_rms": 200
}

def load_config(config_path=CONFIG_FILE):
    """加载配置文件，若不存在则用默认值创建"""
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            # 浅合并：仅更新存在的键，命令映射会完全替换
            for key in user_config:
                if key == "command_map" and isinstance(user_config[key], dict):
                    config["command_map"] = user_config[key]  # 完全替换命令表
                elif key in config:
                    config[key] = user_config[key]
        except (json.JSONDecodeError, Exception) as e:
            print(f"[配置] 读取失败 ({e})，将重新生成默认配置文件")
            # 如果读取失败，则用默认值覆盖写入
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
            return DEFAULT_CONFIG.copy()
    else:
        # 初次运行，生成默认配置文件
        print(f"[配置] 未找到 {config_path}，已自动生成默认配置文件，可按需修改")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        return DEFAULT_CONFIG.copy()

    return config

# 加载配置并设置全局变量
config = load_config()
MODEL_PATH = config["model_path"]
SAMPLE_RATE = config["sample_rate"]
CHUNK_SIZE = config["chunk_size"]
CHANNELS = config["channels"]
FORMAT = pyaudio.paInt16 if config["format"] == "paInt16" else pyaudio.paInt16
TARGET_WINDOW_KEYWORD = config["target_window_keyword"]
ACTIVATE_ON_START = config["activate_on_start"]
COMMAND_MAP = config["command_map"]
MATCH_MODE = config["match_mode"]
COOLDOWN_TIME = config["cooldown_time"]
ENABLE_PARTIAL_RESULTS = config["enable_partial_results"]
MIN_RMS = config["min_rms"]

# ======================== 核心实现 ========================
class VoiceKeyboard:
    def __init__(self):
        self.keyboard = Controller()
        self.last_triggered = {}          # 按键动作 : 上次触发时间
        self.partial_history = deque(maxlen=3)  # 用于部分结果去重（若启用）
        self.model = None
        self.recognizer = None
        self.audio = None
        self.stream = None

    def load_model(self, path):
        """加载 vosk 模型"""
        print(f"[初始化] 加载语音模型：{path}")
        self.model = Model(path)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)

    def start_mic(self):
        """打开麦克风流"""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

    def rms(self, data: bytes) -> float:
        """计算音频数据的RMS（均方根，近似音量）"""
        if len(data) == 0:
            return 0
        count = len(data) // 2
        fmt = "<{}h".format(count)
        shorts = struct.unpack(fmt, data)
        sum_squares = sum(s ** 2 for s in shorts)
        return math.sqrt(sum_squares / count)

    def activate_target(self):
        """激活目标窗口（如果配置了关键词）"""
        if not TARGET_WINDOW_KEYWORD:
            return True
        try:
            windows = gw.getWindowsWithTitle(TARGET_WINDOW_KEYWORD)
            if windows:
                win = windows[0]
                # 安全激活：先还原再激活，忽略已知的假异常
                try:
                    if win.isMinimized:
                        win.restore()
                    if not win.isActive:
                        win.activate()
                except Exception as e:
                    if "0 - 操作成功完成" not in str(e) and "Error code from Windows: 0" not in str(e):
                        raise
                time.sleep(0.05)
                return True
            else:
                print(f"[警告] 未找到包含 '{TARGET_WINDOW_KEYWORD}' 的窗口，按键将全局发送")
                return False
        except Exception as e:
            print(f"[警告] 激活窗口异常: {e}")
            return False

    def match_command(self, text: str):
        """根据匹配模式，返回匹配到的第一个动作"""
        for cmd, action in COMMAND_MAP.items():
            if MATCH_MODE == "word":
                # 使用词边界匹配（支持中文）
                if re.search(r'\b' + re.escape(cmd) + r'\b', text):
                    return action
            else:  # contains
                if cmd in text:
                    return action
        return None

    def should_trigger(self, action: str):
        """检查冷却时间是否已过"""
        now = time.time()
        last = self.last_triggered.get(action, 0)
        if now - last >= COOLDOWN_TIME:
            self.last_triggered[action] = now
            return True
        return False

    def execute_action(self, action: str):
        """执行键盘动作"""
        try:
            # 组合键处理，如 ctrl_c, alt_f4
            if '_' in action:
                parts = action.split('_')
                if len(parts) == 2:
                    mod_str, key_str = parts
                    modifier = getattr(Key, mod_str, None) if hasattr(Key, mod_str) else mod_str
                    with self.keyboard.pressed(modifier):
                        self.keyboard.press(key_str)
                        self.keyboard.release(key_str)
                else:
                    print(f"[错误] 无法解析组合键：{action}")
                return

            # 单键（Key 枚举或字符）
            if hasattr(Key, action):
                key = getattr(Key, action)
            else:
                key = action
            self.keyboard.press(key)
            self.keyboard.release(key)
        except Exception as e:
            print(f"[错误] 按键模拟失败：{action} - {e}")

    def on_result(self, text: str, is_final: bool):
        """处理识别文本"""
        if not text or (not is_final and not ENABLE_PARTIAL_RESULTS):
            return

        # 去重：若启用部分结果，相同内容不重复触发
        if not is_final and ENABLE_PARTIAL_RESULTS:
            if text in self.partial_history:
                return
            self.partial_history.append(text)

        action = self.match_command(text)
        if action:
            if TARGET_WINDOW_KEYWORD:
                self.activate_target()
            if self.should_trigger(action):
                print(f"[识别] '{text}' → 触发按键 '{action}'")
                self.execute_action(action)

    def run(self):
        """主循环"""
        print("=" * 50)
        print("语音指令键盘助手已启动")
        if TARGET_WINDOW_KEYWORD:
            print(f"目标窗口：{TARGET_WINDOW_KEYWORD}")
        else:
            print("已设为全局按键模式")
        print(f"匹配模式：{MATCH_MODE}，冷却时间：{COOLDOWN_TIME}秒")
        if MIN_RMS > 0:
            print(f"音量门限：{MIN_RMS}")
        print("指令列表：")
        for cmd, act in COMMAND_MAP.items():
            print(f"    {cmd}  →  {act}")
        print("说“退出程序”可结束运行，或按 Ctrl+C 中断")
        print("=" * 50)

        if TARGET_WINDOW_KEYWORD and ACTIVATE_ON_START:
            self.activate_target()

        try:
            while True:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                if len(data) == 0:
                    continue

                # 音量门限
                if MIN_RMS > 200:
                    rms_val = self.rms(data)
                    if rms_val < MIN_RMS:
                        continue

                # 送入识别器
                if self.recognizer.AcceptWaveform(data):
                    final_json = json.loads(self.recognizer.Result())
                    final_text = final_json.get("text", "").strip()
                    self.on_result(final_text, is_final=True)
                else:
                    partial_json = json.loads(self.recognizer.PartialResult())
                    partial_text = partial_json.get("partial", "").strip()
                    if partial_text:
                        self.on_result(partial_text, is_final=False)

        except KeyboardInterrupt:
            print("\n[退出] 用户中断")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            print("[清理] 资源已释放")

if __name__ == "__main__":
    vk = VoiceKeyboard()
    try:
        vk.load_model(MODEL_PATH)
        vk.start_mic()
        vk.run()
    except FileNotFoundError as e:
        print(f"[致命错误] 模型文件未找到：{e}")
        print("请确认 MODEL_PATH 指向正确的 vosk 模型目录")
    except Exception as e:
        print(f"[致命错误] {e}")