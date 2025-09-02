import os
from pathlib import Path

class SerialCounter:
    TXT_FILE = 'result/serial_counter.txt'

    def __init__(self):
        self.file_path = Path(self.TXT_FILE)
        self.serial = self._load_serial()

    def _load_serial(self):
        """尝试读取TXT文件"""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    # 添加文件锁防止并发读取问题
                    content = f.read().strip()
                    return int(content) if content.isdigit() else 0
            except (ValueError, IOError, OSError):
                # 文件内容不是有效的数字或读取失败，重置
                return 0
        else:
            return 0

    def get_serial(self):
        return self.serial

    def new_serial(self):
        self.serial += 1
        try:
            with open(self.file_path, 'w') as f:
                # 添加排他锁防止并发写入
                f.write(str(self.serial))
                f.flush()  # 确保立即写入磁盘
                os.fsync(f.fileno())  # 强制同步到磁盘
        except (IOError, OSError) as e:
            # 如果写入失败，回滚serial值
            self.serial -= 1
            raise IOError(f"Failed to write serial to file: {e}")
        return self.serial
