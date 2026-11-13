from pynput import keyboard
import numpy as np
import time


class LinuxKeyboardController:
    def __init__(self):
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.max_linear = 1.0
        self.max_angular = 2.0
        
        # 当前按下的键
        self.pressed_keys = set()
        # Keys that transitioned from released to pressed since the last poll.
        self.just_pressed_keys = set()
        
        # 启动监听器
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
    
    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                key_name = key.char.lower()
                if key_name not in self.pressed_keys:
                    self.just_pressed_keys.add(key_name)
                self.pressed_keys.add(key_name)
        except AttributeError:
            # 特殊键处理
            if key == keyboard.Key.up:
                self.pressed_keys.add('up')
            elif key == keyboard.Key.down:
                self.pressed_keys.add('down')
            elif key == keyboard.Key.left:
                self.pressed_keys.add('left')
            elif key == keyboard.Key.right:
                self.pressed_keys.add('right')
            elif key == keyboard.Key.space:
                self.pressed_keys.add('space')
    
    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            if key == keyboard.Key.up:
                self.pressed_keys.discard('up')
            elif key == keyboard.Key.down:
                self.pressed_keys.discard('down')
            elif key == keyboard.Key.left:
                self.pressed_keys.discard('left')
            elif key == keyboard.Key.right:
                self.pressed_keys.discard('right')
            elif key == keyboard.Key.space:
                self.pressed_keys.discard('space')
    
    def get_command(self):
        """获取控制命令"""
        vx, vy, omega = 0.0, 0.0, 0.0
        
        # 移动控制
        if 'w' in self.pressed_keys or 'up' in self.pressed_keys:
            vx = self.max_linear
        if 's' in self.pressed_keys or 'down' in self.pressed_keys:
            vx = -self.max_linear
        if 'a' in self.pressed_keys or 'left' in self.pressed_keys:
            vy = self.max_linear
        if 'd' in self.pressed_keys or 'right' in self.pressed_keys:
            vy = -self.max_linear
        
        # 旋转控制
        if 'q' in self.pressed_keys:
            omega = self.max_angular
        if 'e' in self.pressed_keys:
            omega = -self.max_angular
        
        # 急停
        if 'space' in self.pressed_keys:
            vx, vy, omega = 0.0, 0.0, 0.0
        
        return np.array([vx, vy, omega])

    def consume_key_press(self, key_name):
        """Return whether ``key_name`` was newly pressed since the last poll."""
        key_name = key_name.lower()
        if key_name not in self.just_pressed_keys:
            return False
        self.just_pressed_keys.remove(key_name)
        return True
    
    def close(self):
        self.listener.stop()

if __name__ == "__main__":
    kbd_controller = LinuxKeyboardController()
    try:
        while True:
            cmd = kbd_controller.get_command()
            print(cmd)
    except KeyboardInterrupt:
        kbd_controller.close()
