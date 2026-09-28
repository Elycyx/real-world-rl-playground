"""Accessible keyboard teleoperation using gym-hil's intervention protocol."""

from threading import Lock
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gym_hil.mujoco_gym_env import MAX_GRIPPER_COMMAND
from gym_hil.wrappers.hil_wrappers import InputsControlWrapper
from gym_hil.wrappers.intervention_utils import KeyboardController


class ManipulationKeyboard(KeyboardController):
    _motion_keys: ClassVar[dict[str, str]] = {
        "w": "forward_x",
        "s": "backward_x",
        "a": "forward_y",
        "d": "backward_y",
        "q": "forward_z",
        "e": "backward_z",
    }

    def __init__(self):
        super().__init__(x_step_size=1.0, y_step_size=1.0, z_step_size=1.0)
        self._pressed: set[str] = set()
        self._toggle_lock = Lock()
        self._gripper_toggles = 0

    @staticmethod
    def _name(key) -> str | None:
        character = getattr(key, "char", None)
        if character:
            if character == " ":
                return "space"
            return character.lower()
        return getattr(key, "name", None)

    def press(self, key) -> None:
        name = self._name(key)
        if name is None or name in self._pressed:
            return
        self._pressed.add(name)
        if name in self._motion_keys:
            self.key_states[self._motion_keys[name]] = True
        elif name == "g" and self.should_intervene():
            with self._toggle_lock:
                self._gripper_toggles += 1
        elif name == "space":
            self.key_states["intervention"] = not self.key_states["intervention"]
            print(f"Human takeover: {'ON' if self.key_states['intervention'] else 'OFF'}", flush=True)
        elif name == "enter":
            self.episode_end_status = "success"
        elif name in ("backspace", "esc"):
            self.episode_end_status = "failure"

    def release(self, key) -> None:
        name = self._name(key)
        if name is None:
            return
        self._pressed.discard(name)
        if name in self._motion_keys:
            self.key_states[self._motion_keys[name]] = False

    def consume_gripper_toggle(self) -> bool:
        with self._toggle_lock:
            toggle = self._gripper_toggles % 2 != 0
            self._gripper_toggles = 0
        return toggle

    def start(self) -> None:
        from pynput import keyboard

        self.listener = keyboard.Listener(on_press=self.press, on_release=self.release)
        self.listener.start()
        print("Keyboard: Space=takeover on/off | WASD=XY | Q/E=up/down | "
              "G=toggle gripper | Enter=success | Backspace/Esc=failure", flush=True)

    def reset(self) -> None:
        super().reset()
        self.open_gripper_command = False
        self.close_gripper_command = False
        self.episode_end_status = None
        self.consume_gripper_toggle()


class KeyboardIntervention(InputsControlWrapper):
    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.controller = ManipulationKeyboard()
        self.auto_reset = False
        self.use_gripper = True
        self.controller.start()

    def get_gamepad_action(self):
        is_active, action, terminate, success, rerecord = super().get_gamepad_action()
        toggle = self.controller.consume_gripper_toggle()
        if is_active and toggle:
            closed = float(np.asarray(self.env.unwrapped.get_gripper_pose()).item()) >= MAX_GRIPPER_COMMAND / 2
            action[3] = 0 if closed else 2
            print(f"Gripper: {'OPEN' if closed else 'CLOSED'}", flush=True)
        return is_active, action, terminate, success, rerecord
