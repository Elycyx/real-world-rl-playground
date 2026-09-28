"""MuJoCo GLFW viewer without Simulate's built-in keyboard shortcuts."""

import glfw
import gymnasium as gym
import mujoco


class KeyboardSafeViewer(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._window = None
        self._context = None
        model = env.unwrapped.model
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize the MuJoCo viewer")
        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        self._window = glfw.create_window(960, 720, "PickCube — keyboard-safe viewer", None, None)
        if self._window is None:
            raise RuntimeError("GLFW could not create the MuJoCo viewer window")
        try:
            glfw.make_context_current(self._window)
            self._scene = mujoco.MjvScene(model, maxgeom=10000)
            self._context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            self._camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(model, self._camera)
            self._options = mujoco.MjvOption()
            self._cursor = None
            glfw.set_key_callback(self._window, lambda *_: None)
            glfw.set_cursor_pos_callback(self._window, self._move_camera)
            glfw.set_scroll_callback(self._window, self._zoom_camera)
            self.sync()
        except Exception:
            self._destroy_window()
            raise

    def _move_camera(self, window, xpos, ypos):
        position = (xpos, ypos)
        previous = self._cursor
        self._cursor = position
        if previous is None:
            return
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if not (left or right):
            return
        height = max(glfw.get_window_size(window)[1], 1)
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        action = (mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V) if right else (
            mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
        mujoco.mjv_moveCamera(self.unwrapped.model, action,
                             (xpos - previous[0]) / height, (ypos - previous[1]) / height,
                             self._scene, self._camera)

    def _zoom_camera(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.unwrapped.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                             0, -0.05 * yoffset, self._scene, self._camera)

    def sync(self) -> None:
        glfw.make_context_current(self._window)
        glfw.poll_events()
        if glfw.window_should_close(self._window):
            raise RuntimeError("MuJoCo viewer was closed during interactive training")
        width, height = glfw.get_framebuffer_size(self._window)
        if width > 0 and height > 0:
            mujoco.mjv_updateScene(self.unwrapped.model, self.unwrapped.data, self._options,
                                  None, self._camera, mujoco.mjtCatBit.mjCAT_ALL, self._scene)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self._context)
            mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), self._scene, self._context)
            glfw.swap_buffers(self._window)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.sync()
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        self.sync()
        return result

    def _destroy_window(self):
        if self._window is not None:
            glfw.make_context_current(self._window)
            if self._context is not None:
                self._context.free()
                self._context = None
            glfw.make_context_current(None)
            glfw.destroy_window(self._window)
            self._window = None

    def close(self):
        try:
            self._destroy_window()
        finally:
            self.env.close()
