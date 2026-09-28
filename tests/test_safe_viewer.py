from types import SimpleNamespace

import gymnasium as gym
import pytest

from robot_rl import safe_viewer


def test_keyboard_viewer_ignores_keys_and_keeps_rendering(monkeypatch):
    calls = []
    callbacks = {}
    window = object()
    closed = [False]
    dragging = [False]

    class Context:
        def free(self):
            calls.append("free_context")

    glfw = SimpleNamespace(
        TRUE=1, VISIBLE=0, PRESS=1, KEY_LEFT_SHIFT=1, KEY_RIGHT_SHIFT=2,
        MOUSE_BUTTON_LEFT=3, MOUSE_BUTTON_RIGHT=4,
        init=lambda: True,
        default_window_hints=lambda: None,
        window_hint=lambda *args: None,
        create_window=lambda *args: window,
        make_context_current=lambda current: calls.append(("context", current)),
        set_key_callback=lambda current, callback: callbacks.update(key=callback),
        set_cursor_pos_callback=lambda current, callback: callbacks.update(cursor=callback),
        set_scroll_callback=lambda current, callback: callbacks.update(scroll=callback),
        poll_events=lambda: calls.append("poll"),
        window_should_close=lambda current: closed[0],
        get_framebuffer_size=lambda current: (640, 480),
        get_window_size=lambda current: (640, 480),
        get_mouse_button=lambda current, button: glfw.PRESS if dragging[0] and button == glfw.MOUSE_BUTTON_LEFT else 0,
        get_key=lambda current, key: 0,
        swap_buffers=lambda current: calls.append("swap"),
        destroy_window=lambda current: calls.append("destroy"),
    )
    mujoco = SimpleNamespace(
        MjvScene=lambda model, maxgeom: object(),
        MjrContext=lambda model, font: Context(),
        MjvCamera=lambda: object(),
        MjvOption=lambda: object(),
        mjtFontScale=SimpleNamespace(mjFONTSCALE_150=150),
        mjtCatBit=SimpleNamespace(mjCAT_ALL=7),
        mjtFramebuffer=SimpleNamespace(mjFB_WINDOW=1),
        mjtMouse=SimpleNamespace(mjMOUSE_ROTATE_V=1, mjMOUSE_ROTATE_H=2, mjMOUSE_MOVE_H=3,
                                  mjMOUSE_MOVE_V=4, mjMOUSE_ZOOM=5),
        MjrRect=lambda *args: args,
        mjv_defaultFreeCamera=lambda *args: None,
        mjv_updateScene=lambda *args: calls.append("scene"),
        mjr_setBuffer=lambda *args: None,
        mjr_render=lambda *args: calls.append("render"),
        mjv_moveCamera=lambda *args: calls.append("move_camera"),
    )
    monkeypatch.setattr(safe_viewer, "glfw", glfw)
    monkeypatch.setattr(safe_viewer, "mujoco", mujoco)

    class Task(gym.Env):
        model = object()
        data = object()

        def reset(self, **kwargs):
            return {}, {}

        def step(self, action):
            return {}, 0.0, False, False, {}

        def close(self):
            calls.append("close_task")

    viewer = safe_viewer.KeyboardSafeViewer(Task())
    assert calls.count("render") == 1
    callbacks["key"](window, ord("g"), 0, glfw.PRESS, 0)
    assert calls.count("render") == 1
    callbacks["cursor"](window, 0, 0)
    dragging[0] = True
    callbacks["cursor"](window, 10, 5)
    callbacks["scroll"](window, 0, 1)
    assert calls.count("move_camera") == 2
    viewer.reset()
    viewer.step(None)
    assert calls.count("render") == 3
    closed[0] = True
    with pytest.raises(RuntimeError, match="viewer was closed"):
        viewer.step(None)
    viewer.close()
    assert calls[-5:] == [("context", window), "free_context", ("context", None), "destroy", "close_task"]
