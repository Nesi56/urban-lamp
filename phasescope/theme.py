"""Light and dark colour themes for the whole application.

Both themes use Qt's Fusion style (the native Windows style ignores most palette
colours) with a palette and a style sheet that also gives the controls some room.
Image areas stay dark in both themes, as usual for scientific images.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtGui, QtWidgets

THEMES = {"light": "Light", "dark": "Dark"}

_COLORS = {
    "light": {"window": "#f3f4f6", "base": "#ffffff", "alt": "#f6f7f9", "text": "#1d1f23",
              "button": "#fbfbfc", "border": "#c8cdd5", "hover": "#e7eaef", "accent": "#2f6fd0",
              "accent_soft": "#dbe7fb", "muted": "#5f6368", "disabled": "#9aa0a6", "tip": "#ffffe1"},
    "dark": {"window": "#1f2227", "base": "#16181c", "alt": "#23262c", "text": "#e3e5e8",
             "button": "#2b2f36", "border": "#3d424b", "hover": "#353a42", "accent": "#4a8cf0",
             "accent_soft": "#26364f", "muted": "#a1a7af", "disabled": "#6b7078", "tip": "#2b2f36"},
}


def colors(theme: str) -> dict[str, str]:
    return _COLORS.get(theme, _COLORS["light"])


def palette(theme: str) -> QtGui.QPalette:
    c = {k: QtGui.QColor(v) for k, v in colors(theme).items()}
    R, G = QtGui.QPalette.ColorRole, QtGui.QPalette.ColorGroup
    pal = QtGui.QPalette()
    for role, key in ((R.Window, "window"), (R.WindowText, "text"), (R.Base, "base"), (R.AlternateBase, "alt"),
                      (R.Text, "text"), (R.Button, "button"), (R.ButtonText, "text"), (R.ToolTipBase, "tip"),
                      (R.ToolTipText, "text"), (R.Highlight, "accent"), (R.Link, "accent"),
                      (R.PlaceholderText, "muted"), (R.Mid, "border"), (R.Midlight, "hover")):
        pal.setColor(role, c[key])
    pal.setColor(R.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(R.BrightText, QtGui.QColor("#ff5252"))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(G.Disabled, role, c["disabled"])
    return pal


def stylesheet(theme: str) -> str:
    c = colors(theme)
    return f"""
QToolBar {{ spacing: 6px; padding: 6px 10px; border: none; border-bottom: 1px solid {c['border']}; }}
QToolBar::separator {{ width: 1px; margin: 6px 12px; background: {c['border']}; }}
QToolBar QToolButton {{ padding: 5px 10px; border: 1px solid transparent; border-radius: 6px; }}
QToolBar QToolButton:hover {{ background: {c['hover']}; border-color: {c['border']}; }}
QToolBar QToolButton:checked {{ background: {c['accent_soft']}; border-color: {c['accent']}; }}
QToolBar QToolButton#primaryButton {{ background: {c['accent']}; color: #ffffff; font-weight: bold;
    border-color: {c['accent']}; padding: 6px 16px; }}
QToolBar QToolButton#primaryButton::menu-button {{ border: none; width: 20px;
    border-left: 1px solid rgba(255, 255, 255, 110); }}
QToolBar QToolButton#primaryButton:disabled {{ background: {c['hover']}; color: {c['disabled']};
    border-color: {c['border']}; }}
QToolBar QLabel {{ padding: 0 2px; }}
QGroupBox {{ font-weight: bold; border: 1px solid {c['border']}; border-radius: 8px; margin-top: 18px;
    padding: 14px 10px 10px 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 4px; }}
QPushButton {{ background: {c['button']}; border: 1px solid {c['border']}; border-radius: 6px; padding: 5px 12px; }}
QPushButton:hover {{ background: {c['hover']}; border-color: {c['accent']}; }}
QPushButton:pressed, QPushButton:checked {{ background: {c['accent_soft']}; border-color: {c['accent']}; }}
QPushButton:disabled {{ color: {c['disabled']}; border-color: {c['border']}; background: {c['window']}; }}
QPushButton::menu-indicator {{ subcontrol-position: right center; right: 6px; }}
QLabel[muted="true"] {{ color: {c['muted']}; }}
QDockWidget::title {{ padding: 6px 8px; }}
QStatusBar {{ border-top: 1px solid {c['border']}; }}
QToolTip {{ color: {c['text']}; background: {c['tip']}; border: 1px solid {c['border']}; padding: 4px; }}
"""


def apply(app: QtWidgets.QApplication, theme: str) -> None:
    """Switch the whole application to ``theme`` ('light' or 'dark')."""
    theme = theme if theme in THEMES else "light"
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    css = stylesheet(theme)
    if app.property("phasescope_theme") == theme and app.styleSheet() == css:
        return  # already applied: re-styling every widget again is slow
    app.setPalette(palette(theme))
    app.setStyleSheet(css)
    app.setProperty("phasescope_theme", theme)


def current(app: QtWidgets.QApplication | None = None) -> str:
    app = app or QtWidgets.QApplication.instance()
    theme = app.property("phasescope_theme") if app is not None else None
    return theme if theme in THEMES else "light"
