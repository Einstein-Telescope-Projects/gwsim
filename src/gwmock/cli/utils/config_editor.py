# ruff: noqa PLC0415

"""Textual-based interactive configuration editor for gwmock.

Launches a TUI with:
- Left panel: live view of current configuration
- Right panel: command output and help
- Bottom input: slash-command REPL
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.content import Content
from textual.widgets import Header, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from gwmock.cli.utils.config_state import (
    SECTION_DESC,
    SECTION_EXTRA,
    SECTION_KEYS,
    ConfigState,
)
from gwmock.cli.utils.discovery import (
    discover_geometries,
    discover_glitch_models,
    discover_population_presets,
    discover_psds,
    discover_source_types,
    discover_waveform_models,
)

# Descriptions for discovered options
PSD_DESCRIPTIONS = {
    "ET_10_full_cryo_psd": "Einstein Telescope 10km arms, full cryogenic cooling",
    "ET_10_HF_psd": "Einstein Telescope 10km arms, high-frequency optimized",
    "ET_15_full_cryo_psd": "Einstein Telescope 15km arms, full cryogenic cooling",
    "ET_15_HF_psd": "Einstein Telescope 15km arms, high-frequency optimized",
    "ET_20_full_cryo_psd": "Einstein Telescope 20km arms, full cryogenic cooling",
    "ET_20_HF_psd": "Einstein Telescope 20km arms, high-frequency optimized",
    "ET_D_psd": "Einstein Telescope Design study baseline",
}

GEOMETRY_DESCRIPTIONS = {
    "ET-Triangle-EMR": "Einstein Telescope triangle layout, equatorial mountain region",
    "ET-Triangle-Sardinia": "Einstein Telescope triangle layout, Sardinia region",
    "ET-2L-Aligned": "Einstein Telescope 2-arm L-shape, aligned configuration",
    "ET-2L-Misaligned": "Einstein Telescope 2-arm L-shape, misaligned configuration",
    "ET-EMR": "Einstein Telescope single detector, equatorial mountain region",
    "ET-L": "Einstein Telescope single L-shaped detector",
    "ET-Sardinia": "Einstein Telescope single detector, Sardinia region",
    "ET-triangle": "Einstein Telescope triangle layout (alias)",
    "H1L1": "LIGO Hanford + LIGO Livingston (current detectors)",
    "H1L1V1": "LIGO Hanford + LIGO Livingston + Virgo (current detectors)",
    "HLVK": "LIGO Hanford + LIGO Livingston + Virgo + KAGRA (current detectors)",
}

SECTION_EXAMPLES = {
    "noise": [
        "/noise psd ET_10_full_cryo_psd",
        "/noise seed 42",
        "/noise detectors ET-Triangle-EMR",
    ],
    "signal": [
        "/signal source-type bbh",
        "/signal waveform-model IMRPhenomXPHM",
        "/signal detectors ET-Triangle-EMR",
        "/signal minimum-frequency 20",
    ],
    "population": [
        "/population backend file",
        "/population path /path/to/population.h5",
        "/population n-samples 100",
    ],
    "globals": [
        "/globals sampling-frequency 4096",
        "/globals duration 1024",
        "/globals start-time 1000000000",
    ],
    "batch": [
        "/batch scheduler slurm",
        "/batch job-name my_simulation",
        "/batch resources nodes 4",
        "/batch submit account myproject",
    ],
}

CSS = """
#content {
    height: 1fr;
}

#config-panel {
    width: 38%;
    border: tall $accent;
    padding: 1;
    overflow-y: auto;
}

#output-panel {
    width: 62%;
    border: tall $primary;
    padding: 0 1;
}

#command-input {
    dock: bottom;
    height: 3;
}

#suggestion-panel {
    dock: bottom;
    height: auto;
    max-height: 10;
    margin-bottom: 3;
    border: tall $primary;
    background: $surface;
    display: none;
}

#suggestion-panel.visible {
    display: block;
}
"""


def _render_tree(data: dict | Any, indent: int = 0) -> str:
    """Render a nested dict as a coloured tree string."""
    if not isinstance(data, dict):
        return str(data)
    lines: list[str] = []
    pad = "  " * indent
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict) and not value:
            continue
        if isinstance(value, dict):
            lines.append(f"{pad}[bold cyan]{key}[/bold cyan]:")
            lines.append(_render_tree(value, indent + 1))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                lines.append(f"{pad}[bold cyan]{key}[/bold cyan]:")
                for i, item in enumerate(value):
                    lines.append(f"{pad}  [dim]\\[{i}][/dim]")
                    lines.append(_render_tree(item, indent + 2))
            else:
                items = ", ".join(str(v) for v in value)
                lines.append(f"{pad}[bold]{key}[/bold]: [cyan]{items}[/cyan]")
        elif isinstance(value, bool):
            colour = "green" if value else "red"
            lines.append(f"{pad}[bold]{key}[/bold]: [{colour}]{value}[/{colour}]")
        else:
            lines.append(f"{pad}[bold]{key}[/bold]: {value}")
    return "\n".join(lines)


class ConfigEditorApp(App):  # type: ignore[misc]
    """Interactive gwmock configuration editor."""

    TITLE = "gwmock config editor"
    CSS = CSS
    BINDINGS = [
        Binding("tab", "accept_suggestion", "Accept suggestion", show=False),
        Binding("escape", "dismiss_suggestions", "Dismiss suggestions", show=False),
    ]

    def __init__(self, load_path: Path | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._state = ConfigState()
        self._load_path = load_path
        self._history: list[str] = []
        self._history_index: int = -1  # -1 means not navigating history
        self._current_input: str = ""  # Store current input when navigating history
        self._navigating_history: bool = False  # Flag to prevent suggestions during history navigation
        self._handlers: dict[str, Any] = {
            "help": self._cmd_help,
            "config": self._cmd_config,
            "geometries": self._cmd_geometries,
            "psds": self._cmd_psds,
            "source-types": self._cmd_source_types,
            "waveforms": self._cmd_waveforms,
            "glitches": self._cmd_glitches,
            "presets": self._cmd_presets,
            "noise": self._cmd_noise,
            "signal": self._cmd_signal,
            "population": self._cmd_population,
            "globals": self._cmd_globals,
            "batch": self._cmd_batch,
            "reset": self._cmd_reset,
            "load": self._cmd_load,
            "save": self._cmd_save,
            "generate-script": self._cmd_generate_script,
            "template": self._cmd_template,
            "quit": self._cmd_quit,
        }

    # -- Textual lifecycle ------------------------------------------------- #

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="content"):
            yield Static(id="config-panel")
            yield RichLog(id="output-panel", wrap=True, highlight=True, markup=True, auto_scroll=True)
        yield OptionList(id="suggestion-panel")
        yield Input(id="command-input", placeholder="Type /help for available commands")

    def on_mount(self) -> None:
        if self._load_path:
            self._cmd_load([str(self._load_path)])
        out = self.query_one("#output-panel", RichLog)
        out.write("[bold cyan]Welcome to the gwmock config editor[/bold cyan]")
        out.write("")
        out.write("gwmock simulates gravitational wave detector data for testing analysis pipelines.")
        out.write("This editor helps you build a configuration file step by step.")
        out.write("")
        out.write("[bold]Typical workflow:[/bold]")
        out.write("  1. Choose your detector geometry and noise model")
        out.write("  2. Optionally add signal injections (gravitational wave sources)")
        out.write("  3. Set global simulation parameters (duration, sampling rate)")
        out.write("  4. Save your configuration")
        out.write("")
        out.write("Type [bold]/help[/bold] to see all available commands.")
        out.write("Type [bold]/psds[/bold] or [bold]/geometries[/bold] to explore options.")
        out.write("")
        self._refresh_panel()
        self.query_one("#command-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._hide_suggestions()
        raw = event.value.strip()
        event.input.value = ""
        if not raw:
            return

        # Add to history (avoid duplicates at the end)
        if not self._history or self._history[-1] != raw:
            self._history.append(raw)

        # Reset history navigation
        self._history_index = -1
        self._current_input = ""

        self._dispatch(raw)
        self._refresh_panel()

    def on_input_changed(self, event: Input.Changed) -> None:
        # Skip suggestions when navigating history
        if self._navigating_history:
            return

        text = event.value
        suggestions = self._get_suggestions(text)
        panel = self.query_one("#suggestion-panel", OptionList)
        panel.clear_options()
        if suggestions:
            for s in suggestions:
                panel.add_option(Option(s))
            panel.add_class("visible")
            panel.highlighted = 0
        else:
            panel.remove_class("visible")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._accept_suggestion_at(event.option_index)

    def _hide_suggestions(self) -> None:
        panel = self.query_one("#suggestion-panel", OptionList)
        panel.clear_options()
        panel.remove_class("visible")

    def on_key(self, event) -> None:
        panel = self.query_one("#suggestion-panel", OptionList)
        is_visible = "visible" in panel.classes

        if event.key == "escape" and is_visible:
            self._hide_suggestions()
            self.query_one("#command-input", Input).focus()
            event.prevent_default()
            return

        if event.key == "tab" and is_visible and panel.highlighted is not None:
            self._accept_suggestion_at(panel.highlighted)
            event.prevent_default()
            return

        if event.key == "down" and is_visible:
            if panel.highlighted is not None and panel.highlighted < len(panel.options) - 1:
                panel.highlighted = panel.highlighted + 1
            event.prevent_default()
            return

        if event.key == "up" and is_visible:
            if panel.highlighted is not None and panel.highlighted > 0:
                panel.highlighted = panel.highlighted - 1
            event.prevent_default()
            return

        # History navigation when suggestion panel is not visible
        inp = self.query_one("#command-input", Input)

        if event.key == "up" and not is_visible and self._history:
            # Navigate backward in history (older commands)
            self._navigating_history = True
            if self._history_index == -1:
                # Start navigating: save current input
                self._current_input = inp.value
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                # Move to older command
                self._history_index -= 1

            inp.value = self._history[self._history_index]
            inp.cursor_position = len(inp.value)
            event.prevent_default()
            return

        if event.key == "down" and not is_visible and self._history_index >= 0:
            # Navigate forward in history (newer commands)
            self._navigating_history = True
            if self._history_index < len(self._history) - 1:
                # Move to newer command
                self._history_index += 1
                inp.value = self._history[self._history_index]
                inp.cursor_position = len(inp.value)
            else:
                # At the end: restore current input and exit history mode
                self._history_index = -1
                self._navigating_history = False
                inp.value = self._current_input
                inp.cursor_position = len(inp.value)
            event.prevent_default()
            return

        # Reset history navigation flag when user types a regular character
        if event.key not in ("up", "down", "left", "right", "enter", "escape", "tab"):
            self._navigating_history = False
            self._history_index = -1

    def _accept_suggestion_at(self, index: int) -> None:
        panel = self.query_one("#suggestion-panel", OptionList)
        option = panel.get_option_at_index(index)
        inp = self.query_one("#command-input", Input)
        text = inp.value
        suggestion = str(option.prompt)
        parts = text.split()

        if not parts:
            # Empty input, treat as command completion
            inp.value = "/" + suggestion + " "
        elif len(parts) == 1 and not text.endswith(" "):
            # Single word without trailing space: completing the command itself
            # e.g., "/n" → "/noise "
            inp.value = "/" + suggestion + " "
        else:
            # Has trailing space or multiple words: completing a key or value
            # Replace the last partial word if present, otherwise append
            if text.endswith(" "):
                inp.value = text + suggestion + " "
            else:
                # Replace the last partial word
                parts[-1] = suggestion
                inp.value = " ".join(parts) + " "

        inp.cursor_position = len(inp.value)
        self._hide_suggestions()
        inp.focus()

    def action_accept_suggestion(self) -> None:
        panel = self.query_one("#suggestion-panel", OptionList)
        if "visible" in panel.classes and panel.highlighted is not None:
            self._accept_suggestion_at(panel.highlighted)

    def action_dismiss_suggestions(self) -> None:
        self._hide_suggestions()

    def _get_suggestions(self, text: str) -> list[str]:
        has_trailing_space = text.endswith(" ")
        text = text.strip()
        if not text:
            return []

        all_commands = sorted(self._handlers.keys())

        if not text.startswith("/"):
            return [c for c in all_commands if c.startswith(text.lower())]

        without_slash = text[1:]
        parts = without_slash.split()

        if not parts or (len(parts) == 1 and not has_trailing_space):
            prefix = parts[0].lower() if parts else ""
            return [c for c in all_commands if c.startswith(prefix)]

        cmd = parts[0]

        # Handle /template suggestions
        if cmd == "template":
            if len(parts) == 1:
                return ["noise", "signal", "glitch"]
            prefix = parts[1].lower()
            return [t for t in ["noise", "signal", "glitch"] if t.startswith(prefix)]

        # Handle /generate-script suggestions
        if cmd == "generate-script":
            if len(parts) == 1:
                return ["slurm", "local"]
            if len(parts) == 2 and not has_trailing_space:
                prefix = parts[1].lower()
                return [t for t in ["slurm", "local"] if t.startswith(prefix)]
            return []

        if cmd not in SECTION_KEYS:
            return []

        section_keys = list(SECTION_KEYS[cmd].keys())
        extra_keys = [e[0] for e in SECTION_EXTRA.get(cmd, [])]

        if len(parts) == 1:
            return section_keys + extra_keys

        key = parts[1]

        if has_trailing_space or len(parts) >= 3:
            active_key = key if not has_trailing_space else parts[-1]
            if cmd == "noise" and active_key == "psd":
                return discover_psds()
            if active_key == "detectors":
                return discover_geometries()
            if cmd == "noise" and active_key == "glitch" and len(parts) >= 3 and parts[2] == "add":
                return discover_glitch_models()
            if cmd == "signal" and active_key == "source-type":
                return discover_source_types()
            if cmd == "signal" and active_key == "waveform-model":
                return discover_waveform_models()
            if cmd == "population" and active_key == "backend":
                return ["file", "cbc_prior", "bbh", "bns_prior", "nsbh_prior"]
            if cmd == "population" and active_key == "source-type":
                return discover_source_types()
            if cmd == "globals" and active_key == "total-duration":
                return ["1 day", "6 hours", "1 hour", "3600"]
            return []

        prefix = parts[1].lower()
        return [k for k in section_keys + extra_keys if k.startswith(prefix)]

    # -- dispatch ---------------------------------------------------------- #

    def _dispatch(self, raw: str) -> None:
        out = self.query_one("#output-panel", RichLog)
        if not raw.startswith("/"):
            out.write("[red]Commands start with /. Type /help for available commands.[/red]")
            return
        try:
            parts = shlex.split(raw[1:])
        except ValueError as exc:
            out.write(f"[red]Invalid syntax: {exc}[/red]")
            return
        if not parts:
            return
        cmd, args = parts[0], parts[1:]
        handler = self._handlers.get(cmd)
        if handler is None:
            out.write(f"[red]Unknown command: /{cmd}[/red]  —  type /help for available commands.")
            return
        try:
            handler(args)
        except Exception as exc:  # noqa: BLE001
            out.write(f"[red]Error: {exc}[/red]")

    # -- panel refresh ----------------------------------------------------- #

    def _refresh_panel(self) -> None:
        panel = self.query_one("#config-panel", Static)
        data = self._state.to_dict()
        orchestration = data.get("orchestration", {})

        # Build progress indicator
        sections = []
        if orchestration.get("noise"):
            sections.append("[green]✓[/green] Noise")
        if orchestration.get("signal"):
            sections.append("[green]✓[/green] Signal")
        if orchestration.get("population"):
            sections.append("[green]✓[/green] Population")
        if data.get("globals"):
            sections.append("[green]✓[/green] Globals")
        if data.get("batch"):
            sections.append("[green]✓[/green] Batch")

        header = "[bold underline]Current Configuration[/bold underline]\n"
        if sections:
            header += "\n[dim]Progress:[/dim] " + " | ".join(sections) + "\n\n"
        else:
            header += "\n[dim]Progress:[/dim] [dim]No sections configured yet[/dim]\n\n"

        if not data:
            markup = header + "[dim]Type [/dim][bold]/help[/bold][dim] to get started.[/dim]"
        else:
            markup = header + _render_tree(data)
        panel.update(Content.from_markup(markup))

    # -- output helpers ---------------------------------------------------- #

    def _out(self) -> RichLog:
        return self.query_one("#output-panel", RichLog)

    def _show_list(self, title: str, items: list[str], descriptions: dict[str, str] | None = None) -> None:
        out = self._out()
        out.write(f"\n[bold cyan]{title}[/bold cyan]")
        if not items:
            out.write("[dim]  (none discovered — is the sub-package installed?)[/dim]")
        else:
            for item in items:
                desc = descriptions.get(item, "") if descriptions else ""
                if desc:
                    out.write(f"  • [bold]{item}[/bold] — {desc}")
                else:
                    out.write(f"  • {item}")
        out.write("")

    def _show_section(self, section: str) -> None:
        out = self._out()
        keys = SECTION_KEYS.get(section, {})
        descs = SECTION_DESC.get(section, {})
        extra = SECTION_EXTRA.get(section, [])
        examples = SECTION_EXAMPLES.get(section, [])
        current = self._state.get_section(section)

        out.write(f"\n[bold cyan]{section.title()} Configuration[/bold cyan]")
        out.write("─" * 40)
        out.write("\n[bold]Available settings:[/bold]")
        for key, desc in descs.items():
            out.write(f"  [bold]{key}[/bold]".ljust(32) + desc)
        for sub, desc in extra:
            out.write(f"  [bold]{sub}[/bold]".ljust(32) + desc)

        if examples:
            out.write("\n[bold]Examples:[/bold]")
            for example in examples:
                out.write(f"  [dim]{example}[/dim]")

        if current:
            out.write(f"\n[bold]Current settings:[/bold]")
            out.write(_render_tree(current, indent=1))
        else:
            out.write(f"\n[dim]No {section} settings configured.[/dim]")
        out.write("")

    # -- discovery commands ------------------------------------------------ #

    def _cmd_geometries(self, _args: list[str]) -> None:
        self._show_list("Available Network Geometries", discover_geometries(), GEOMETRY_DESCRIPTIONS)

    def _cmd_psds(self, _args: list[str]) -> None:
        self._show_list("Available PSD Files", discover_psds(), PSD_DESCRIPTIONS)

    def _cmd_source_types(self, _args: list[str]) -> None:
        self._show_list("Available Source Types", discover_source_types())

    def _cmd_waveforms(self, _args: list[str]) -> None:
        self._show_list("Available Waveform Models", discover_waveform_models())

    def _cmd_glitches(self, _args: list[str]) -> None:
        self._show_list("Available Glitch Models", discover_glitch_models())

    def _cmd_presets(self, _args: list[str]) -> None:
        self._show_list("Available Population Presets", discover_population_presets())

    # -- section commands -------------------------------------------------- #

    def _section_set(self, section: str, args: list[str]) -> None:
        out = self._out()
        if not args:
            self._show_section(section)
            return
        if len(args) < 2:
            out.write(f"[red]Usage: /{section} <key> <value>[/red]")
            out.write(f"Type [bold]/{section}[/bold] without arguments to see available keys.")
            return
        key = args[0]
        value = " ".join(args[1:])
        self._state.set(section, key, value)
        out.write(f"[green]Set {section}.{key} = {value}[/green]")

    def _cmd_noise(self, args: list[str]) -> None:
        out = self._out()
        if args and args[0] == "glitch":
            if len(args) < 2:
                out.write("[red]Usage: /noise glitch add <kind>  |  /noise glitch remove <index>[/red]")
                return
            if args[1] == "add":
                if len(args) < 3:
                    out.write("[red]Usage: /noise glitch add <kind>[/red]")
                    out.write("Use /glitches to see available glitch models.")
                    return
                idx = self._state.add_glitch(args[2])
                out.write(f"[green]Added glitch [{idx}]: {args[2]}[/green]")
            elif args[1] == "remove":
                if len(args) < 3:
                    out.write("[red]Usage: /noise glitch remove <index>[/red]")
                    return
                removed = self._state.remove_glitch(int(args[2]))
                out.write(f"[green]Removed glitch: {removed}[/green]")
            else:
                out.write("[red]Usage: /noise glitch add <kind>  |  /noise glitch remove <index>[/red]")
            return
        self._section_set("noise", args)

    def _cmd_signal(self, args: list[str]) -> None:
        self._section_set("signal", args)

    def _cmd_population(self, args: list[str]) -> None:
        self._section_set("population", args)

    def _cmd_globals(self, args: list[str]) -> None:
        self._section_set("globals", args)

    def _cmd_batch(self, args: list[str]) -> None:
        out = self._out()
        if not args:
            self._show_section("batch")
            return
        if args[0] in ("resources", "submit") and len(args) >= 3:
            key, value = args[1], args[2]
            if args[0] == "resources":
                self._state.set_batch_resource(key, value)
            else:
                self._state.set_batch_submit(key, value)
            out.write(f"[green]Set batch.{args[0]}.{key} = {value}[/green]")
            return
        if len(args) >= 2 and args[0] not in ("resources", "submit"):
            self._section_set("batch", args)
            return
        out.write("[red]Usage: /batch <key> <value>[/red]")
        out.write("       /batch resources <key> <value>")
        out.write("       /batch submit <key> <value>")

    # -- meta commands ----------------------------------------------------- #

    def _cmd_help(self, _args: list[str]) -> None:
        out = self._out()
        out.write("\n[bold cyan]gwmock config editor — Help[/bold cyan]")
        out.write("━" * 40)
        out.write("\n[bold]Getting started:[/bold]")
        out.write("  gwmock simulates gravitational wave detector data.")
        out.write("  Build your config step by step using the commands below.")
        out.write("")
        out.write("[bold]Suggested workflow:[/bold]")
        out.write("  1. Start with a template: [bold]/template <type>[/bold]")
        out.write("  2. Customize settings: [bold]/noise[/bold], [bold]/signal[/bold], etc.")
        out.write("  3. Configure execution: [bold]/batch[/bold]")
        out.write("  4. [bold]/save <filename>[/bold]")
        out.write("  5. Generate scripts: [bold]/generate-script <type> <file>[/bold]")
        out.write("")
        out.write("[bold]Templates:[/bold]")
        out.write("  [bold]/template noise[/bold]".ljust(36) + "Pure noise simulation")
        out.write("  [bold]/template signal[/bold]".ljust(36) + "Signal + noise + population")
        out.write("  [bold]/template glitch[/bold]".ljust(36) + "Noise with glitches")
        out.write("")
        out.write("[bold]Discover available options:[/bold]")
        out.write("  [bold]/geometries[/bold]".ljust(36) + "Detector network configurations")
        out.write("  [bold]/psds[/bold]".ljust(36) + "Noise power spectral densities")
        out.write("  [bold]/source-types[/bold]".ljust(36) + "Gravitational wave source types")
        out.write("  [bold]/waveforms[/bold]".ljust(36) + "Waveform models for signals")
        out.write("  [bold]/glitches[/bold]".ljust(36) + "Non-Gaussian noise models")
        out.write("  [bold]/presets[/bold]".ljust(36) + "Population file presets")
        out.write("")
        out.write("[bold]Configure sections:[/bold]")
        out.write("  Type without arguments to see available keys and examples")
        out.write("  [bold]/noise <key> <value>[/bold]".ljust(36) + "Detector noise settings")
        out.write("  [bold]/signal <key> <value>[/bold]".ljust(36) + "Signal injection settings")
        out.write("  [bold]/population <key> <value>[/bold]".ljust(36) + "Population file settings")
        out.write("  [bold]/globals <key> <value>[/bold]".ljust(36) + "Simulation parameters")
        out.write("  [bold]/batch <key> <value>[/bold]".ljust(36) + "Job scheduler settings")
        out.write("")
        out.write("[bold]Manage your config:[/bold]")
        out.write("  [bold]/config[/bold]".ljust(36) + "Show full current configuration")
        out.write("  [bold]/load <file>[/bold]".ljust(36) + "Load an existing config")
        out.write("  [bold]/save <file>[/bold]".ljust(36) + "Validate and save config")
        out.write("  [bold]/generate-script <type> <file>[/bold]".ljust(36) + "Generate SLURM or local scripts")
        out.write("  [bold]/reset [section|all][/bold]".ljust(36) + "Clear settings")
        out.write("  [bold]/help[/bold]".ljust(36) + "Show this help")
        out.write("  [bold]/quit[/bold]".ljust(36) + "Exit the editor")
        out.write("")

    def _cmd_config(self, _args: list[str]) -> None:
        out = self._out()
        data = self._state.to_dict()
        out.write("\n[bold cyan]Current Configuration[/bold cyan]")
        out.write("━" * 40)
        if not data:
            out.write("[dim]No settings configured.[/dim]")
        else:
            out.write(_render_tree(data))
        out.write("")

    def _cmd_load(self, args: list[str]) -> None:
        out = self._out()
        if not args:
            out.write("[red]Usage: /load <filename>[/red]")
            return
        path = Path(args[0])
        if not path.exists():
            out.write(f"[red]File not found: {path}[/red]")
            return
        self._state.load(path)
        # Track the loaded config file path
        self._state._config_file = str(path)
        out.write(f"[green]Loaded configuration from {path}[/green]")

    def _cmd_save(self, args: list[str]) -> None:
        out = self._out()
        if not args:
            out.write("[red]Usage: /save <filename>[/red]")
            return
        path = Path(args[0])

        valid, error = self._state.validate()
        if not valid:
            out.write("[red]Validation failed:[/red]")
            out.write(error)
            return

        import yaml

        try:
            config_dict = self._state.to_dict()
            if path.exists():
                backup_path = path.with_suffix(f"{path.suffix}.backup")
                backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

            # Track the saved config file path
            self._state._config_file = str(path)

            out.write(f"[green]Configuration saved to {path}[/green]")
            out.write("")
            out.write("[bold]Next steps:[/bold]")
            out.write("")
            out.write("  [cyan]Run locally:[/cyan]")
            out.write(f"    gwmock simulate {path}")
            out.write("")
            out.write("  [cyan]Generate execution script:[/cyan]")
            out.write("    /generate-script slurm submit.sh")
            out.write("    /generate-script local run.sh")
            out.write("")
            out.write("  [cyan]View documentation:[/cyan]")
            out.write("    https://leuven-gravity-institute.github.io/gwmock/")
            out.write("")
        except Exception as exc:  # noqa: BLE001
            out.write(f"[red]Failed to save: {exc}[/red]")

    def _cmd_reset(self, args: list[str]) -> None:
        out = self._out()
        if not args:
            out.write("[yellow]Usage: /reset <section> or /reset all[/yellow]")
            out.write("Sections: noise, signal, population, globals, batch")
            return
        section = args[0]
        if section == "all":
            self._state.reset()
            out.write("[green]All settings reset.[/green]")
        elif section in SECTION_KEYS:
            self._state.reset(section)
            out.write(f"[green]{section} settings reset.[/green]")
        else:
            out.write(f"[red]Unknown section: {section}[/red]")
            out.write(f"Valid sections: {', '.join(SECTION_KEYS)}")

    def _cmd_generate_script(self, args: list[str]) -> None:
        out = self._out()
        if not args:
            out.write("[yellow]Usage: /generate-script <type> <output-file>[/yellow]")
            out.write("Types: slurm, local")
            out.write("Example: /generate-script slurm submit.sh")
            return

        script_type = args[0]
        output_file = args[1] if len(args) > 1 else "submit.sh"
        config_data = self._state.to_dict()

        if "batch" not in config_data:
            out.write("[red]No batch configuration found. Use /batch to configure.[/red]")
            return

        batch_config = config_data["batch"]
        job_name = batch_config.get("job-name", "gwmock_job")
        scheduler = batch_config.get("scheduler", "slurm")
        chunks = batch_config.get("chunks", {})
        chunks_enabled = chunks.get("enabled", False)
        n_chunks = chunks.get("n-chunks", 1)
        chunks_parallel = chunks.get("parallel", True)

        # Get the config file path (assume it's been saved)
        config_file = "config.yaml"  # Default
        if self._state._config_file:
            config_file = self._state._config_file

        if script_type == "slurm":
            script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err
"""
            # Add resources
            resources = batch_config.get("resources", {})
            for key, value in resources.items():
                script += f"#SBATCH --{key}={value}\n"

            # Add submit options
            submit = batch_config.get("submit", {})
            if submit:
                for key, value in submit.items():
                    script += f"#SBATCH --{key}={value}\n"

            script += "\n"

            # Add extra lines
            extra_lines = batch_config.get("extra-lines", [])
            if extra_lines:
                for line in extra_lines:
                    script += f"{line}\n"
                script += "\n"

            # Add simulation command
            if chunks_enabled and n_chunks > 1:
                if chunks_parallel:
                    script += f"# Array job for {n_chunks} chunks\n"
                    script += f"#SBATCH --array=0-{n_chunks - 1}\n\n"
                    script += f"gwmock simulate {config_file} --chunk ${{SLURM_ARRAY_TASK_ID}}\n"
                else:
                    script += f"# Sequential execution of {n_chunks} chunks\n"
                    for i in range(n_chunks):
                        script += f"gwmock simulate {config_file} --chunk {i}\n"
            else:
                script += f"gwmock simulate {config_file}\n"

        elif script_type == "local":
            script = f"""#!/bin/bash
# Local execution script
"""
            # Add extra lines
            extra_lines = batch_config.get("extra-lines", [])
            if extra_lines:
                for line in extra_lines:
                    script += f"{line}\n"
                script += "\n"

            # Add simulation command
            if chunks_enabled and n_chunks > 1:
                if chunks_parallel:
                    script += f"# Parallel execution of {n_chunks} chunks\n"
                    for i in range(n_chunks):
                        script += f"gwmock simulate {config_file} --chunk {i} &\n"
                    script += "wait\n"
                else:
                    script += f"# Sequential execution of {n_chunks} chunks\n"
                    for i in range(n_chunks):
                        script += f"gwmock simulate {config_file} --chunk {i}\n"
            else:
                script += f"gwmock simulate {config_file}\n"
        else:
            out.write(f"[red]Unknown script type: {script_type}[/red]")
            out.write("Valid types: slurm, local")
            return

        try:
            with open(output_file, "w") as f:
                f.write(script)
            out.write(f"[green]Script generated: {output_file}[/green]")
            if script_type == "slurm":
                out.write(f"Submit with: [bold]sbatch {output_file}[/bold]")
            else:
                out.write(f"Run with: [bold]bash {output_file}[/bold]")
        except Exception as exc:  # noqa: BLE001
            out.write(f"[red]Failed to generate script: {exc}[/red]")

    def _cmd_template(self, args: list[str]) -> None:
        out = self._out()
        if not args:
            out.write("[yellow]Usage: /template <type>[/yellow]")
            out.write("Available templates:")
            out.write("  [bold]noise[/bold]       - Pure noise simulation")
            out.write("  [bold]signal[/bold]      - Signal + noise + population")
            out.write("  [bold]glitch[/bold]      - Noise with glitches")
            return

        template_type = args[0]
        self._state.reset()

        if template_type == "noise":
            self._state.set("noise", "psd", "ET_10_full_cryo_psd")
            self._state.set("noise", "seed", "42")
            self._state.set("noise", "detectors", "ET-Triangle-EMR")
            out.write("[green]Template loaded: noise[/green]")
            out.write("Configure PSD with /noise psd <value>")
            out.write("Configure detectors with /noise detectors <value>")

        elif template_type == "signal":
            self._state.set("noise", "psd", "ET_10_full_cryo_psd")
            self._state.set("noise", "seed", "42")
            self._state.set("noise", "detectors", "ET-Triangle-EMR")
            self._state.set("signal", "source-type", "bbh")
            self._state.set("signal", "detectors", "ET-Triangle-EMR")
            self._state.set("population", "backend", "file")
            out.write("[green]Template loaded: signal[/green]")
            out.write("Configure population path with /population path <file>")

        elif template_type == "glitch":
            self._state.set("noise", "psd", "ET_10_full_cryo_psd")
            self._state.set("noise", "seed", "42")
            self._state.set("noise", "detectors", "ET-Triangle-EMR")
            # Add a glitch using the proper method
            self._state.add_glitch("gengli_blip")
            out.write("[green]Template loaded: glitch[/green]")
            out.write("Configure glitch parameters with /noise glitches")

        else:
            out.write(f"[red]Unknown template: {template_type}[/red]")
            out.write("Valid templates: noise, signal, glitch")

        self._refresh_panel()

    def _cmd_quit(self, _args: list[str]) -> None:
        self.exit()
