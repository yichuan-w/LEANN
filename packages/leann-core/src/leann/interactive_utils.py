"""
Interactive session utilities for LEANN applications.

Provides shared readline functionality and command handling across
CLI, API, and RAG example interactive modes.
"""

import atexit
import os
from pathlib import Path
from typing import Callable, Optional

# Try to import readline with fallback for Windows
try:
    import readline

    HAS_READLINE = True
except ImportError:
    # Windows doesn't have readline by default
    HAS_READLINE = False
    readline = None


class InteractiveSession:
    """Manages interactive session with optional readline support and common commands."""

    def __init__(
        self,
        history_name: str,
        prompt: str = "You: ",
        welcome_message: str = "",
    ):
        """
        Initialize interactive session with optional readline support.

        Args:
            history_name: Name for history file (e.g., "cli", "api_chat")
                         (ignored if readline not available)
            prompt: Input prompt to display
            welcome_message: Message to show when starting session

        Note:
            On systems without readline (e.g., Windows), falls back to basic input()
            with limited functionality (no history, no line editing).
        """
        self.history_name = history_name
        self.prompt = prompt
        self.welcome_message = welcome_message
        self._setup_complete = False

    def setup_readline(self):
        """Setup readline with history support (if available)."""
        if self._setup_complete:
            return

        if not HAS_READLINE:
            # Readline not available (likely Windows), skip setup
            self._setup_complete = True
            return

        # History file setup
        history_dir = Path.home() / ".leann" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / f"{self.history_name}.history"

        # Load history if exists
        try:
            readline.read_history_file(str(history_file))
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass

        # Save history on exit
        atexit.register(readline.write_history_file, str(history_file))

        # Optional: Enable vi editing mode (commented out by default)
        # readline.parse_and_bind("set editing-mode vi")

        self._setup_complete = True

    def _show_help(self):
        """Show available commands."""
        print("Commands:")
        print("  quit/exit/q - Exit the chat")
        print("  help - Show this help message")
        print("  clear - Clear screen")
        print("  history - Show command history")

    def _show_history(self):
        """Show command history."""
        if not HAS_READLINE:
            print("  History not available (readline not supported on this system)")
            return

        history_length = readline.get_current_history_length()
        if history_length == 0:
            print("  No history available")
            return

        for i in range(history_length):
            item = readline.get_history_item(i + 1)
            if item:
                print(f"  {i + 1}: {item}")

    def get_user_input(self) -> Optional[str]:
        """
        Get user input with readline support.

        Returns:
            User input string, or None if EOF (Ctrl+D)
        """
        try:
            return input(self.prompt).strip()
        except KeyboardInterrupt:
            print("\n(Use 'quit' to exit)")
            return ""  # Return empty string to continue
        except EOFError:
            print("\nGoodbye!")
            return None

    def run_interactive_loop(self, handler_func: Callable[[str], None]):
        """
        Run the interactive loop with a custom handler function.

        Args:
            handler_func: Function to handle user input that's not a built-in command
                         Should accept a string and handle the user's query
        """
        self.setup_readline()

        if self.welcome_message:
            print(self.welcome_message)

        while True:
            user_input = self.get_user_input()

            if user_input is None:  # EOF (Ctrl+D)
                break

            if not user_input:  # Empty input or KeyboardInterrupt
                continue

            # Handle built-in commands
            command = user_input.lower()
            if command in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif command == "help":
                self._show_help()
            elif command == "clear":
                os.system("clear" if os.name != "nt" else "cls")
            elif command == "history":
                self._show_history()
            else:
                # Regular user input - pass to handler
                try:
                    handler_func(user_input)
                except Exception as e:
                    print(f"Error: {e}")


def create_cli_session(index_name: str) -> InteractiveSession:
    """Create an interactive session for CLI usage."""
    return InteractiveSession(
        history_name=index_name,
        prompt="\nYou: ",
        welcome_message="LEANN Assistant ready! Type 'quit' to exit, 'help' for commands\n"
        + "=" * 40,
    )


def create_api_session() -> InteractiveSession:
    """Create an interactive session for API chat."""
    return InteractiveSession(
        history_name="api_chat",
        prompt="You: ",
        welcome_message="Leann Chat started (type 'quit' to exit, 'help' for commands)\n"
        + "=" * 40,
    )


def create_rag_session(app_name: str, data_description: str) -> InteractiveSession:
    """Create an interactive session for RAG examples."""
    return InteractiveSession(
        history_name=f"{app_name}_rag",
        prompt="You: ",
        welcome_message=f"[Interactive Mode] Chat with your {data_description} data!\nType 'quit' or 'exit' to stop, 'help' for commands.\n"
        + "=" * 40,
    )
