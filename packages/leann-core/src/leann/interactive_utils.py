"""
Interactive session utilities for LEANN applications.

Provides shared readline functionality and command handling across
CLI, API, and RAG example interactive modes.
"""

import atexit
import os
import readline
from pathlib import Path
from typing import Callable, Optional


class InteractiveSession:
    """Manages interactive session with readline support and common commands."""

    def __init__(
        self,
        history_name: str,
        prompt: str = "You: ",
        welcome_message: str = "",
        enable_commands: bool = True,
        custom_commands: Optional[dict[str, Callable[[str], bool]]] = None,
    ):
        """
        Initialize interactive session with readline support.

        Args:
            history_name: Name for history file (e.g., "cli", "api_chat")
            prompt: Input prompt to display
            welcome_message: Message to show when starting session
            enable_commands: Enable built-in commands (help, clear, history)
            custom_commands: Additional commands {command: handler_func}
                           Handler should return True if it handled the command
        """
        self.history_name = history_name
        self.prompt = prompt
        self.welcome_message = welcome_message
        self.enable_commands = enable_commands
        self.custom_commands = custom_commands or {}
        self._setup_complete = False

    def setup_readline(self):
        """Setup readline with history support."""
        if self._setup_complete:
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

    def handle_builtin_command(self, user_input: str) -> bool:
        """
        Handle built-in commands.

        Args:
            user_input: The user's input string

        Returns:
            True if command was handled, False if not a built-in command
        """
        if not self.enable_commands:
            return False

        command = user_input.lower().strip()

        if command in ["quit", "exit", "q"]:
            print("Goodbye!")
            return "quit"

        elif command == "help":
            self._show_help()
            return True

        elif command == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            return True

        elif command == "history":
            self._show_history()
            return True

        # Check custom commands
        for cmd_name, handler in self.custom_commands.items():
            if command == cmd_name.lower():
                return handler(user_input)

        return False

    def _show_help(self):
        """Show available commands."""
        print("Commands:")
        print("  quit/exit/q - Exit the chat")
        print("  help - Show this help message")
        print("  clear - Clear screen")
        print("  history - Show command history")

        # Show custom commands
        if self.custom_commands:
            for cmd_name in self.custom_commands.keys():
                print(f"  {cmd_name} - Custom command")

    def _show_history(self):
        """Show command history."""
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
        Get user input with readline support and command handling.

        Returns:
            User input string, None if quit command, or continues loop for built-in commands
        """
        try:
            user_input = input(self.prompt).strip()

            # Handle built-in commands
            result = self.handle_builtin_command(user_input)
            if result == "quit":
                return None
            elif result is True:  # Command was handled, continue loop
                return self.get_user_input()

            # Return the input for the caller to process
            return user_input

        except KeyboardInterrupt:
            print("\n(Use 'quit' to exit)")
            return self.get_user_input()
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

            if user_input is None:  # Quit command or EOF
                break

            if not user_input:  # Empty input
                continue

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
        enable_commands=True,
    )


def create_api_session() -> InteractiveSession:
    """Create an interactive session for API chat."""
    return InteractiveSession(
        history_name="api_chat",
        prompt="You: ",
        welcome_message="Leann Chat started (type 'quit' to exit, 'help' for commands)\n"
        + "=" * 40,
        enable_commands=True,
    )


def create_rag_session(app_name: str, data_description: str) -> InteractiveSession:
    """Create an interactive session for RAG examples."""
    return InteractiveSession(
        history_name=f"{app_name}_rag",
        prompt="You: ",
        welcome_message=f"[Interactive Mode] Chat with your {data_description} data!\nType 'quit' or 'exit' to stop, 'help' for commands.\n"
        + "=" * 40,
        enable_commands=True,
    )
