import os

class FolderTree:
    def __init__(self, root_path: str, show_hidden: bool = False, max_depth: int = None):
        self.root_path = os.path.abspath(root_path)
        self.show_hidden = show_hidden
        self.max_depth = max_depth

    def generate(self):
        self._print_tree(self.root_path)

    def _print_tree(self, current_path: str, prefix: str = "", depth: int = 0):
        if self.max_depth is not None and depth > self.max_depth:
            return

        try:
            entries = sorted(os.listdir(current_path))
        except PermissionError:
            print(prefix + "└── [Permission Denied]")
            return

        if not self.show_hidden:
            entries = [e for e in entries if not e.startswith('.')]

        for index, entry in enumerate(entries):
            path = os.path.join(current_path, entry)
            connector = "├── " if index < len(entries) - 1 else "└── "
            print(prefix + connector + entry)

            if os.path.isdir(path):
                extension = "│   " if index < len(entries) - 1 else "    "
                self._print_tree(path, prefix + extension, depth + 1)
