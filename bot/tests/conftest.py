# conftest.py — make bot/ modules importable from tests/ (same flat-import
# style the bot itself uses: `import config`, `import features`, …)

import os
import sys

BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BOT_DIR not in sys.path:
    sys.path.insert(0, BOT_DIR)
