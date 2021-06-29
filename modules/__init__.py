import sys
from pathlib import Path

effdet = Path(__file__).resolve().with_name('detr')
sys.path.append(str(effdet))
