from pathlib import Path
from matplotlib import font_manager as fm
path=Path(__file__).resolve().parents[1]/'assets/fonts/NanumMyeongjo-Regular.ttf'
fm.fontManager.addfont(path)
KOREAN_FONT=fm.FontProperties(fname=path).get_name()
