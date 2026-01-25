import os, json, hashlib, datetime
from pathlib import Path
import pandas as pd

def _now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

def _short_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:6]

class RunLogger:
    def __init__(self, root_dir="experiments", index_file="experiments/index.csv"):
        self.root_dir = Path(root_dir)
        self.index_file = Path(index_file)
        self.run_dir = None
        self.run_id = None
        self.run_time = None

    def start(self, config: dict):
        now = datetime.datetime.now()
        day = now.strftime("%Y-%m-%d")
        run_id = f"run_{now.strftime('%H%M%S')}_{_short_hash(config)}"
        run_dir = self.root_dir / day / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        self.run_dir = run_dir
        self.run_id = run_id
        self.run_time = now.strftime("%Y-%m-%d %H:%M:%S")

        self.save_json("config.json", config)
        return run_id, run_dir

    def save_json(self, name: str, obj: dict):
        p = self.run_dir / name
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return str(p)

    def save_df(self, name: str, df: pd.DataFrame):
        p = self.run_dir / name
        df.to_csv(p, index=False, encoding="utf-8-sig")
        return str(p)

    def save_fig(self, name: str, fig):
        p = self.run_dir / name
        fig.savefig(p, bbox_inches="tight", dpi=150)
        return str(p)

    def append_index_row(self, row: dict):
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame([row])
        if self.index_file.exists():
            existing_df = pd.read_csv(self.index_file)
            merged_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
            merged_df.to_csv(self.index_file, index=False, encoding="utf-8-sig")
        else:
            new_df.to_csv(self.index_file, index=False, encoding="utf-8-sig")


def build_dashboard(index_file="experiments/index.csv", output_file="experiments/index.html",
                    template_file="dashboard/experiment_dashboard.html"):
    """生成实验总览HTML（从 index.csv 构建）"""
    index_path = Path(index_file)
    output_path = Path(output_file)
    if not index_path.exists():
        return None

    df = pd.read_csv(index_path)
    if df.empty:
        return None

    # 将 run_dir 处理为相对路径，方便在本地直接打开HTML
    def _rel_path(p):
        try:
            return str(Path(p).relative_to(output_path.parent))
        except Exception:
            return str(p)

    if "run_dir" in df.columns:
        df["run_dir"] = df["run_dir"].apply(_rel_path)
        df["run_link"] = df["run_dir"].apply(lambda p: f'<a href="{p}/index.html">打开</a>' if p else "")
        # 直接链接到常用产出文件，减少页面跳转层级
        df["config_link"] = df["run_dir"].apply(lambda p: f'<a href="{p}/config.json">config</a>' if p else "")
        df["metrics_link"] = df["run_dir"].apply(lambda p: f'<a href="{p}/metrics.json">metrics</a>' if p else "")
        df["curve_csv_link"] = df["run_dir"].apply(lambda p: f'<a href="{p}/equity_curve.csv">curve.csv</a>' if p else "")
        df["curve_png_link"] = df["run_dir"].apply(lambda p: f'<a href="{p}/equity_curve.png">curve.png</a>' if p else "")
    if "equity_curve" in df.columns:
        df["equity_curve"] = df["equity_curve"].apply(_rel_path)
        df["curve_link"] = df["equity_curve"].apply(lambda p: f'<a href="{p}">曲线</a>' if p else "")

    # 准备展示列
    display_cols = []
    for col in ["run_id", "run_time", "strategy", "total_return_pct", "sharpe", "max_drawdown",
                "run_link", "curve_link"]:
        if col in df.columns:
            display_cols.append(col)
    if not display_cols:
        display_cols = list(df.columns)

    table_html = df[display_cols].to_html(index=False, escape=False)
    template_path = Path(template_file)
    if template_path.exists():
        html = template_path.read_text(encoding="utf-8").format(table_html=table_html)
    else:
        html = table_html
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return str(output_path)
