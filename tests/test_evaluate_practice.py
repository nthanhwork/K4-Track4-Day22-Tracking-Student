"""Kiểm tra bản vá NumPy có tác dụng trong tiến trình con chấm điểm."""

from evaluate_practice import run_trackeval


def test_trackeval_child_receives_numpy_aliases_and_arguments(tmp_path):
    """Chạy script giả để xác nhận alias và tham số tới đúng tiến trình con."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_mot_challenge.py").write_text(
        "import sys\n"
        "import numpy as np\n"
        "assert np.float is float\n"
        "assert np.int is int\n"
        "assert sys.argv[0].endswith('run_mot_challenge.py')\n"
        "assert sys.argv[sys.argv.index('--BENCHMARK') + 1] == 'LAB22'\n"
        "assert sys.argv[sys.argv.index('--SEQ_INFO') + 1] == 'video_1'\n"
        "assert sys.argv[sys.argv.index('--TRACKERS_TO_EVAL') + 1] == 'thu_nghiem'\n",
        encoding="utf-8",
    )
    run_trackeval(tmp_path, "thu_nghiem", "LAB22", "train")
