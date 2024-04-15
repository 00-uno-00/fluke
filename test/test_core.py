from __future__ import annotations
from rich.live import Live
from rich.progress import Progress
import torch
import sys
sys.path.append(".")
sys.path.append("..")

from fl_bench import GlobalSettings, ObserverSubject  # NOQA


def test_settings():
    settings = GlobalSettings()
    settings.set_seed(42)
    assert settings.get_seed() == 42
    settings.set_device("cpu")
    assert settings.get_device() == torch.device("cpu")
    settings.set_device("cuda")
    assert settings.get_device() == torch.device("cuda")
    settings.set_device("mps")
    assert settings.get_device() == torch.device("mps")
    settings.set_device("auto")
    if torch.cuda.is_available():
        assert settings.get_device() == torch.device("cuda")
    else:
        assert settings.get_device() == torch.device("cpu")
    live = settings.get_live_renderer()
    assert live is not None and isinstance(live, Live)
    progress_fl = settings.get_progress_bar("FL")
    progress_s = settings.get_progress_bar("clients")
    progress_c = settings.get_progress_bar("server")
    assert progress_fl is not None and isinstance(progress_fl, Progress)
    assert progress_s is not None and isinstance(progress_s, Progress)
    assert progress_c is not None and isinstance(progress_c, Progress)
    assert progress_fl != progress_s
    assert progress_fl != progress_c
    assert progress_s != progress_c


def test_observer():
    subj = ObserverSubject()
    assert subj._observers == []
    subj.attach("test")
    assert subj._observers == ["test"]
    subj.detach("test")
    assert subj._observers == []


if __name__ == "__main__":
    test_settings()
    test_observer()
