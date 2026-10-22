"""Render a stored transcript (canonical JSON) into download formats on demand.
Storing one canonical form and deriving the rest keeps storage small and formats consistent."""


def _ts(sec: float, sep: str) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{sep}{ms:03}"


def to_txt(t: dict) -> str:
    return t.get("text", "").strip() + "\n"


def to_srt(t: dict) -> str:
    blocks = [
        f"{i}\n{_ts(seg['start'], ',')} --> {_ts(seg['end'], ',')}\n{seg['text']}\n"
        for i, seg in enumerate(t.get("segments", []), 1)
    ]
    return "\n".join(blocks)


def to_vtt(t: dict) -> str:
    cues = [f"{_ts(seg['start'], '.')} --> {_ts(seg['end'], '.')}\n{seg['text']}\n" for seg in t.get("segments", [])]
    return "WEBVTT\n\n" + "\n".join(cues)
