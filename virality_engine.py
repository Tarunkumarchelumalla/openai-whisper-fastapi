import re
from typing import List, Dict


class ViralityEngine:
    """
    Rule-based virality scoring engine (V1)

    Input  : Whisper captions JSON
    Output : Ranked clip candidates with scores & reasons
    """

    # ---------------- CONFIG ---------------- #

    HOOK_KEYWORDS = {
        "how", "why", "what", "this", "these", "most", "people",
        "i found", "the reason", "so how", "you can"
    }

    INFO_KEYWORDS = {
        "api", "model", "system", "whisper", "openai", "queue",
        "worker", "server", "process", "resource", "fastapi", "llm"
    }

    FILLER_WORDS = {
        "okay", "yes", "basically", "let's", "wait", "uh", "um"
    }

    MIN_CLIP_SEC = 10
    MAX_CLIP_SEC = 45

    # ---------------- UTILS ---------------- #

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"\w+", text.lower()))

    @staticmethod
    def _keyword_hits(text: str, keywords: set) -> int:
        text = text.lower()
        return sum(1 for k in keywords if k in text)

    # ---------------- BUILD CLIPS ---------------- #

    def _build_candidates(self, segments: List[Dict]) -> List[List[Dict]]:
        clips = []
        window = []

        for seg in segments:
            window.append(seg)
            duration = window[-1]["end"] - window[0]["start"]

            if duration >= self.MIN_CLIP_SEC:
                clips.append(window.copy())

            if duration >= self.MAX_CLIP_SEC:
                window.pop(0)

        return clips

    # ---------------- SCORING ---------------- #

    def _score_clip(self, clip: List[Dict]) -> Dict:
        start = clip[0]["start"]
        end = clip[-1]["end"]
        duration = end - start

        text = " ".join(s["text"] for s in clip)
        words = self._word_count(text)

        # Speech density (energy proxy)
        speech_density = words / max(duration, 1)
        density_score = min(speech_density / 3.5, 1.0)

        # Hook (first ~5s)
        early_text = " ".join(s["text"] for s in clip[:2])
        hook_hits = self._keyword_hits(early_text, self.HOOK_KEYWORDS)
        hook_score = min(hook_hits / 2, 1.0)

        # Info density
        info_hits = self._keyword_hits(text, self.INFO_KEYWORDS)
        info_score = min(info_hits / 4, 1.0)

        # Filler penalty
        filler_hits = self._keyword_hits(text, self.FILLER_WORDS)
        filler_penalty = min(filler_hits / 6, 1.0)

        final_score = (
            hook_score * 0.35 +
            density_score * 0.30 +
            info_score * 0.25 -
            filler_penalty * 0.20
        )

        return {
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(duration, 2),
            "score": round(max(final_score, 0), 3),
            "signals": {
                "hook": round(hook_score, 2),
                "speech_density": round(speech_density, 2),
                "info": round(info_score, 2),
                "filler": round(filler_penalty, 2)
            },
            "text_preview": text[:180] + "..."
        }

    # ---------------- PUBLIC API ---------------- #

    def generate(self, captions: Dict, top_k: int = 5) -> Dict:
        segments = captions.get("segments", [])

        if not segments:
            return {"clips": [], "total_candidates": 0}

        candidates = self._build_candidates(segments)
        scored = [self._score_clip(c) for c in candidates]

        scored.sort(key=lambda x: x["score"], reverse=True)

        return {
            "total_candidates": len(scored),
            "clips": scored[:top_k]
        }
