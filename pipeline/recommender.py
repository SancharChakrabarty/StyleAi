"""
pipeline/recommender.py
LLM Recommendation Engine — Pipeline Stage 3
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "models/llm/merged"
_cached_recommender = None

LLM_DIR = str(Path(__file__).parent.parent / "models" / "llm")


def generate_recommendation(
    enriched_input: Dict[str, Any],
    model_path: str = DEFAULT_MODEL_PATH,
    backend:    str = "local",
    **generation_kwargs,
) -> str:
    rec = _get_cached_recommender(model_path, backend)
    try:
        report = rec.generate(enriched_input, **generation_kwargs)
    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Report generation failed: {exc}") from exc

    if not report.strip():
        raise RuntimeError("LLM returned an empty response.")

    return report


def _get_cached_recommender(model_path: str, backend: str):
    global _cached_recommender

    if _cached_recommender is not None:
        return _cached_recommender

    # Insert models/llm at position 0 so it takes priority over models/vision
    if LLM_DIR not in sys.path:
        sys.path.insert(0, LLM_DIR)
    else:
        # Move it to front to ensure it wins over models/vision
        sys.path.remove(LLM_DIR)
        sys.path.insert(0, LLM_DIR)

    # Import directly from file path to avoid any sys.path confusion
    import importlib.util
    llm_infer_path = Path(LLM_DIR) / "infer.py"
    spec   = importlib.util.spec_from_file_location("llm_infer", llm_infer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    StyleRecommender       = module.StyleRecommender
    OllamaStyleRecommender = module.OllamaStyleRecommender

    if backend == "ollama":
        logger.info("Using Ollama backend for LLM …")
        _cached_recommender = OllamaStyleRecommender()
    else:
        model_dir = Path(model_path)
        if not model_dir.exists():
            # No LLM model — use fallback rule-based report
            logger.warning(
                "LLM model not found at '%s'. Using rule-based fallback report.", model_path
            )
            _cached_recommender = _RuleBasedRecommender()
        else:
            logger.info("Loading local LLM from %s …", model_path)
            _cached_recommender = StyleRecommender(model_path)

    return _cached_recommender


class _RuleBasedRecommender:
    """
    Fallback when no LLM model is available.
    Generates a structured style report from attributes alone.
    """

    def generate(self, enriched_input: Dict[str, Any], **_) -> str:
        attrs  = enriched_input.get("attributes", {})
        trends = enriched_input.get("current_trends", [])

        gender      = "man" if attrs.get("male") == "yes" else "woman"
        hair_color  = attrs.get("hair_color", "natural")
        facial_hair = attrs.get("facial_hair") == "yes"
        eyeglasses  = attrs.get("eyeglasses") == "yes"
        bald        = attrs.get("bald") == "yes"
        wavy        = attrs.get("wavy_hair") == "yes"
        straight    = attrs.get("straight_hair") == "yes"
        cheekbones  = attrs.get("high_cheekbones") == "yes"
        pale        = attrs.get("pale_skin") == "yes"

        hair_desc = "bald" if bald else f"{hair_color} {'wavy' if wavy else 'straight' if straight else ''} hair".strip()

        report = f"## 👤 Your Personalized Style Report\n\n"
        report += f"**Profile:** {gender.capitalize()} with {hair_desc}"
        if facial_hair:
            report += ", facial hair"
        if eyeglasses:
            report += ", eyeglasses"
        report += ".\n\n"

        report += "## 💇 Grooming Recommendations\n\n"
        if bald:
            report += "- Embrace the clean look — keep your scalp moisturized and well-maintained.\n"
            report += "- A well-groomed beard complements a bald head beautifully.\n"
        elif wavy:
            report += "- Use a light curl cream or mousse to define your natural wave pattern.\n"
            report += "- Avoid over-washing — 2-3 times per week keeps natural oils balanced.\n"
        elif straight:
            report += "- A matte pomade or light wax will keep your style clean and polished.\n"
            report += "- Consider a textured cut to add dimension to straight hair.\n"

        if facial_hair:
            report += "- Keep your facial hair trimmed and shaped regularly for a groomed appearance.\n"
            report += "- Use beard oil to maintain softness and shine.\n"
        else:
            report += "- A clean-shaven look benefits from a quality moisturizer post-shave.\n"

        if eyeglasses:
            report += "- Your frames are part of your style — choose a complementary shape to your face.\n"

        report += "\n## 👔 Style Recommendations\n\n"

        if pale:
            report += "- Rich, deep colors like navy, forest green, and burgundy work exceptionally well for your complexion.\n"
            report += "- Avoid washed-out pastels — opt for saturated hues.\n"
        else:
            report += "- Earth tones, warm neutrals, and classic navy are universally flattering for your complexion.\n"
            report += "- Don't shy away from bold colors — your complexion carries them well.\n"

        if cheekbones:
            report += "- Your strong bone structure suits structured pieces — blazers, tailored shirts, fitted jackets.\n"

        report += "- Invest in well-fitted basics: white tee, dark jeans, clean sneakers.\n"
        report += "- Layering adds depth — try an open button-down over a crew-neck tee.\n"

        if trends:
            report += "\n## 🔥 Current Trends for You\n\n"
            for i, trend in enumerate(trends[:3], 1):
                report += f"**Trend {i}:** {trend[:200]}{'...' if len(trend) > 200 else ''}\n\n"

        report += "\n---\n*Report generated by StyleAI Vision Pipeline*"
        return report
