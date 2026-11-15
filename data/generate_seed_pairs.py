"""
data/generate_seed_pairs.py
Synthetic Seed Data Generation

Generates 50 diverse (attribute_json → style_report) training pairs for
LLM fine-tuning. Output is saved as data/training_pairs/seed_pairs.jsonl
in HuggingFace chat format (list of messages).
"""

import json
import logging
import random
from itertools import product
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Attribute Universe ───────────────────────────────────────────────────────

ATTRIBUTES = {
    "face_shape":           ["oval", "round", "square", "heart", "oblong", "diamond"],
    "skin_tone":            ["fair", "light", "medium", "olive", "tan", "deep"],
    "hair_texture":         ["straight", "wavy", "curly", "coily"],
    "hair_length":          ["short", "medium", "long"],
    "facial_hair":          ["none", "stubble", "beard", "mustache"],
    "eye_shape":            ["almond", "round", "hooded", "monolid", "upturned"],
    "jaw_type":             ["soft", "defined", "sharp"],
    "cheekbone_prominence": ["low", "medium", "high"],
}

SYSTEM_PROMPT = (
    "You are a professional stylist and grooming advisor. "
    "Given a person's facial and style attributes, generate "
    "a detailed, encouraging, and personalized style report."
)

# ─── Lookup tables for report generation ─────────────────────────────────────

HAIRCUT_MAP: Dict[str, List[str]] = {
    "oval":    ["virtually any cut works — try a textured crop, curtain bangs, or a tapered fade",
                "slick-back undercut", "loose waves or layers"],
    "round":   ["angular cuts that add height, such as a pompadour or quiff",
                "side-swept styles", "faux hawk with volume on top"],
    "square":  ["softer styles like a textured fringe, side part, or shaggy layers",
                "wavy lob with curtain bangs", "undercut with a soft top"],
    "heart":   ["cuts that add width at the jaw — chin-length bobs, layered waves",
                "textured quiff", "medium shag cut"],
    "oblong":  ["cuts that add width, not height — curtain bangs, side-swept fringes",
                "medium-length curls", "blunt bob"],
    "diamond": ["styles that soften the forehead and jaw — bixie cut, soft waves",
                "layered lob", "side-part undercut with volume"],
}

SKINCARE_MAP: Dict[str, List[str]] = {
    "fair":   ["SPF 50+ daily sunscreen", "gentle hydrating cleanser",
               "vitamin C serum for brightness"],
    "light":  ["SPF 30–50 broad-spectrum", "niacinamide serum for even tone",
               "lightweight moisturizer"],
    "medium": ["SPF 30 daily", "glycolic acid exfoliant twice a week",
               "hyaluronic acid for hydration"],
    "olive":  ["SPF 30", "azelaic acid to keep the complexion even",
               "oil-balancing toner"],
    "tan":    ["SPF 30", "kojic acid or vitamin C for hyperpigmentation",
               "rich night cream"],
    "deep":   ["SPF 30 tinted", "shea butter or ceramide moisturizer",
               "glycolic acid to brighten without stripping"],
}

GROOMING_MAP: Dict[str, List[str]] = {
    "none":     ["clean shave every 2–3 days for a sharp look",
                 "beard balm on bare skin to stay moisturized"],
    "stubble":  ["trim with a 0.5–1 mm guard every 3 days",
                 "edge the neckline for a clean silhouette"],
    "beard":    ["weekly shape-up at a barber or with precision trimmer",
                 "daily beard oil to soften and condition",
                 "comb downward morning and night"],
    "mustache": ["define the upper lip line with scissors weekly",
                 "light mustache wax for control",
                 "keep the philtrum clean-shaved"],
}

PALETTE_MAP: Dict[str, List[str]] = {
    "fair":   ["navy, forest green, burgundy, ivory — cool undertones flatter you;",
               "avoid pure white or neon yellow"],
    "light":  ["dusty rose, slate blue, sage green, camel — soft, muted tones work beautifully;",
               "rich jewel tones for evenings"],
    "medium": ["terracotta, olive, warm white, cobalt blue — warm and earthy tones sing;",
               "avoid overly pale pastels"],
    "olive":  ["mustard yellow, rust, forest green, off-white — warm neutrals are ideal;",
               "gold jewellery over silver"],
    "tan":    ["coral, electric blue, cream, chocolate brown — bold hues pop on your tone;",
               "burnt orange and brick red are your secret weapons"],
    "deep":   ["cobalt, hot pink, ivory, caramel — high contrast looks stunning;",
               "all-white or all-black are effortless wins"],
}

STYLE_MAP: Dict[str, str] = {
    "oval":    "classic and versatile — you can carry structured tailoring or relaxed streetwear equally well.",
    "round":   "elongated silhouettes and vertical lines work best — try slim trousers with a longline jacket.",
    "square":  "softer, textured fabrics balance your strong features — try linen, jersey, and relaxed-fit pieces.",
    "heart":   "A-line or flared bottoms that balance wider shoulders — wide-leg trousers or bootcut jeans are your ally.",
    "oblong":  "layer to create horizontal interest — chunky knits, belted jackets, and cropped tops.",
    "diamond": "tops that widen the shoulder and hem, such as off-shoulder or wide-collar pieces, balance the look beautifully.",
}

TREND_TEMPLATE = """\
Based on your profile, here are three current trend picks that suit you perfectly:

1. **{t1}** — complements your {attr1} and works with your {skin_tone} skin tone.
2. **{t2}** — a strong match for your {face_shape} face shape and {hair_texture} hair.
3. **{t3}** — a versatile piece that anchors your color palette and style direction."""

CURRENT_TRENDS = [
    ("Quiet Luxury (clean cashmere, tailored trousers)",
     "Y2K Streetwear (low-rise, graphic tees, chunky sneakers)",
     "Gorpcore (technical fabrics, utility vests, trail runners)",
     "Coastal Grandmother (linen blazers, wide-brimmed hats)",
     "Dark Academia (tweed, turtlenecks, Oxford shoes)",
     "Dopamine Dressing (bold color blocking, statement accessories)",
     "Workwear Chic (oversized blazers, wide-leg trousers)",
     "Vintage Revival (thrifted denim, band tees, retro sneakers)"),
]


def random_attributes() -> Dict[str, str]:
    """Generate a random but valid attribute combination."""
    return {attr: random.choice(vals) for attr, vals in ATTRIBUTES.items()}


def build_report(attrs: Dict[str, str]) -> str:
    """
    Produce a structured markdown style report from a given attribute dict.
    Each call produces slightly varied phrasing to diversify the training set.
    """
    fs  = attrs["face_shape"]
    st  = attrs["skin_tone"]
    ht  = attrs["hair_texture"]
    hl  = attrs["hair_length"]
    fh  = attrs["facial_hair"]
    es  = attrs["eye_shape"]
    jt  = attrs["jaw_type"]
    cp  = attrs["cheekbone_prominence"]

    haircuts = HAIRCUT_MAP.get(fs, ["a versatile medium cut"])
    skincare = SKINCARE_MAP.get(st, ["SPF 30", "a gentle cleanser", "light moisturizer"])
    grooming = GROOMING_MAP.get(fh, ["regular upkeep"])
    palette  = PALETTE_MAP.get(st, ["neutral tones"])
    style_dir = STYLE_MAP.get(fs, "any silhouette works for you.")

    trends_pool = random.choice(CURRENT_TRENDS)
    t1, t2, t3 = random.sample(trends_pool, 3)

    report = f"""\
## Face Shape Analysis

Your face shape is **{fs}**, paired with a **{jt}** jawline and **{cp}** cheekbone prominence. \
Your **{es}** eyes are a defining feature. \
{"This combination gives you remarkable versatility in styling." if fs == "oval"
 else f"Styling choices that play to your {fs} shape's natural strengths will give you the most polished result."}

## Recommended Haircuts

Your {ht} hair at {hl} length, combined with your {fs} face shape, opens up several great options:
- {haircuts[0].capitalize()}.
- {haircuts[1].capitalize() if len(haircuts) > 1 else "A classic taper also works."}
{"- " + haircuts[2].capitalize() + "." if len(haircuts) > 2 else ""}

Ask your barber/stylist to work *with* your natural {ht} texture rather than against it — \
this dramatically cuts styling time and improves the result.

## Skincare & Grooming Tips

**Skincare (for {st} skin tone):**
- {skincare[0].capitalize()}.
- {skincare[1].capitalize()}.
- {skincare[2].capitalize()}.

**Grooming ({fh} facial hair):**
- {grooming[0].capitalize()}.
{"- " + grooming[1].capitalize() + "." if len(grooming) > 1 else ""}
{"- " + grooming[2].capitalize() + "." if len(grooming) > 2 else ""}

Consistency beats intensity — a 3-minute routine every morning will outperform an hour-long weekly treatment.

## Color Palette (Clothing Colors That Suit Your Skin Tone)

Your {st} skin tone looks best in: **{palette[0]}**

{palette[1] if len(palette) > 1 else ""}

**Quick rule:** when in doubt, reach for a tonal monochrome outfit — it's always sharp and always flattering.

## Fashion Style Direction

Your face structure lends itself to {style_dir}

Key principles for your body of work:
- **Fit is everything** — even budget pieces look expensive when tailored correctly.
- Build a capsule wardrobe around 3 neutral base colors, then accent with 1–2 bolder tones from your palette.
- Invest in quality basics: a structured blazer, dark indigo jeans, and clean white leather sneakers.

## Current Trend Picks That Suit You

{TREND_TEMPLATE.format(
    t1=t1, t2=t2, t3=t3,
    attr1="eye shape", skin_tone=st,
    face_shape=fs, hair_texture=ht,
)}
"""
    return report.strip()


def build_chat_pair(attrs: Dict[str, str]) -> dict:
    """
    Build a single HuggingFace chat-format training pair.

    Returns:
        {
          "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "<json>"},
            {"role": "assistant", "content": "<report>"},
          ]
        }
    """
    user_content  = json.dumps(attrs, indent=2)
    asst_content  = build_report(attrs)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst_content},
        ]
    }


def generate_diverse_samples(n: int = 50) -> List[dict]:
    """
    Generate n samples, ensuring meaningful variation across attributes.

    Strategy:
    - First 30 samples: fully random
    - Last 20 samples: systematically cycle through face shapes × skin tones
    """
    samples = []
    random.seed(42)

    # Random samples
    for _ in range(min(30, n)):
        attrs = random_attributes()
        samples.append(build_chat_pair(attrs))

    # Systematic coverage samples
    face_shapes = ATTRIBUTES["face_shape"]
    skin_tones  = ATTRIBUTES["skin_tone"]
    combos = list(product(face_shapes[:4], skin_tones[:5]))  # 20 combos
    random.shuffle(combos)

    for fs, st in combos[:max(0, n - 30)]:
        attrs = random_attributes()
        attrs["face_shape"] = fs
        attrs["skin_tone"]  = st
        samples.append(build_chat_pair(attrs))

    return samples[:n]


def main() -> None:
    output_path = Path("data/training_pairs/seed_pairs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating 50 synthetic training pairs …")
    samples = generate_diverse_samples(50)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Saved %d pairs to %s", len(samples), output_path)

    # Quick sanity check — print first example
    logger.info("\n── Sample pair 0 ──")
    first = samples[0]
    logger.info("User (attributes):\n%s", first["messages"][1]["content"])
    logger.info("\nAssistant (report excerpt):\n%s",
                first["messages"][2]["content"][:600] + " …")


if __name__ == "__main__":
    main()
