"""
LLM Safety Classifier using Opir-multitask-large-v1.0.

Uses GLiClass (DeBERTaV3-large) for zero-shot multi-label safety classification
across 17 categories (16 risk labels + safe).
"""

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

TOP_LEVEL_SAFETY_LABELS = [
    "toxicity",
    "violence_and_physical_harm",
    "self_harm_and_suicide",
    "sexual_content",
    "child_safety",
    "personal_information_privacy_and_intellectual_property",
    "cybersecurity",
    "criminal_and_illegal_activity",
    "regulated_goods_and_advice",
    "biological_medical_and_environmental_harm",
    "weapons_of_mass_destruction",
    "information_integrity_and_manipulation",
    "ai_system_security_and_reliability",
    "bias_fairness_and_representation",
    "other_or_uncertain",
    "safe_and_benign",
]

SEVERITY_MAP = {
    "toxicity": "low",
    "violence_and_physical_harm": "high",
    "self_harm_and_suicide": "high",
    "sexual_content": "low",
    "child_safety": "high",
    "personal_information_privacy_and_intellectual_property": "medium",
    "cybersecurity": "medium",
    "criminal_and_illegal_activity": "high",
    "regulated_goods_and_advice": "high",
    "biological_medical_and_environmental_harm": "high",
    "weapons_of_mass_destruction": "high",
    "information_integrity_and_manipulation": "medium",
    "ai_system_security_and_reliability": "high",
    "bias_fairness_and_representation": "medium",
    "other_or_uncertain": "medium",
}


class SafetyClassifier:
    """Zero-shot multi-label safety classifier for LLM prompts.

    Uses GLiClass/DeBERTaV3-large to classify text into 17 safety categories.
    The highest-scoring risk label becomes the primary category; remaining scores
    are recorded in tags.
    """

    def __init__(self, model_path: str, device: str = "cpu", threshold: float = 0.5):
        self.threshold = threshold
        self.model = GLiClassModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipeline = ZeroShotClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            classification_type="multi-label",
            device=device,
        )
        self.max_length = getattr(self.tokenizer, "model_max_length", 512)
        # DeBERTa effective max; cap at a reasonable value if the config is absurdly large
        if self.max_length > 10000:
            self.max_length = 512

    def _build_tags(self, scores_list: list) -> str:
        """Format all label scores into a single tags string, sorted by score desc."""
        score_strs = []
        for item in sorted(scores_list, key=lambda x: x["score"], reverse=True):
            score_strs.append(f"{item['label']}={item['score']:.2f}")
        return "scores: " + ", ".join(score_strs)

    def classify(self, prompt: str) -> dict:
        """Classify a single prompt.

        Returns:
            dict with keys: category (str), severity (str), tags (str)
        """
        # Handle empty / whitespace-only prompts
        if not prompt or not prompt.strip():
            return {
                "category": "safe",
                "severity": "safe",
                "tags": "",
            }

        # Truncate long prompts to model max length
        if len(prompt) > self.max_length * 4:  # rough char estimate
            tokens = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_length)
            prompt = self.tokenizer.decode(tokens, skip_special_tokens=True)

        try:
            results = self.pipeline(
                prompt, TOP_LEVEL_SAFETY_LABELS, threshold=self.threshold
            )[0]
        except Exception:
            # If classification fails (e.g. OOM), return safe as fallback
            return {
                "category": "safe",
                "severity": "safe",
                "tags": "",
            }

        # Separate risk labels from safe_and_benign
        detected = [
            item for item in results
            if item["score"] >= self.threshold and item["label"] != "safe_and_benign"
        ]

        if not detected:
            return {
                "category": "safe",
                "severity": "safe",
                "tags": self._build_tags(results),
            }

        # Top-scoring risk label is the primary category
        detected.sort(key=lambda x: x["score"], reverse=True)
        top_label = detected[0]["label"]

        return {
            "category": top_label,
            "severity": SEVERITY_MAP.get(top_label, "medium"),
            "tags": self._build_tags(results),
        }

    def predict_batch(self, prompts: list, batch_size: int = 16) -> list:
        """Classify a batch of prompts.

        TODO: Investigate whether gliclass ZeroShotClassificationPipeline supports
        passing multiple texts at once for true batching. Currently uses a loop.
        """
        results = []
        for prompt in prompts:
            results.append(self.classify(prompt))
        return results
