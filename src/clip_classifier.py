import cv2

_model = None
_preprocess = None
_tokenizer = None
_torch = None
_prompts = [
    ("plant", "a clear photo of a plant"),
    ("plant", "a flower or leafy plant"),
    ("plant", "a tree or garden plant"),
    ("sign", "a sign with text"),
    ("sign", "a warning sign"),
    ("sign", "a printed label or poster"),
    ("other", "a normal indoor object"),
    ("other", "a wall or background"),
    ("other", "nothing important"),
]
_warned = False
_disabled = False


def classify_crop_with_clip(crop):
    global _model, _preprocess, _tokenizer, _torch, _warned, _disabled
    if _disabled:
        return None
    try:
        if _model is None:
            import open_clip
            import torch
            from PIL import Image

            _torch = torch
            _model, _, _preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
            _tokenizer = open_clip.get_tokenizer("ViT-B-32")
            _model.eval()
        else:
            from PIL import Image

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = _preprocess(Image.fromarray(rgb)).unsqueeze(0)
        text = _tokenizer([prompt for _, prompt in _prompts])
        with _torch.no_grad():
            image_features = _model.encode_image(image)
            text_features = _model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

        scores = {"plant": 0.0, "sign": 0.0, "other": 0.0}
        for i, (kind, _) in enumerate(_prompts):
            scores[kind] = max(scores[kind], float(probs[i]))
        kind = max(scores, key=scores.get)
        confidence = scores[kind]
        if confidence < 0.35:
            return "other", "clip confidence low"
        return kind, f"clip confidence {confidence:.2f}"
    except Exception as exc:
        if not _warned:
            print(f"CLIP unavailable: {exc}. Falling back to OCR and simple image heuristics.")
            _warned = True
        _disabled = True
        return None
