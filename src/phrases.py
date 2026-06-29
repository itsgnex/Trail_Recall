import random

_last = {}

_PROMPTS = {
    "plant": [
        "That looks like it may be a plant. Would you like me to tell you about it?",
        "I think you may be looking at a plant. Should I try to identify it?",
        "This appears to be a plant. Would you like a quick explanation?",
        "I can take a closer look at this plant if you want.",
        "Would you like me to check what plant this might be?",
        "I may be seeing a plant here. Should I try to identify it?",
        "Would it help if I told you what this plant might be?",
        "I can try to recognize this plant for you.",
        "This looks plant-like. Would you like me to explain what I can see?",
        "Should I look this plant up for you?",
        "I can check this plant and give you a short answer.",
        "Would you like a quick description of this plant?",
        "I think this may be a plant. Should I explain it?",
        "I can try to identify this plant from the image.",
        "Do you want me to tell you what this plant might be?",
    ],
    "sign": [
        "That looks like it may be a sign. Would you like me to read or explain it?",
        "I think you may be looking at a sign. Should I help with it?",
        "This appears to be a sign. Would you like me to explain what it means?",
        "I can take a closer look at this sign if you want.",
        "Would you like me to read this sign for you?",
        "Should I explain what this sign is saying?",
        "I may be seeing a sign here. Would you like help with it?",
        "I can try to read or interpret this sign.",
        "Would it help if I explained this sign?",
        "This looks like a sign or symbol. Should I check it for you?",
        "I can try to tell you what this sign means.",
        "Would you like a quick explanation of this sign?",
        "I think this may be a sign. Should I read it?",
        "Do you want me to interpret this sign?",
        "I can check this sign and give you a short answer.",
    ],
    "unclear": [
        "I’m not fully sure what this is. Would you like me to describe what I can see?",
        "This is a little unclear. Should I take a closer look?",
        "I may be seeing something in front of you. Would you like help with it?",
        "I’m not certain what this object is. Would you like a quick description?",
        "Would you like me to explain what I can make out?",
        "This image is not very clear yet. Should I still try?",
        "I can try to describe this, but I may need a clearer view.",
        "Would you like me to take a guess based on what I can see?",
        "I’m not sure about this one. Should I explain what appears in the image?",
        "This may need a closer look. Would you like me to try?",
        "I can give a short description if that would help.",
        "Would you like me to check what this might be?",
        "I can try to understand this image for you.",
        "This is not clear enough yet, but I can still try to describe it.",
        "Should I look at this more carefully?",
    ],
}

_CANCEL = [
    "All right, I’ll wait.",
    "Okay, I won’t continue.",
    "No problem.",
    "All right, I’ll stay quiet for now.",
    "Okay, let me know if you need help.",
    "Understood.",
    "Sure, I’ll stop here.",
    "Okay, I’ll leave it for now.",
    "No problem, I’ll wait.",
    "All right.",
]

_REPEAT = [
    "Sure, I’ll say that again.",
    "Of course, I’ll repeat it.",
    "I’ll repeat that.",
    "Sure, here it is again.",
    "No problem, I’ll say it again.",
]

_CLARIFY = [
    "I’m sorry, I wasn’t sure what you wanted. Would you like me to help with this?",
    "I didn’t quite understand. Should I explain what you’re looking at?",
    "I may have missed that. Would you like me to continue?",
    "Could you say that again? I can help if you want.",
    "I’m not sure if you wanted me to continue. Should I explain it?",
]

_UNCLEAR_SIGN = [
    "I can see what may be a sign, but I cannot read it clearly. Please hold it a little steadier.",
    "The sign is not clear enough for me to read. Try moving a little closer.",
    "I can tell there may be a sign, but the text or symbol is hard to see.",
    "I’m not confident about this sign yet. Please hold it straighter or closer.",
    "The sign looks unclear from this angle. A steadier view would help.",
    "I cannot read or interpret this sign clearly yet. Try centering it in the box.",
    "This may be a sign, but I need a clearer view before explaining it.",
    "The sign is too blurry or far away for a reliable answer.",
    "I can try again if you hold the sign more steady.",
    "I’m not able to read this sign clearly from the current image.",
]

_UNCLEAR_PLANT = [
    "This appears to be a plant, but I’m not certain of the exact type from this image.",
    "It looks like a plant, but I need a clearer view of the leaves or flowers to identify it.",
    "This may be a plant, but I cannot identify the exact type confidently yet.",
    "I can see plant-like features, but the image is not clear enough for a reliable name.",
    "Try centering the leaves or flowers more clearly, and I can check again.",
    "This looks like a plant, but I do not want to guess the exact species.",
    "I’m not fully sure what plant this is from the current view.",
    "I may need a closer image of the leaves or flowers to identify this plant.",
    "The plant is visible, but the exact type is uncertain.",
    "I can describe what I see, but I cannot name the plant confidently yet.",
]


def _pick(category, phrases):
    last = _last.get(category)
    choices = [phrase for phrase in phrases if phrase != last] or phrases
    phrase = random.choice(choices)
    _last[category] = phrase
    return phrase


def get_prompt(kind):
    return _pick(kind, _PROMPTS.get(kind, _PROMPTS["unclear"]))


def get_cancel_response():
    return _pick("cancel", _CANCEL)


def get_repeat_response():
    return _pick("repeat", _REPEAT)


def get_clarification_response():
    return _pick("clarify", _CLARIFY)


def get_unclear_sign_response():
    return _pick("unclear_sign", _UNCLEAR_SIGN)


def get_unclear_plant_response():
    return _pick("unclear_plant", _UNCLEAR_PLANT)
