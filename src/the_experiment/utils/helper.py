######################################################################
# HELPER: Convert booleans to text or emojis, depending on DISPLAY_MODE
######################################################################
DISPLAY_MODE = "tf"  # or "bits" or "emoji", updated by user radio buttons


def bool_str(b, mode="bits"):
    if mode == "bits":
        return "1" if b else "0"
    elif mode == "emoji":
        return "✅" if b else "❌"
    else:  # "tf"
        return "True" if b else "False"
