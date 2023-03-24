import enum

LABELS = [
    "NOTHING",
    "CARRY",
    "PASS_L",
    "PASS_R",
    "KICK_L",
    "KICK_R",
    "RUCK",
    "TACKLE_S_D", ## Tackle, Single, Dominant
    "TACKLE_S", ## Tackle, Single
    "TACKLE_D_D", ## Tackle, Double, Dominant
    "TACKLE_D", ## Tackle, Double
    "TACKLE_M", ## Tackle, Missed
    "LINEOUT",
    "SCRUM",
    "MAUL"    
]

Label = enum.Enum("Label", LABELS)
NUM_CLASSES = len(LABELS)

SEPARATOR_CHAR = ":"

LabelMapper = {}
for label, enum in zip(LABELS, Label):
    LabelMapper[f"Label.{label}"] = enum

def load_from_file(path: str):

    labels = []

    with open(path, "rt") as f:
        for line in f:
            if line.strip():
                label, num = line.split(SEPARATOR_CHAR)

                labels += [LabelMapper[label]] * int(num)

    return labels
