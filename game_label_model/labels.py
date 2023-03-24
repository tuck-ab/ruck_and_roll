import enum

LABELS_RAW = [
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

LABELS = [
    "NOTHING",
    "CARRY",
    "PASS",
    "KICK",
    "RUCK",
    "TACKLE",
    "LINEOUT",
    "SCRUM",
    "MAUL"
]

TRANSLATOR = {
    "NOTHING": "NOTHING",
    "CARRY": "CARRY",
    "PASS_L": "PASS",
    "PASS_R": "PASS",
    "KICK_L": "KICK",
    "KICK_R": "KICK",
    "RUCK": "RUCK",
    "TACKLE_S_D": "TACKLE",
    "TACKLE_S": "TACKLE",
    "TACKLE_D_D": "TACKLE",
    "TACKLE_D": "TACKLE",
    "TACKLE_M": "TACKLE",
    "LINEOUT": "LINEOUT",
    "SCRUM": "SCRUM",
    "MAUL": "MAUL"    
}



Label = enum.Enum("Label", LABELS)
NUM_CLASSES = len(LABELS)

SEPARATOR_CHAR = ":"

LabelMapper = {}
for label, enum in zip(LABELS, Label):
    LabelMapper[f"Label.{label}"] = enum


def load_from_file(path: str, expanded=True):

    labels = []

    with open(path, "rt") as f:
        for line in f:
            if line.strip():
                label, num = line.split(SEPARATOR_CHAR)

                if expanded:
                    labels += [LabelMapper[TRANSLATOR[label]]] * int(num)
                else:
                    labels.append((LabelMapper[TRANSLATOR[label]], int(num)))

    return labels
