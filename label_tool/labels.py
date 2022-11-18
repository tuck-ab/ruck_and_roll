import enum

LABELS = [
    "NOTHING", 
    "FORCEFUL_TACKLE", 
    "ABSORBING_TACKLE",
    "OTHER_TACKLE", 
    "RUCK", 
    "MAUL", 
    "LINEOUT", 
    "SCRUM"
]

Label = enum.Enum("Label", LABELS)

mapper = {}
for label, enum in zip(LABELS, Label):
    mapper[f"Label.{label}"] = enum
