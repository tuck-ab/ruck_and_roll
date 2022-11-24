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

LABELS_OLD = {
    "NOTHING", 
    "FORCEFUL_TACKLE", 
    "ABSORBING_TACKLE",
    "OTHER_TACKLE", 
    "RUCK", 
    "MAUL", 
    "LINEOUT", 
    "SCRUM"
}

Label = enum.Enum("Label", LABELS)

mapper = {}
for label, enum in zip(LABELS, Label):
    mapper[f"Label.{label}"] = enum
