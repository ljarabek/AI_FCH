def popravi_sumnike(s: str) -> str:
    s = s.replace("š", "s")
    s = s.replace("ž", "z")
    s = s.replace("č", "c")
    s = s.replace("ć", "c")

    return s
