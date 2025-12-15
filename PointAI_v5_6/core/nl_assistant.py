class NaturalLanguageAssistant:
    def __init__(self):
        self.mode = "rules"

    def set_mode(self, mode: str):
        self.mode = mode

    def translate_to_command(self, text: str) -> str | None:
        t = text.lower().strip()

        # ----- POINT SIZE -----
        if "punto" in t or "point" in t:
            for tok in t.split():
                if tok.replace(".", "", 1).isdigit():
                    return f"pointsize {tok}"

        # ----- ROI -----
        if "roi" in t or "raggio" in t:
            for tok in t.split():
                if tok.replace(".", "", 1).isdigit():
                    return f"roi {tok}"

        # ----- EXPORT -----
        if "export" in t or "esporta" in t:
            return "export"

        # ----- REMOVE -----
        if "rimuovi" in t or "cancella" in t:
            return "removebbox"

        return None
