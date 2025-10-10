import csv, os

class Logger:
    def __init__(self, filepath="results/results.csv"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hour", "wind_speed", "line_status", "ENS"])

    def log(self, hour, wind_speed, line_status, ens):
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow([hour, wind_speed, line_status, ens])
