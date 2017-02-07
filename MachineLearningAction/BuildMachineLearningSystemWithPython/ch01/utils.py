import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data");

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "chart");

for dir in [DATA_DIR, CHART_DIR]:
    if (not os.path.exists(dir)):
        os.mkdir(dir)
